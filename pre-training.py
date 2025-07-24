import os
import time
import pathlib
import argparse

import dgl
import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.optim as optim

from model.transformer import multiheaded
from model.alignn import EdgeGatedGraphConv
from model.graphformer import Graphformer, encoder
from utils.datasets import StructureDataset
from utils.masker import MaskAtom

from torch.nn import Linear
from lightning.fabric import Fabric
from torch.utils.data import DataLoader
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import FSDPStrategy


def pre_train(args, loader, main_model, atom_model, main_optim, atom_optim, criterion, fabric: Fabric):
    main_model.train(), atom_model.train()
    main_optim.zero_grad(), atom_optim.zero_grad()
    epoch_loss = torch.zeros(2).to(fabric.local_rank)

    for iteration, (graphs, lg, fg, _) in enumerate(loader):

        is_accumulating = iteration % args.n_cum != 0

        g = graphs[0]
        nsg = graphs[1]
        node_idxs = nsg.ndata[dgl.NID].tolist()
        node_truths = nsg.ndata['node_feats']

        with fabric.no_backward_sync(main_model, enabled=is_accumulating), fabric.no_backward_sync(atom_model, enabled=is_accumulating):
            _, node_rep, _, _, _ = main_model(g, lg, fg)
            pred_node = atom_model(node_rep[node_idxs])
            loss = criterion(pred_node, node_truths)
            fabric.backward(loss)

        if not is_accumulating:
            main_optim.step()
            main_optim.zero_grad()
            atom_optim.step()
            atom_optim.zero_grad()

        # Save Loss
        epoch_loss[0] += (loss.item() * args.n_cum)
        epoch_loss[1] += args.batch_size

    fabric.all_reduce(epoch_loss, reduce_op='sum')
    epoch_loss = epoch_loss[0] / epoch_loss[1]
    fabric.print('Epoch loss: %.4f' % epoch_loss)
    return main_model, atom_model, main_optim, atom_optim, epoch_loss


def validate(args, loader, main_model, atom_model, criterion, fabric):
    main_model.eval(), atom_model.eval()
    epoch_loss = torch.zeros(2).to(fabric.local_rank)
    epoch_matches = torch.empty(0).to(fabric.local_rank)

    for graphs, lg, fg, _ in loader:

        g = graphs[0]
        nsg = graphs[1]
        node_idxs = nsg.ndata[dgl.NID].tolist()
        node_truths = nsg.ndata['node_feats']

        with torch.no_grad():
            _, node_rep, _, _, _ = main_model(g, lg, fg)
            pred_node = atom_model(node_rep[node_idxs])
            loss = criterion(pred_node, node_truths)

        # Save Loss
        epoch_loss[0] += (loss.item() * args.n_cum)
        epoch_loss[1] += args.batch_size

        # Get accuracy of current iteration
        pred_atoms = (torch.sigmoid(pred_node) > 0.5).float()
        correct_atoms = torch.all(pred_atoms == node_truths, dim=1)
        epoch_matches = torch.cat((epoch_matches, correct_atoms), dim=0)

    # Get overall accuracy accross all iterations
    accuracy = epoch_matches.to(torch.float32).mean()
    fabric.print(f'Validation accuracy: {accuracy}')

    # Get overall loss and return it
    fabric.all_reduce(epoch_loss, reduce_op='sum')
    epoch_loss = epoch_loss[0] / epoch_loss[1]
    return epoch_loss, accuracy


def main(args):

    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)

    # ------------------------------------- FABRIC SETUP -------------------------------------
    logger = CSVLogger(
        root_dir=args.save_dir,
        name=args.run_name,
        flush_logs_every_n_steps=1
    )
    policy = {encoder, EdgeGatedGraphConv, multiheaded}
    fsdp_strategy = FSDPStrategy(auto_wrap_policy=policy, activation_checkpointing_policy=policy, state_dict_type='full')
    if args.accelerator == 'cpu' or args.accelerator == 'mps':
        fabric = Fabric(accelerator=args.accelerator, devices=args.n_devices, num_nodes=args.n_nodes, loggers=logger)
    elif args.accelerator == 'gpu' or args.accelerator == 'cuda':
        fabric = Fabric(accelerator=args.accelerator, devices=args.n_devices, num_nodes=args.n_nodes, strategy=fsdp_strategy, loggers=logger)
    else:
        fabric = Fabric(accelerator='auto', devices=args.n_devices, num_nodes=args.n_nodes, loggers=logger)
    fabric.launch()

    # ------------------------------------- DATASET SETUP -------------------------------------
    data = StructureDataset(args, transform=MaskAtom(
        num_atom_fea=args.num_atom_fea, node_feat_name='node_feats', mask_rate=args.mask_rate
    ))
    training_data, validation_data = torch.utils.data.random_split(data, [args.train_split, args.val_split])
    training_loader = DataLoader(training_data, collate_fn=data.collate_pre, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, collate_fn=data.collate_pre, batch_size=args.batch_size, shuffle=True)
    training_loader, validation_loader = fabric.setup_dataloaders(training_loader), fabric.setup_dataloaders(validation_loader)

    # ------------------------------------- MODEL, OPTIMIZER AND LOSS/ERROR FUNCTION SETUP -------------------------------------
    main_model = Graphformer(args=args)
    main_model.freeze_pretrain()
    atom_model = nn.Sequential(Linear(args.hidden_dims, args.num_atom_fea), nn.Sigmoid())
    main_model, atom_model = fabric.setup_module(main_model), fabric.setup_module(atom_model)

    main_optim = optim.Adam(filter(lambda p: p.requires_grad, main_model.parameters()), lr=args.lr, weight_decay=args.decay)
    atom_optim = optim.Adam(atom_model.parameters(), lr=args.lr, weight_decay=args.decay)
    main_optim, atom_optim = fabric.setup_optimizers(main_optim), fabric.setup_optimizers(atom_optim)

    if args.load_model == 1:
        # Check if there are model checkpoints
        pt_main_path = osp.join(args.model_path, f'mm_checkpoint.{args.begin_epoch}epochs.ckpt')
        pt_atom_path = osp.join(args.model_path, f'am_checkpoint.{args.begin_epoch}epochs.ckpt')
        assert osp.exists(pt_main_path) and osp.exists(pt_atom_path), f'No models checkpoint for epoch {args.begin_epoch} exist in path {str(args.model_path)}'
        # Load models
        main_state = {'model': main_model, 'optim_state': main_optim}
        fabric.load(pt_main_path, state=main_state)
        atom_state = {'node_model': atom_model, 'node_optim': atom_optim}
        fabric.load(pt_atom_path, state=atom_state)

    criterion = nn.BCEWithLogitsLoss()

    fabric.print('-------------------- Pre-Training Started --------------------', flush=True)
    lowest_error = np.inf
    per_epoch_times = []
    start_time = time.time()

    for epoch in range(args.begin_epoch, args.epochs + 1):
        # -------------------- TRAINING --------------------
        training_start_time = time.time()
        main_model, atom_model, main_optim, atom_optim, epoch_loss = pre_train(args, training_loader, main_model, atom_model, main_optim, atom_optim, criterion, fabric)
        fabric.print(f'Training time: {time.time() - training_start_time} seconds')

        # ------------------- VALIDATION -------------------
        validation_start_time = time.time()
        epoch_error, epoch_accuracy = validate(args, validation_loader, main_model, atom_model, criterion, fabric)
        fabric.print(f'Validation time: {time.time() - validation_start_time} seconds')

        # ------------------- LOG RESULTS ------------------
        per_epoch_time = time.time() - training_start_time
        fabric.print('Completed Epoch %d of %d in %.2f s' % (epoch, args.epochs, per_epoch_time), flush=True)
        per_epoch_times.append(per_epoch_time)

        fabric.log_dict({'Pre-Training Loss': epoch_loss, 'Pre-Training Error': epoch_error, 'Pre-Training Accuracy': epoch_accuracy},  step=epoch)

        # ----------------- CHECKPOINT MODEL ---------------
        if epoch % 10 == 0:
            main_state = {'model': main_model, 'optim_state': main_optim}
            fabric.save(osp.join(args.model_path, f'mm_checkpoint.{epoch}epochs.ckpt'), main_state)
            atom_state = {'node_model': atom_model, 'node_optim': atom_optim}
            fabric.save(osp.join(args.model_path, f'am_checkpoint.{epoch}epochs.ckpt'), atom_state)

        # ------------- SAVE LOWEST ERROR MODEL ------------
        if epoch_error < lowest_error:
            lowest_error = epoch_error
            main_state = {'model': main_model, 'optim_state': main_optim}
            fabric.save(osp.join(args.model_path, args.pretrain_model), main_state)

    fabric.print(f'Average per epoch time: {np.mean(per_epoch_times)} seconds, Total {args.epochs} epochs time: {time.time() - start_time} seconds')
    fabric.print('-------------------- Pre-Training Finished --------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implementation of Pre-Training strategy for the Molecular Graph Transformer")
    # Fabric Arguments
    parser.add_argument('--n_devices', type=int, default=8, help='number of gpus/cpus that the code has access to (default: 8)')
    parser.add_argument('--n_nodes', type=int, default=1, help='number of nodes/computers on which the model is being trained (default: 1)')
    parser.add_argument('--accelerator', type=str, default='cuda', choices=['cpu', 'gpu', 'mps', 'cuda', 'tpu'],
                        help='device type on which the training is happening [cpu, gpu, mps (apple M1/M2 only), cuda (NVIDIA GPUs only), tpu] (default: cuda)')
    # Save and Load Arguments
    parser.add_argument('--root', type=str, help='root directory for all datasets (default: None)', required=True)
    parser.add_argument('--model_path', type=str, help='directory in which to save the trained model', required=True)
    parser.add_argument('--run_name', type=str, default=None, help='name of run for logging purposes')
    parser.add_argument('--save_dir', type=str, default=None, help='directory in which to save the test results')
    parser.add_argument('--pretrain_model', type=str, default='pretrain.ckpt', help='name with which to save the pretrained model')
    parser.add_argument('--load_model', type=int, default=0, choices=[0, 1], help='whether there is a model to be loaded (0: no model loading, 1: load checkpoint)')
    parser.add_argument('--n_ckpt', type=int, default=10, help='number of epochs to wait before saving a checkpoint of the model (default: 10)')
    # Training Arguments
    parser.add_argument('--n_cum', type=int, default=8, help='number of batches to accumulate the error for')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for training (default: 2)')
    parser.add_argument('--train_split', type=float, default=0.8, help='number of items in dataset to be used for training (default: 0.8)')
    parser.add_argument('--val_split', type=float, default=0.2, help='number of items in dataset to be used for validation (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for (default: 100)')
    parser.add_argument('--begin_epoch', type=int, default=1, help='set to restart training from a specific epoch')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--decay', type=float, default=1e-5, help='weight decay for the optimizers (default: 1e-5)')
    # Model Arguments
    parser.add_argument('--process', type=int, default=1, choices=[0, 1], help='whether the graphs for the structures/molecules need to be created during dataset loading (default: True)')
    parser.add_argument('--max_nei_num', type=int, default=12, help='maximum number of neighbour allowed for each atom in the local graph (default: 12)')
    parser.add_argument('--local_radius', type=int, default=8, help='radius used to form the local graph (default: 8)')
    parser.add_argument('--periodic', type=int, default=1, choices=[0, 1], help='whether the input structure is a periodic structure or not (default: True)')
    parser.add_argument('--periodic_radius', type=int, default=12, help='radius used to form the fully connected graph (default: 12)')
    parser.add_argument('--num_atom_fea', type=int, default=90, help='length of feature vector for atoms (default: 90)')
    parser.add_argument('--num_edge_fea', type=int, default=1, help='length of feature vector for edges in local graph (default: 1)')
    parser.add_argument('--num_angle_fea', type=int, default=1, help='length of feature vector for edges in line graph (default: 1)')
    parser.add_argument('--num_pe_fea', type=int, default=10, help='length of feature vector for atom\'s positional encoding (default: 10)')
    parser.add_argument('--num_clmb_fea', type=int, default=1, help='length of feature vector for edges in fully connected graph (default: 1)')
    parser.add_argument('--num_edge_bins', type=int, default=80, help='number of bins for RBF expansion of edges in local graph (default: 80)')
    parser.add_argument('--num_angle_bins', type=int, default=40, help='number of bins for RBF expansion of edges in line graph (default: 40)')
    parser.add_argument('--num_clmb_bins', type=int, default=120, help='number of bins for RBF expansion of edges in fully connected graph (default: 120)')
    parser.add_argument('--embedding_dims', type=int, default=128, help='dimension of embedding layer (default: 128)')
    parser.add_argument('--hidden_dims', type=int, default=512, help='dimensions of each hidden layer (default: 512)')
    parser.add_argument('--out_dims', type=int, default=3, help='length of output vector of the network (default: 3)')
    parser.add_argument('--num_layers', type=int, default=1, help='number of encoders in the network (default: 1)')
    parser.add_argument('--n_mha', type=int, default=1, help='number of attention layers in each encoder (default: 1)')
    parser.add_argument('--n_alignn', type=int, default=3, help='number of graph convolutions in each encoder (default: 3)')
    parser.add_argument('--n_gnn', type=int, default=3, help='number of graph convolutions in each encoder (default: 3)')
    parser.add_argument('--n_heads', type=int, default=4, help='number of attention heads (default: 4)')
    parser.add_argument('--residual', type=int, default=1, choices=[0, 1], help='whether to add residuality to the network or not (default: True)')
    parser.add_argument('--mask_rate', type=float, default=0.2, help='percentage of node to be masked (default: 0.2)')

    args = parser.parse_args()

    args.residual = bool(args.residual)
    args.periodic = bool(args.periodic)
    args.process = bool(args.process)

    if args.save_dir is None:
        args.save_dir = osp.join(os.getcwd(), 'output', 'pre-train')
        if not osp.exists(args.save_dir):
            directory = pathlib.Path(args.save_dir)
            directory.mkdir(parents=True, exist_ok=True)

    if args.run_name is None:
        args.run_name = f'{args.num_layers}_{args.n_mha}_{args.n_alignn}_{args.n_gnn}'

    main(args)
