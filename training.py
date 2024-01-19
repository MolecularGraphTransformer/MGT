import os
import time
import argparse
import numpy as np
import os.path as osp

from model.transformer import multiheaded
from model.alignn import EdgeGatedGraphConv
from model.graphformer import Graphformer, encoder
from utils.datasets import StructureDataset

import torch
import torch.nn as nn
from lightning.fabric import Fabric
from torch.utils.data import DataLoader
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import FSDPStrategy


def train(args, model, train_loader, val_loader, optimizer, criterion, error_func, fabric: Fabric):    

    fabric.print('-------------------- Training and Validation Started --------------------')
    lowest_error = 1000000
    per_epoch_times = []
    start_time = time.time()
    for epoch in range(args.begin_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        epoch_loss = torch.zeros(2).to(fabric.local_rank)

        # -------------------- TRAINING --------------------
        model.train()
        optimizer.zero_grad()
        training_start_time = time.time()
        for iteration, (g, lg, fg, target, _) in enumerate(train_loader):

            is_accumulating = iteration % args.n_cum != 0

            with fabric.no_backward_sync(model, enabled=is_accumulating):
                output, _, _, _, _ = model(g, lg, fg)
                loss = criterion(output, target) / args.n_cum

                fabric.backward(loss)

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()

            # Save Loss
            epoch_loss[0] += (loss.item() * args.n_cum)
            epoch_loss[1] += args.batch_size
        fabric.print(f'Training time: {time.time() - training_start_time} seconds')

        # -------------------- VALIDATION --------------------
        model.eval()
        val_loss = torch.zeros(2).to(fabric.local_rank)
        epoch_error = torch.zeros(2).to(fabric.local_rank)
        epoch_indiv_error = [torch.zeros(2).to(fabric.local_rank) for _ in range(args.out_dims)]
        validation_start_time = time.time()
        for g, lg, fg, target, _ in val_loader:

            with torch.no_grad():
                output, _, _, _, _ = model(g, lg, fg)

            # Get overall loss and error
            loss = criterion(output, target)
            val_loss[0] += loss.item()
            val_loss[1] += args.batch_size
            error = error_func(output, target)
            epoch_error[0] += error.item()
            epoch_error[1] += args.batch_size

            # Get individual errors
            if args.out_dims > 1:
                targets = torch.hsplit(target, int(target.shape[1]))
                outputs = torch.hsplit(output, int(output.shape[1]))
                individual_errors = [error_func(outputs[i], targets[i]) for i in range(len(outputs))]
                for i, error in enumerate(individual_errors):
                    epoch_indiv_error[i][0] += error.item()
                    epoch_indiv_error[i][1] += args.batch_size
        fabric.print(f'Validation time: {time.time() - validation_start_time} seconds')

        # Average errors and losses
        fabric.all_reduce(epoch_loss, reduce_op='sum')
        epoch_loss = epoch_loss[0]/epoch_loss[1]
        fabric.all_reduce(val_loss, reduce_op='sum')
        val_loss = val_loss[0]/val_loss[1]
        fabric.all_reduce(epoch_error, reduce_op='sum')
        epoch_error = epoch_error[0]/epoch_error[1]
        if args.out_dims > 1:
            for i, error in enumerate(epoch_indiv_error):
                error = fabric.all_reduce(error, reduce_op='sum')
                epoch_indiv_error[i] = error[0]/error[1]

        per_epoch_time = time.time() - epoch_start_time
        per_epoch_times.append(per_epoch_time)

        # Print results
        fabric.print('Epoch %d of %d with %.2f s' % (epoch, args.epochs, per_epoch_time))
        fabric.print('Epoch loss: %.4f \\ Validation loss: %.4f \\ Validation error: %.4f' % (epoch_loss, val_loss, epoch_error))
        if args.out_dims > 1:
            errors_str = ''
            for i in range(args.out_dims):
                errors_str += str(f'{args.out_names[i]} Error: {epoch_indiv_error[i]}')
                if i + 1 != args.out_dims:
                    errors_str += ' | '
            fabric.print(errors_str)

        # Log results
        results = {
            'Train Loss': epoch_loss,
            'Validation Loss': val_loss,
            'Validation Error': epoch_error,
        }
        if args.out_dims > 1:
            for i in range(args.out_dims):
                results[f'{args.out_names[i]} Error'] = epoch_indiv_error[i]
        fabric.log_dict(results, step=epoch)

        # Save best performing model
        if epoch_error < lowest_error:
            lowest_error = epoch_error

            state = {'model': model, 'optimizer': optimizer}
            fabric.save(osp.join(args.model_path, args.lowest_model), state)

    fabric.print(f'Average per epoch time: {np.mean(per_epoch_times)} seconds, Total {args.epochs} epochs time: {time.time() - start_time} seconds')
    fabric.print('-------------------- Training Finished --------------------')

    state = {'model': model, 'optimizer': optimizer}
    fabric.save(osp.join(args.model_path, args.final_model), state)


def main(args):

    if not osp.exists(args.model_path):
        os.makedirs(args.model_path)

    # ------------------------------------- FABRIC SETUP -------------------------------------
    logger = CSVLogger(
        root_dir='/jmain02/home/J2AD007/txk101/mxa59-txk101/python_scripts/FSDP/outputs/train/extensive', 
        name=args.run_name,
        flush_logs_every_n_steps=1
    )
    policy = {encoder, EdgeGatedGraphConv, multiheaded}
    fsdp_strategy = FSDPStrategy(auto_wrap_policy=policy, activation_checkpointing_policy=policy, state_dict_type='full')
    fabric = Fabric(accelerator=args.accelerator, devices=args.n_devices, num_nodes=args.n_nodes, strategy=fsdp_strategy, loggers=logger)
    fabric.launch()

    # ------------------------------------- DATASET SETUP -------------------------------------
    dataset = StructureDataset(args)
    training_data, validation_data = torch.utils.data.random_split(dataset, [args.train_split, args.val_split])
    training_loader = DataLoader(training_data, collate_fn=dataset.collate_tt, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, collate_fn=dataset.collate_tt, batch_size=args.batch_size, shuffle=True)
    training_loader, validation_loader = fabric.setup_dataloaders(training_loader), fabric.setup_dataloaders(validation_loader)    

    # ------------------------------------- MODEL, OPTIMIZER AND LOSS/ERROR FUNCTION SETUP -------------------------------------
    model = Graphformer(args=args)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum(p.numel() for p in model_parameters)
    fabric.print(f'ARCHITECTURE: \n    Layers - {args.num_layers} \n    MHAs - {args.n_mha} \n    ALIGNNs - {args.n_alignn} \n    GNNs - {args.n_gnn} \n    Parameters - {num_params} ')

    if args.load_model == 1:
        fabric.print('Pre-Training not yet implemented')
        exit()
        # model.freeze_train()

    model = fabric.setup_module(model)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.decay)
    optimizer = fabric.setup_optimizers(optimizer)

    # if args.load_model == 1:
    #     pretrain_path = osp.join(args.model_path, args.pretrain_model)
    #     state_dicts = fabric.load(pretrain_path)
    #     model.load_state_dict(state_dicts['model'])
    if args.load_model == 2:
        state = {'model': model, 'optimizer': optimizer}
        lowest_model = osp.join(args.model_path, args.lowest_model)
        fabric.load(lowest_model, state)

    criterion = nn.MSELoss()
    error_func = nn.L1Loss()

    train(args, model, training_loader, validation_loader, optimizer, criterion, error_func, fabric)


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
    parser.add_argument('--final_model', type=str, default='end_model.ckpt', help='name with which to save the model')
    parser.add_argument('--lowest_model', type=str, default='lowest.ckpt', help='name with which to save the model with the best performance')
    parser.add_argument('--load_model', type=int, default=0, choices=[0, 1, 2], help='whether there is a model to be loaded (0: no model loading, 1: load pretrained, 2: load checkpoint)')
    parser.add_argument('--out_names', nargs='+', type=str, default=None, help='names of the outputs [for logging purposes only]')
    # Training Arguments
    parser.add_argument('--n_cum', type=int, default=8, help='number of batches to accumulate the error for')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for training (default: 2)')
    parser.add_argument('--train_split', type=int, help='number of items in dataset to be used for training', required=True)
    parser.add_argument('--val_split', type=int, help='number of items in dataset to be used for validation', required=True)
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for (default: 100)')
    parser.add_argument('--begin_epoch', type=int, default=1, help='set to restart training from a specific epoch')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--decay', type=float, default=1e-5, help='weight decay for the optimizers (default: 1e-5)')
    # Model and Dataset Arguments
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
    parser.add_argument('--num_layers', type=int, default=3, help='number of encoders in the network (default: 3)')
    parser.add_argument('--n_mha', type=int, default=1, help='number of attention layers in each encoder (default: 1)')
    parser.add_argument('--n_alignn', type=int, default=2, help='number of graph convolutions in each encoder (default: 2)')
    parser.add_argument('--n_gnn', type=int, default=2, help='number of graph convolutions in each encoder (default: 2)')
    parser.add_argument('--n_heads', type=int, default=4, help='number of attention heads (default: 4)')
    parser.add_argument('--residual', type=int, default=1, choices=[0, 1], help='whether to add residuality to the network or not (default: True)')
    
    args = parser.parse_args()

    if args.out_names is None:
        args.out_names = []
        for i in range(args.out_dims):
            args.out_names.append(str(i + 1))
    assert len(args.out_names) == args.out_dims, 'number of outputs and output names not the same'
    args.residual = bool(args.residual)
    args.periodic = bool(args.periodic)

    if args.save_dir is None:
        args.save_dir = osp.join(os.getcwd(), 'output', 'train')
        if not osp.exists(args.save_dir):
            os.mkdir(args.save_dir)

    if args.run_name is None:
        args.run_name = f'{args.num_layers}_{args.n_mha}_{args.n_alignn}_{args.n_gnn}'

    main(args)
