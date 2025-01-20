import os
import time
import pathlib
import argparse
import os.path as osp

from model.transformer import multiheaded
from model.alignn import EdgeGatedGraphConv
from model.graphformer import Graphformer, encoder
from utils.datasets import StructureDataset

import torch
import torch.nn as nn
from lightning.fabric import Fabric
from dgl.dataloading import GraphDataLoader
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import FSDPStrategy


def test(args, model, loader, fabric: Fabric):

    fabric.print('-------------------- Run Started --------------------')
    model.eval()
    start_time = time.time()
    for g, lg, fg, ids in loader:

        # Get model's output
        with torch.no_grad():
            output, _, _, _, _ = model(g, lg, fg)

        # Get predictions as lists
        preds = output.squeeze(dim=0).tolist()

        # save data to be logged
        test_data = {
            'id': ids[0]
        }
        for i in range(args.out_dims):
            test_data[f'prediction {args.out_names[i]}'] = preds[i]
        fabric.log_dict(test_data)
    end_time = time.time()
    fabric.print('-------------------- Run Finished --------------------')


def main(args):

    logger = CSVLogger(
        root_dir=args.save_dir, 
        name=args.run_name,
        flush_logs_every_n_steps=1
    )
    policy = {encoder, EdgeGatedGraphConv, multiheaded}
    fsdp_strategy = FSDPStrategy(auto_wrap_policy=policy, state_dict_type='full')
    fabric = Fabric(accelerator=args.accelerator, devices=args.n_devices, num_nodes=args.n_nodes, strategy=fsdp_strategy, loggers=logger)
    fabric.launch()

    # ------------------------------------- DATASET SETUP -------------------------------------
    data = StructureDataset(args, process=args.process)
    testing_loader = GraphDataLoader(data, collate_fn=data.collate_run, batch_size=1, shuffle=True)
    testing_loader = fabric.setup_dataloaders(testing_loader)

    # ------------------------------------- MODEL AND LOSS/ERROR FUNCTION SETUP -------------------------------------
    model = Graphformer(args=args)
    model = fabric.setup_module(model)

    saved_model = osp.join(args.model_path, args.model_name)
    assert osp.exists(saved_model), f'No model save as {args.model_name} exists in path {str(args.model_path)}'
    state_dicts = {'model': model}
    fabric.load(saved_model, state=state_dicts)

    criterion = nn.L1Loss()

    test(args, model, testing_loader, criterion, fabric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implementation of Pre-Training strategy for the Molecular Graph Transformer")
    # Fabric Arguments
    parser.add_argument('--n_devices', type=int, default=1, help='number of gpus/cpus that the code has access to (default: 8)')
    parser.add_argument('--n_nodes', type=int, default=1, help='number of nodes/computers on which the model is being trained (default: 1)')
    parser.add_argument('--accelerator', type=str, default='cuda', choices=['cpu', 'gpu', 'mps', 'cuda', 'tpu'], 
                        help='device type on which the training is happening [cpu, gpu, mps (apple M1/M2 only), cuda (NVIDIA GPUs only), tpu] (default: cuda)')
    # Save and Load Arguments
    parser.add_argument('--root', type=str, help='root directory for all datasets (default: None)', required=True)
    parser.add_argument('--model_path', type=str, help='directory in which to save the trained model', required=True)
    parser.add_argument('--run_name', type=str, default=None, help='name of run for logging purposes')
    parser.add_argument('--save_dir', type=str, default=None, help='directory in which to save the test results')
    parser.add_argument('--model_name', type=str, default='lowest_model.ckpt', help='name of the model to load')
    parser.add_argument('--out_names', nargs='+', type=str, default=None, help='names of the outputs [for logging purposes only]')
    # Model and Dataset Arguments
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
    args.process = bool(args.process)

    if args.save_dir is None:
        args.save_dir = osp.join(os.getcwd(), 'output', 'run')
        if not osp.exists(args.save_dir):
            directory = pathlib.Path(args.save_dir)
            directory.mkdir(parents=True, exist_ok=True)
    
    if args.run_name is None:
        args.run_name = f'{args.num_layers}_{args.n_mha}_{args.n_alignn}_{args.n_gnn}'

    main(args)
