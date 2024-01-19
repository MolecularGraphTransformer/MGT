import csv
import json
import random
import warnings
from typing import List, Tuple

import numpy as np
import os.path as osp

import dgl
import torch.utils.data
from dgl import load_graphs
from pymatgen.core import Structure


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(float(key)): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())

        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return {"angle_feats": bond_cosine.unsqueeze(1)}


class StructureDataset(torch.utils.data.Dataset):
    ''' Dataset for Molecular Graph Representations '''

    def __init__(self, args, scope: int = 0, process: bool = False, random_seed: int = 123, transform=None):

        assert scope in [0, 1, 2], 'Error setting scope of dataset, options are:\n 0: training and testing;\n 1: pre-training;\n 2: run'
        self.scope = scope
        
        self.root = args.root
        self.random_seed = random_seed
        self.transform = transform

        self.process = process
        if self.process:
            self.raw_dir = osp.join(self.root, 'raw')
            self.max_nei_num = args.max_nei_num
            self.pe_dim = args.num_pe_fea
            self.radius = args.local_radius
            self.random_seed = random_seed
            self.periodic = args.periodic
            if args.periodic:
                self.periodic_radius = args.periodic_radius

            atom_init_file = osp.join(self.root, 'atom_init.json')
            assert osp.exists(atom_init_file), 'atom_init.json file does not exist!'
            self.cai = AtomCustomJSONInitializer(atom_init_file)
        else:
            self.proc_dir = osp.join(self.root, 'processed')

        id_prop_file = osp.join(self.root, 'id_prop.csv')
        assert osp.exists(id_prop_file), 'id_prop.csv file does not exist'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

    def __len__(self):
        return len(self.id_prop_data)

    def shuffle(self):
        random.seed(self.random_seed)
        random.shuffle(self.id_prop_data)
        return

    def __getitem__(self, idx):
        cif_id = self.id_prop_data[idx][0]

        if self.process:
            g, lg, fg = self._construct_graph(cif_id)
        else:
            g, lg, fg = load_graphs(osp.join(self.proc_dir, f'{cif_id}.bin'))[0]

        if self.transform:
            g = self.transform(g)

        if self.scope == 0:
            props = [float(x) for x in self.id_prop_data[idx][1:]]
            props = np.array(props)
            return g, lg, fg, torch.tensor(props, dtype=torch.float32), cif_id
        elif self.scope == 1:
            return g, lg, fg
        else:
            return g, lg, fg, cif_id
            

    def _construct_graph(self, cif_id):
        structure = Structure.from_file(osp.join(self.raw_dir, cif_id + '.cif'))
        a, b, c = structure.lattice.abc
        d = a ** 2 + b ** 2 + c ** 2
        diag = d ** 0.5

        if diag > self.periodic_radius:
            diag = self.periodic_radius

        # Get atom features
        atom_fea = np.vstack([self.cai.get_atom_fea(structure[i].specie.number) for i in range(len(structure))])
        atom_fea = torch.Tensor(atom_fea)

        # Get neighbours
        local_nbrs = structure.get_all_neighbors(self.radius, include_index=True)
        local_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in local_nbrs]
        if self.periodic:
            full_nbrs = structure.get_all_neighbors(diag, include_index=True)
        else:
            full_nbrs = structure.sites

        nbr_idx, nbr_fea, nbr_disp, fc_idx, fc_coulomb = [], [], [], [], []

        # Get edges
        for idx, nbr in enumerate(local_nbrs):
            atm = structure[idx]
            if len(nbr) < 12:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_idx.extend(list(map(lambda x: (idx, x[2]), nbr)))
                nbr_fea.extend(list(map(lambda x: x[1], nbr)))
                nbr_disp.extend(list(map(lambda x: x.coords - structure[idx].coords, nbr)))

                fc_idx.extend(list(map(lambda x: (idx, x[2]), full_nbrs[idx])))
                distances = np.array(list(map(lambda x: x[1], full_nbrs[idx])))
                charges = np.array(list(map(lambda x: x.specie.Z * atm.specie.Z, full_nbrs[idx])))
                fc_coulomb.extend(charges / distances)
            else:
                nbr_idx.extend(list(map(lambda x: (idx, x[2]), nbr[:12])))
                nbr_fea.extend(list(map(lambda x: x[1], nbr[:12])))
                nbr_disp.extend(list(map(lambda x: x.coords - atm.coords, nbr[:12])))

                fc_idx.extend(list(map(lambda x: (idx, x[2]), full_nbrs[idx])))
                distances = np.array(list(map(lambda x: x[1], full_nbrs[idx])))
                charges = np.array(list(map(lambda x: x.specie.Z * atm.specie.Z, full_nbrs[idx])))
                fc_coulomb.extend(charges / distances)

        edge_idx, edge_fea, fc_index, fc_fea, edge_disp = torch.LongTensor(np.array(nbr_idx)).t().contiguous(), \
                                                          torch.tensor(np.array(nbr_fea), dtype=torch.float32).unsqueeze(dim=1), \
                                                          torch.LongTensor(np.array(fc_idx)).t().contiguous(), \
                                                          torch.tensor(np.array(fc_coulomb), dtype=torch.float32).unsqueeze(dim=1), \
                                                          torch.tensor(np.array(nbr_disp), dtype=torch.float32)

        # Construct Local Graph
        G = dgl.graph(data=(edge_idx[0], edge_idx[1]), num_nodes=atom_fea.shape[0])
        G.ndata['node_feats'] = atom_fea
        G.edata['edge_feats'] = edge_fea
        G.edata['r'] = edge_disp
        # Get Positional Encodings
        G.ndata['pes'] = dgl.lap_pe(G, 10, padding=True)

        # Construct Full Graph
        FG = dgl.graph(data=(fc_index[0], fc_index[1]), num_nodes=atom_fea.shape[0])
        FG.edata['fc_feats'] = fc_fea

        # Construct Line Graph
        LG = G.line_graph(shared=True)
        LG.apply_edges(compute_bond_cosines)

        return G, LG, FG

    @staticmethod
    def collate_run(samples: List[Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph, str]]):
        graphs, line_graphs, full_graphs, labels, ids = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)
        batched_full_graph = dgl.batch(full_graphs)
        if len(labels[0].size()) > 0:
            return batched_graph, batched_line_graph, batched_full_graph, ids
        else:
            return batched_graph, batched_line_graph, batched_full_graph, ids

    @staticmethod
    def collate_tt(samples: List[Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph, torch.Tensor, str]]):
        graphs, line_graphs, full_graphs, labels, ids = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)
        batched_full_graph = dgl.batch(full_graphs)
        if len(labels[0].size()) > 0:
            return batched_graph, batched_line_graph, batched_full_graph, torch.stack(labels), ids
        else:
            return batched_graph, batched_line_graph, batched_full_graph, torch.tensor(labels), ids

    @staticmethod
    def collate_pre(samples: List[Tuple[Tuple, dgl.DGLGraph, dgl.DGLGraph, torch.Tensor]]):
        graphs, line_graphs, full_graphs = map(list, zip(*samples))
        if len(samples[0][0]) == 2:
            graphs, nodes_sub, line_graphs, full_graphs = map(list, zip(*samples))
            cum_n = 0
            for i, g in enumerate(graphs):
                nodes_sub[i].ndata[dgl.NID] = nodes_sub[i].ndata[dgl.NID] + cum_n
                cum_n += g.num_nodes()
            batched_graph = dgl.batch(graphs)
            batched_nodes = dgl.batch(nodes_sub)
            batched_line_graph = dgl.batch(line_graphs)
            batched_full_graph = dgl.batch(full_graphs)
            return (batched_graph, batched_nodes), batched_line_graph, batched_full_graph
        else:
            graphs, nodes_sub, edges_sub = [s[0][0] for s in samples], [s[0][1] for s in samples], [s[0][2] for s in samples]
            cum_n, cum_e = 0, 0
            for i, g in enumerate(graphs):
                nodes_sub[i].ndata[dgl.NID] = nodes_sub[i].ndata[dgl.NID] + cum_n
                edges_sub[i].edata[dgl.EID] = edges_sub[i].edata[dgl.EID] + cum_e
                cum_n += g.num_nodes()
                cum_e += g.num_edges()
            batched_graph = dgl.batch(graphs)
            batched_nodes = dgl.batch(nodes_sub)
            batched_edges = dgl.batch(edges_sub)
            batched_line_graph = dgl.batch(line_graphs)
            batched_full_graph = dgl.batch(full_graphs)
            return (batched_graph, batched_nodes, batched_edges), batched_line_graph, batched_full_graph
