import dgl
import torch
import random

from dgl import BaseTransform


class MaskAtom(BaseTransform):
    def __init__(self, num_atom_fea, num_edge_fea, mask_rate, node_feat_name, edge_feat_name, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_fea:
        :param num_edge_fea:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_fea = num_atom_fea
        self.num_edge_fea = num_edge_fea
        self.node_feat_name = node_feat_name
        self.edge_feat_name = edge_feat_name
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

    def __call__(self, g, masked_atom_indices=None):
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = g.num_nodes()
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        nsg = dgl.node_subgraph(g, masked_atom_indices, relabel_nodes=True, store_ids=True)
        nodes = nsg.ndata.pop(self.node_feat_name)
        for id in masked_atom_indices:
            g.ndata[self.node_feat_name][id] = torch.zeros(self.num_atom_fea)
        nsg.ndata[self.node_feat_name] = nodes
        if self.mask_edge:
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(torch.stack(g.edges()).numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            esg = dgl.edge_subgraph(g, connected_edge_indices, store_ids=True)
            edges = esg.edata.pop(self.edge_feat_name)
            for id in connected_edge_indices:
                g.edata[self.edge_feat_name][id] = torch.zeros(self.num_edge_fea)
            esg.edata[self.edge_feat_name] = edges
            return g, nsg, esg
        else:
            return g, nsg

    def __repr__(self):
        return '{}(num_atom_fea={}, num_edge_fea={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_fea, self.num_edge_fea,
            self.mask_rate, self.mask_edge)