import dgl
import torch
import random

from dgl import BaseTransform


class MaskAtom(BaseTransform):
    def __init__(self, num_atom_fea, mask_rate, node_feat_name):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        :param num_atom_fea:
        :param mask_rate: % of atoms to be masked
        masked atoms
        """
        self.num_atom_fea = num_atom_fea
        self.node_feat_name = node_feat_name
        self.mask_rate = mask_rate

    def __call__(self, g, masked_atom_indices=None):
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                      [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples (num_atoms * mask rate) number of atom indices, 
            otherwise a list of atom idx that sets the atoms to be masked 
            (for debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
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
        return g, nsg

    def __repr__(self):
        return '{}(num_atom_fea={}, mask_rate={})'.format(self.__class__.__name__, self.num_atom_fea, self.mask_rate)