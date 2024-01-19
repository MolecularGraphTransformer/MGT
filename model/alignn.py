from typing import Tuple, Union
import dgl
import numpy
import torch
import dgl.function as fn
from torch import nn, Tensor
from torch.nn import functional as F


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.
    see also arxiv:2003.0098.
    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(self, feature_dims: int, norm: Union[bool, Tuple[bool, bool]] = True, residual: bool = True):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual

        if isinstance(norm, bool):
            norm = (norm, norm)
        self.norm = norm

        self.src_gate = nn.Linear(feature_dims, feature_dims)
        self.dst_gate = nn.Linear(feature_dims, feature_dims)
        self.edge_gate = nn.Linear(feature_dims, feature_dims)
        if norm[0]:
            self.norm_nodes = nn.LayerNorm(feature_dims)

        self.src_update = nn.Linear(feature_dims, feature_dims)
        self.dst_update = nn.Linear(feature_dims, feature_dims)
        if norm[1]:
            self.norm_edges = nn.LayerNorm(feature_dims)

    def forward(self, g: dgl.DGLGraph, node_feats: Tensor, edge_feats: Tensor) -> torch.Tensor:
        """Edge-gated graph convolution.
        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        if self.norm[0]:
            x = self.norm_nodes(x)
            x = F.silu(x)
        else:
            x = F.silu(x)
        if self.norm[1]:
            m = self.norm_edges(m)
            y = F.silu(m)
        else:
            y = F.silu(m)

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNLayer(nn.Module):
    def __init__(self, feature_dims: int, edge_norm: Union[bool, Tuple[bool, bool]] = True, node_norm: Union[bool, Tuple[bool, bool]] = True):
        super(ALIGNNLayer, self).__init__()
        if isinstance(edge_norm, bool):
            edge_norm = (edge_norm, edge_norm)
        if isinstance(node_norm, bool):
            node_norm = (node_norm, node_norm)
        self.edge_update = EdgeGatedGraphConv(feature_dims=feature_dims, norm=edge_norm)
        self.atom_update = EdgeGatedGraphConv(feature_dims=feature_dims, norm=node_norm)

    def forward(self, g: dgl.DGLGraph, lg: dgl.DGLGraph, x: Tensor, y: Tensor, z: Tensor):
        # Convolution on line graph
        y, z = self.edge_update(g=lg, node_feats=y, edge_feats=z)
        # Convolution on atomistic graph
        x, y = self.atom_update(g=g, node_feats=x, edge_feats=y)

        return x, y, z
