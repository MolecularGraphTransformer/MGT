import math
from typing import Union, Tuple, Optional

import dgl
import torch
import dgl.function as fn
import torch.nn.functional as F
from torch import nn
from dgl import softmax_edges
from dgl.nn.functional import edge_softmax

class multiheaded(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, 
                 concat: bool = True, dropout: float = 0.2, bias: bool = True, 
                 residual: bool = True, norm: Union[bool, Tuple[bool, bool]] = True):
        super(multiheaded, self).__init__()

        if isinstance(norm, bool):
            norm = (norm, norm)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.residual = residual
        self.norm = norm

        self.lin_query = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_key = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_value = nn.Linear(in_channels, heads * out_channels, bias=bias)

        self.lin_edge = nn.Linear(in_channels, heads * out_channels, bias=bias)

        if self.concat:
            if self.norm[0]:
                self.norm_node = nn.LayerNorm(heads * out_channels)
            if self.norm[1]:
                self.norm_edge = nn.LayerNorm(heads * out_channels)
        else:
            if self.norm[0]:
                self.norm_node = nn.LayerNorm(out_channels)
            if self.norm[1]:
                self.norm_edge = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_query.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor, edge_feats: torch.Tensor):
        r"""
        Args:
            g: dgl.DGLGraph
                The graph
            node_feats: torch.Tensor
                The node features
            edge_feats: torch.Tensor
                The edge features
            return_attention (bool, optional):
                If set to :obj:`True`, will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        g = g.local_var()

        g.ndata['query'] = self.lin_query(node_feats).view(-1, self.heads, self.out_channels)
        g.ndata['key'] = self.lin_key(node_feats).view(-1, self.heads, self.out_channels)
        g.ndata['value'] = self.lin_value(node_feats).view(-1, self.heads, self.out_channels)
        g.apply_edges(fn.u_mul_v('query', 'key', 'scores'))
        m = g.edata.pop('scores') + self.lin_edge(edge_feats).view(-1, self.heads, self.out_channels)
        
        scores = m / math.sqrt(self.out_channels)
        g.edata['alpha'] = F.dropout(edge_softmax(g, scores).sum(dim=-1).unsqueeze(dim=-1), p=self.dropout)
        g.update_all(fn.u_mul_e('value', 'alpha', 'm'), fn.sum('m', 'h'))
        x = g.ndata.pop('h')

        if self.concat:
            x = x.view(-1, self.heads * self.out_channels)
            m = m.view(-1, self.heads * self.out_channels)
        else:
            x = x.mean(dim=1)
            m = m.mean(dim=1)

        if self.norm[0]:
            x = self.norm_node(x)
        if self.norm[1]:
            m = self.norm_edge(m)

        if self.residual:
            x += node_feats
            y = m + edge_feats

        return x, y
