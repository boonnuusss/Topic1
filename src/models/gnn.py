"""Small PyTorch GNN modules for STEC edge regression."""

from __future__ import annotations


def build_graph_sage_edge_regressor(
    *,
    node_dim: int,
    edge_dim: int,
    hidden_dim: int = 64,
    embedding_dim: int = 32,
    dropout: float = 0.2,
):
    """Create a pure-PyTorch GraphSAGE-style edge regressor.

    The architecture follows the local reference notebook
    ``GNN_sTEC_improve.ipynb`` but avoids a hard dependency on
    ``torch_geometric``.  Message passing uses mean neighbor aggregation,
    while edge attributes are consumed by the final edge decoder.
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MeanSAGELayer(nn.Module):
        """GraphSAGE mean aggregation without external graph libraries."""

        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.lin_self = nn.Linear(in_dim, out_dim)
            self.lin_neigh = nn.Linear(in_dim, out_dim)

        def forward(self, x, edge_index):
            src = edge_index[0]
            dst = edge_index[1]
            agg = x.new_zeros((x.shape[0], x.shape[1]))
            agg.index_add_(0, dst, x[src])
            deg = x.new_zeros((x.shape[0], 1))
            one = torch.ones((src.shape[0], 1), dtype=x.dtype, device=x.device)
            deg.index_add_(0, dst, one)
            neigh = agg / deg.clamp_min(1.0)
            return self.lin_self(x) + self.lin_neigh(neigh)

    class GraphSAGEEdgeRegressor(nn.Module):
        """Receiver/satellite graph encoder plus edge-level STEC decoder."""

        def __init__(self):
            super().__init__()
            self.lin_node = nn.Linear(node_dim, hidden_dim)
            self.conv1 = MeanSAGELayer(hidden_dim, hidden_dim)
            self.conv2 = MeanSAGELayer(hidden_dim, embedding_dim)
            decoder_input_dim = embedding_dim * 2 + edge_dim
            self.edge_decoder = nn.Sequential(
                nn.Linear(decoder_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
            )
            self.dropout = float(dropout)

        def encode(self, x, message_edge_index):
            x = F.relu(self.lin_node(x))
            x = F.relu(self.conv1(x, message_edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            return self.conv2(x, message_edge_index)

        def decode(self, z, decode_edge_index, decode_edge_attr):
            src = z[decode_edge_index[0]]
            dst = z[decode_edge_index[1]]
            edge_input = torch.cat([src, dst, decode_edge_attr], dim=1)
            return self.edge_decoder(edge_input)

        def forward(
            self,
            x,
            message_edge_index,
            message_edge_attr,
            *,
            decode_edge_index=None,
            decode_edge_attr=None,
        ):
            if decode_edge_index is None:
                decode_edge_index = message_edge_index
            if decode_edge_attr is None:
                decode_edge_attr = message_edge_attr
            z = self.encode(x, message_edge_index)
            return self.decode(z, decode_edge_index, decode_edge_attr)

    return GraphSAGEEdgeRegressor()
