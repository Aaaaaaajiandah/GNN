"""
Supply Chain Graph Neural Network
- Runs on Intel Arc XPU (B580) via PyTorch XPU backend
- BF16 precision for memory efficiency
- Propagates sector growth signals upstream (suppliers) and downstream (customers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    """Select best available device: XPU > CUDA > CPU"""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"[Device] Intel XPU detected: {torch.xpu.get_device_name(0)}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] CPU (no XPU/CUDA found)")
    return device


DEVICE = get_device()
# BF16 supported on XPU and modern CUDA; fallback to FP32
USE_BF16 = DEVICE.type in ("xpu", "cuda")
DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
print(f"[Precision] Using {'BF16' if USE_BF16 else 'FP32'}")


# ─── Graph Convolution ────────────────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    """Simple message-passing layer: aggregate neighbor features + self."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : [N, in_dim]  node features
        adj : [N, N]        adjacency (can be directed, weighted)
        """
        # Normalize adjacency rows
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg

        agg = adj_norm @ x           # [N, in_dim]  neighbour aggregation
        combined = torch.cat([x, agg], dim=-1)   # [N, in_dim*2]
        out = self.linear(combined)
        out = self.norm(out)
        return F.gelu(out)


# ─── Supply Chain GNN ─────────────────────────────────────────────────────────

class SupplyChainGNN(nn.Module):
    """
    Two-stream GNN:
      - upstream_adj  : edges pointing TO suppliers  (company → supplier)
      - downstream_adj: edges pointing TO customers  (company → customer)

    Predicts revenue_impact for each node given a shock vector.
    """

    def __init__(self, node_feat_dim: int, hidden: int = 128, layers: int = 3):
        super().__init__()
        # Encode node features + shock together so shock is part of propagation
        self.encoder = nn.Linear(node_feat_dim + 1, hidden)  # +1 for shock

        self.up_convs   = nn.ModuleList([GraphConvLayer(hidden, hidden) for _ in range(layers)])
        self.down_convs = nn.ModuleList([GraphConvLayer(hidden, hidden) for _ in range(layers)])

        # Shock diffusion: propagate shock signal through the graph separately
        self.shock_encoder = nn.Linear(1, hidden)
        self.shock_up_conv   = GraphConvLayer(hidden, hidden)
        self.shock_down_conv = GraphConvLayer(hidden, hidden)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        node_feats: torch.Tensor,     # [N, feat_dim]
        upstream_adj: torch.Tensor,   # [N, N]
        downstream_adj: torch.Tensor, # [N, N]
        shock: torch.Tensor,          # [N, 1]  sector growth signal per node
    ) -> torch.Tensor:                # [N, 1]  predicted revenue impact

        # Encode node features WITH shock concatenated — shock is part of node state
        h = F.gelu(self.encoder(torch.cat([node_feats, shock], dim=-1)))  # [N, hidden]

        # Propagate node embeddings upstream (toward suppliers)
        hu = h
        for conv in self.up_convs:
            hu = conv(hu, upstream_adj)

        # Propagate node embeddings downstream (toward customers)
        hd = h
        for conv in self.down_convs:
            hd = conv(hd, downstream_adj)

        # Separately propagate the raw shock signal through the graph
        # This explicitly diffuses the shock to neighbours
        hs = F.gelu(self.shock_encoder(shock))           # [N, hidden]
        hs_up   = self.shock_up_conv(hs, upstream_adj)   # shock felt by suppliers
        hs_down = self.shock_down_conv(hs, downstream_adj) # shock felt by customers
        hs_combined = hs_up + hs_down                    # [N, hidden]

        combined = torch.cat([hu, hd, h, hs_combined], dim=-1)  # [N, hidden*4]
        return self.head(combined)                               # [N, 1]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def to_device(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(device=DEVICE, dtype=DTYPE)


def build_adjacency(edges: list[tuple[int, int]], n_nodes: int, weighted: bool = False,
                    weights: list[float] | None = None) -> torch.Tensor:
    """Build a dense adjacency matrix from an edge list."""
    adj = torch.zeros(n_nodes, n_nodes)
    for i, (src, dst) in enumerate(edges):
        w = weights[i] if (weighted and weights) else 1.0
        adj[src, dst] = w
    return adj


def split_adjacency(adj: torch.Tensor):
    """
    Given a single directed adjacency, return:
      upstream_adj  : adj transposed  (who supplies TO a node)
      downstream_adj: adj as-is       (who a node supplies TO)
    """
    return adj.T.contiguous(), adj
