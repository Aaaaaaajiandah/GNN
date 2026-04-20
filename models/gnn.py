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
        self.encoder = nn.Linear(node_feat_dim, hidden)

        self.up_convs   = nn.ModuleList([GraphConvLayer(hidden, hidden) for _ in range(layers)])
        self.down_convs = nn.ModuleList([GraphConvLayer(hidden, hidden) for _ in range(layers)])

        # Shock projection: sector growth signal → node embedding delta
        self.shock_proj = nn.Linear(1, hidden)

        # Output head: predict revenue impact %
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
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

        h = F.gelu(self.encoder(node_feats))  # [N, hidden]

        # Propagate upstream (towards suppliers)
        hu = h
        for conv in self.up_convs:
            hu = conv(hu, upstream_adj)

        # Propagate downstream (towards customers)
        hd = h
        for conv in self.down_convs:
            hd = conv(hd, downstream_adj)

        # Inject shock signal
        hs = F.gelu(self.shock_proj(shock))   # [N, hidden]

        combined = torch.cat([hu, hd, hs], dim=-1)  # [N, hidden*3]
        return self.head(combined)                   # [N, 1]


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
