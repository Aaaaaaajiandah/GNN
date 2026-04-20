"""
Data layer: company graph, synthetic dataset generation, and feature engineering.

Company node features (all normalized to [0,1]):
  0  revenue_bn          annual revenue in $B
  1  margin              net profit margin
  2  debt_ratio          total debt / equity
  3  yoy_growth          year-over-year revenue growth
  4  sector_id (embed)   one-hot sector (10 categories)
  5..14                  sector one-hot
  15 supplier_count      number of upstream suppliers
  16 customer_count      number of downstream customers
  17 market_cap_bn       market cap
  18 r_and_d_pct         R&D spend as % of revenue
  19 capex_pct           CapEx as % of revenue
"""

import json
import random
from dataclasses import dataclass, field, asdict
from typing import Optional
import torch

SECTORS = [
    "Semiconductors", "Automotive", "Aerospace", "Energy",
    "Retail", "Healthcare", "Financials", "Materials",
    "Software", "Industrials",
]
SECTOR_IDX = {s: i for i, s in enumerate(SECTORS)}
N_SECTORS = len(SECTORS)

NODE_FEAT_DIM = 10 + N_SECTORS   # 10 numeric + sector one-hot


@dataclass
class Company:
    id: int
    name: str
    ticker: str
    sector: str
    revenue_bn: float
    margin: float
    debt_ratio: float
    yoy_growth: float
    market_cap_bn: float
    r_and_d_pct: float
    capex_pct: float
    # filled in after graph construction
    supplier_ids: list[int] = field(default_factory=list)
    customer_ids: list[int] = field(default_factory=list)
    # optional analyst data
    sector_growth_forecast: Optional[float] = None   # e.g. 0.50 means +50 %
    forecast_horizon_months: Optional[int] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Edge:
    supplier_id: int
    customer_id: int
    relationship_strength: float  # 0-1, % of supplier revenue from this customer


# ─── Synthetic data generation ────────────────────────────────────────────────

_COMPANY_NAMES = [
    "NovaTech", "AlphaCorp", "BetaManufacturing", "GammaSystems",
    "DeltaLogistics", "EpsilonEnergy", "ZetaSemi", "EtaAuto",
    "ThetaAero", "IotaHealth", "KappaRetail", "LambdaFinancial",
    "MuMaterials", "NuSoftware", "XiIndustrial", "OmicronChips",
    "PiDrives", "RhoMotors", "SigmaJets", "TauPharma",
    "UpsilonStores", "PhiBank", "ChiSteel", "PsiCode",
    "OmegaHeavy", "StellarSemi", "PeakAuto", "CrystalAero",
    "PulseEnergy", "NexRetail", "CoreHealth", "ApexFinance",
    "SteelRoot", "ByteSoft", "IronIndustrial", "LumenChips",
    "FluxDrives", "ArcMotors", "SkyJets", "GenPharma",
]

random.seed(42)


def _make_ticker(name: str, used: set) -> str:
    base = "".join(c for c in name.upper() if c.isalpha())[:4]
    t = base
    i = 1
    while t in used:
        t = base[:3] + str(i)
        i += 1
    used.add(t)
    return t


def generate_companies(n: int = 40) -> list[Company]:
    names = _COMPANY_NAMES[:n]
    used_tickers: set[str] = set()
    companies = []
    for i, name in enumerate(names):
        sector = random.choice(SECTORS)
        companies.append(Company(
            id=i,
            name=name,
            ticker=_make_ticker(name, used_tickers),
            sector=sector,
            revenue_bn=round(random.uniform(0.5, 500.0), 2),
            margin=round(random.uniform(-0.05, 0.35), 4),
            debt_ratio=round(random.uniform(0.1, 2.5), 3),
            yoy_growth=round(random.uniform(-0.15, 0.40), 4),
            market_cap_bn=round(random.uniform(0.2, 2000.0), 2),
            r_and_d_pct=round(random.uniform(0.0, 0.20), 4),
            capex_pct=round(random.uniform(0.01, 0.15), 4),
        ))
    return companies


def generate_edges(companies: list[Company], avg_edges_per_node: int = 3) -> list[Edge]:
    n = len(companies)
    edges: list[Edge] = []
    seen: set[tuple[int, int]] = set()
    target = n * avg_edges_per_node
    attempts = 0
    while len(edges) < target and attempts < target * 10:
        attempts += 1
        supplier_id = random.randint(0, n - 1)
        customer_id = random.randint(0, n - 1)
        if supplier_id == customer_id:
            continue
        if (supplier_id, customer_id) in seen:
            continue
        seen.add((supplier_id, customer_id))
        edges.append(Edge(
            supplier_id=supplier_id,
            customer_id=customer_id,
            relationship_strength=round(random.uniform(0.05, 1.0), 3),
        ))
        companies[supplier_id].customer_ids.append(customer_id)
        companies[customer_id].supplier_ids.append(supplier_id)
    return edges


def inject_forecasts(companies: list[Company], n_forecasts: int = 5):
    """Randomly assign analyst sector growth forecasts to some companies."""
    chosen = random.sample(companies, min(n_forecasts, len(companies)))
    for c in chosen:
        c.sector_growth_forecast = round(random.uniform(0.05, 0.80), 3)
        c.forecast_horizon_months = random.choice([6, 12, 24, 36])


# ─── Feature engineering ──────────────────────────────────────────────────────

def _normalize(val, lo, hi):
    return max(0.0, min(1.0, (val - lo) / (hi - lo + 1e-9)))


def company_to_features(c: Company, n_companies: int) -> list[float]:
    feats = [
        _normalize(c.revenue_bn, 0, 500),
        _normalize(c.margin, -0.1, 0.4),
        _normalize(c.debt_ratio, 0, 3),
        _normalize(c.yoy_growth, -0.2, 0.5),
        _normalize(c.market_cap_bn, 0, 2000),
        _normalize(c.r_and_d_pct, 0, 0.25),
        _normalize(c.capex_pct, 0, 0.2),
        _normalize(len(c.supplier_ids), 0, 20),
        _normalize(len(c.customer_ids), 0, 20),
        _normalize(c.sector_growth_forecast or 0.0, 0, 1),
    ]
    # Sector one-hot
    one_hot = [0.0] * N_SECTORS
    one_hot[SECTOR_IDX[c.sector]] = 1.0
    return feats + one_hot


def build_tensors(companies: list[Company], edges: list[Edge]):
    """Return (node_feats, adj, upstream_adj, downstream_adj, shock_vec)."""
    n = len(companies)

    feat_matrix = [company_to_features(c, n) for c in companies]
    node_feats = torch.tensor(feat_matrix, dtype=torch.float32)

    # Weighted adjacency: supplier → customer
    adj = torch.zeros(n, n)
    for e in edges:
        adj[e.supplier_id, e.customer_id] = e.relationship_strength

    upstream_adj = adj.T.contiguous()    # customer ← supplier
    downstream_adj = adj                 # supplier → customer

    # Shock vector: sector growth forecast (0 if none)
    shock = torch.tensor(
        [[c.sector_growth_forecast or 0.0] for c in companies],
        dtype=torch.float32,
    )

    return node_feats, adj, upstream_adj, downstream_adj, shock


# ─── Graph save/load ──────────────────────────────────────────────────────────

def save_graph(companies: list[Company], edges: list[Edge], path: str):
    data = {
        "companies": [c.to_dict() for c in companies],
        "edges": [asdict(e) for e in edges],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_graph(path: str) -> tuple[list[Company], list[Edge]]:
    with open(path) as f:
        data = json.load(f)
    companies = [Company(**d) for d in data["companies"]]
    edges = [Edge(**e) for e in data["edges"]]
    return companies, edges
