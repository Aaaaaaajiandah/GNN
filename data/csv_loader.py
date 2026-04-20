"""
csv_loader.py — Load real company + supply chain data from CSV files.

Drop-in replacement for generate_companies() and generate_edges() in dataset.py.

Expected files (see /data/examples/ for templates):
    companies.csv   — one row per company
    edges.csv       — one row per supplier→customer relationship

Usage in train.py or app.py:
    from data.csv_loader import load_companies_from_csv, load_edges_from_csv

    companies = load_companies_from_csv("data/companies.csv")
    edges     = load_edges_from_csv("data/edges.csv", companies)
    # then continue exactly as before
"""

import csv
import os
import sys
import warnings
from dataclasses import field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import Company, Edge, SECTORS

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _float(val, default=0.0):
    try:
        return float(str(val).strip().replace("%", "").replace("$", "").replace(",", ""))
    except (ValueError, TypeError):
        return default

def _int(val, default=0):
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return default

def _str(val, default=""):
    return str(val).strip() if val is not None else default

def _normalise_sector(raw: str) -> str:
    """Best-effort map of arbitrary sector strings to our known 10."""
    raw_l = raw.lower()
    mapping = {
        "semiconductor":  "Semiconductors",
        "chip":           "Semiconductors",
        "auto":           "Automotive",
        "vehicle":        "Automotive",
        "aero":           "Aerospace",
        "defense":        "Aerospace",
        "defence":        "Aerospace",
        "energy":         "Energy",
        "oil":            "Energy",
        "gas":            "Energy",
        "utility":        "Energy",
        "retail":         "Retail",
        "consumer":       "Retail",
        "health":         "Healthcare",
        "pharma":         "Healthcare",
        "biotech":        "Healthcare",
        "medical":        "Healthcare",
        "financ":         "Financials",
        "bank":           "Financials",
        "insur":          "Financials",
        "material":       "Materials",
        "chemical":       "Materials",
        "mining":         "Materials",
        "metal":          "Materials",
        "software":       "Software",
        "tech":           "Software",
        "cloud":          "Software",
        "internet":       "Software",
        "industrial":     "Industrials",
        "manufactur":     "Industrials",
        "construction":   "Industrials",
        "transport":      "Industrials",
    }
    for key, canonical in mapping.items():
        if key in raw_l:
            return canonical
    # If already a known sector, return as-is
    for s in SECTORS:
        if s.lower() == raw_l:
            return s
    warnings.warn(f"Unknown sector '{raw}' — defaulting to 'Industrials'")
    return "Industrials"


# ─── Companies CSV loader ──────────────────────────────────────────────────────

# Required columns   (must be present, even if some values are blank)
COMPANY_REQUIRED = {"name", "ticker", "sector"}

# Optional columns with defaults
COMPANY_OPTIONAL = {
    "revenue_bn":           0.0,
    "margin":               0.0,
    "debt_ratio":           1.0,
    "yoy_growth":           0.0,
    "market_cap_bn":        0.0,
    "r_and_d_pct":          0.0,
    "capex_pct":            0.0,
    "sector_growth_forecast":    None,   # e.g. 0.50 for +50%
    "forecast_horizon_months":   None,   # e.g. 12
}


def load_companies_from_csv(path: str) -> list[Company]:
    """
    Load companies from a CSV file.

    Flexible column matching:
      - Column names are case-insensitive and strip whitespace.
      - Percentage values can be given as  0.15  OR  15  OR  15%  — all mean 15%.
      - Revenue / market_cap accept raw billions (1.5) or with suffix (1.5B / 1500M).
      - sector_growth_forecast likewise: 0.50 or 50 or 50% all mean +50%.

    Returns a list[Company] with IDs assigned 0, 1, 2, …
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"companies CSV not found: {path}")

    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Normalise header keys
        headers = {k.strip().lower(): k for k in (reader.fieldnames or [])}
        missing = COMPANY_REQUIRED - set(headers.keys())
        if missing:
            raise ValueError(f"companies.csv is missing required columns: {missing}")

        for raw_row in reader:
            row = {k.strip().lower(): v for k, v in raw_row.items()}
            rows.append(row)

    companies: list[Company] = []
    seen_tickers: set[str] = set()

    for i, row in enumerate(rows):
        name   = _str(row.get("name"))
        ticker = _str(row.get("ticker")).upper()
        sector = _normalise_sector(_str(row.get("sector", "Industrials")))

        if not name:
            warnings.warn(f"Row {i+2}: empty name — skipping")
            continue

        # Deduplicate tickers
        orig = ticker
        j = 1
        while ticker in seen_tickers:
            ticker = orig[:3] + str(j)
            j += 1
        seen_tickers.add(ticker)

        # Numeric fields — handle % suffix and M/B suffixes for money
        def money(key):
            raw = _str(row.get(key, "0"))
            raw = raw.replace(",", "")
            mult = 1.0
            if raw.upper().endswith("T"):
                mult = 1_000.0; raw = raw[:-1]
            elif raw.upper().endswith("B"):
                mult = 1.0; raw = raw[:-1]
            elif raw.upper().endswith("M"):
                mult = 0.001; raw = raw[:-1]
            elif raw.upper().endswith("K"):
                mult = 0.000_001; raw = raw[:-1]
            return _float(raw) * mult

        def pct(key, default=0.0):
            raw = _str(row.get(key, str(default))).replace("%","").replace(",","").strip()
            val = _float(raw, default)
            # If someone typed 15 meaning 15%, convert to 0.15
            if abs(val) > 1.5:
                val /= 100.0
            return round(val, 6)

        forecast_raw = row.get("sector_growth_forecast") or row.get("growth_forecast") or ""
        forecast = pct("sector_growth_forecast") if forecast_raw.strip() else None

        horizon_raw = row.get("forecast_horizon_months") or row.get("horizon_months") or ""
        horizon = _int(horizon_raw) if horizon_raw.strip() else None

        companies.append(Company(
            id=i,
            name=name,
            ticker=ticker,
            sector=sector,
            revenue_bn=        money("revenue_bn"),
            margin=            pct("margin"),
            debt_ratio=        _float(row.get("debt_ratio", "1.0")),
            yoy_growth=        pct("yoy_growth"),
            market_cap_bn=     money("market_cap_bn"),
            r_and_d_pct=       pct("r_and_d_pct"),
            capex_pct=         pct("capex_pct"),
            sector_growth_forecast=forecast,
            forecast_horizon_months=horizon,
        ))

    print(f"[CSV] Loaded {len(companies)} companies from {path}")
    return companies


# ─── Edges CSV loader ─────────────────────────────────────────────────────────

# You can identify companies by id, ticker, OR name
EDGE_REQUIRED = set()   # we detect format automatically


def load_edges_from_csv(path: str, companies: list[Company]) -> list[Edge]:
    """
    Load supply chain edges from a CSV file.

    Accepted column layouts (auto-detected):

    Layout A — by ticker:
        supplier_ticker, customer_ticker, [relationship_strength]

    Layout B — by company name:
        supplier_name, customer_name, [relationship_strength]

    Layout C — by integer id:
        supplier_id, customer_id, [relationship_strength]

    relationship_strength is optional (defaults to 1.0).
    Rows referencing unknown companies are skipped with a warning.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"edges CSV not found: {path}")

    # Build lookup maps
    by_ticker = {c.ticker.upper(): c for c in companies}
    by_name   = {c.name.lower(): c for c in companies}
    by_id     = {c.id: c for c in companies}

    def lookup(val: str) -> Company | None:
        val = val.strip()
        # Try integer id first
        try:
            return by_id.get(int(val))
        except ValueError:
            pass
        # Try ticker
        upper = val.upper()
        if upper in by_ticker:
            return by_ticker[upper]
        # Try name (case-insensitive)
        return by_name.get(val.lower())

    edges: list[Edge] = []
    skipped = 0

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = [k.strip().lower() for k in (reader.fieldnames or [])]

        # Detect which columns to use
        if "supplier_ticker" in headers and "customer_ticker" in headers:
            sup_col, cus_col = "supplier_ticker", "customer_ticker"
        elif "supplier_name" in headers and "customer_name" in headers:
            sup_col, cus_col = "supplier_name", "customer_name"
        elif "supplier_id" in headers and "customer_id" in headers:
            sup_col, cus_col = "supplier_id", "customer_id"
        elif "supplier" in headers and "customer" in headers:
            sup_col, cus_col = "supplier", "customer"
        else:
            raise ValueError(
                "edges.csv must have columns: supplier_ticker+customer_ticker, "
                "OR supplier_name+customer_name, OR supplier_id+customer_id"
            )

        seen: set[tuple[int,int]] = set()

        for line_no, raw_row in enumerate(reader, start=2):
            row = {k.strip().lower(): v for k, v in raw_row.items()}

            # Skip comment rows (first cell starts with #)
            first_val = next(iter(row.values()), "")
            if str(first_val).strip().startswith("#"):
                continue

            sup_val = row.get(sup_col, "").strip()
            cus_val = row.get(cus_col, "").strip()

            if not sup_val or not cus_val:
                continue

            sup = lookup(sup_val)
            cus = lookup(cus_val)

            if sup is None:
                warnings.warn(f"edges.csv line {line_no}: unknown supplier '{sup_val}' — skipping")
                skipped += 1
                continue
            if cus is None:
                warnings.warn(f"edges.csv line {line_no}: unknown customer '{cus_val}' — skipping")
                skipped += 1
                continue
            if sup.id == cus.id:
                continue

            pair = (sup.id, cus.id)
            if pair in seen:
                continue
            seen.add(pair)

            strength_raw = (
                row.get("relationship_strength")
                or row.get("strength")
                or row.get("weight")
                or "1.0"
            )
            strength = _float(strength_raw, 1.0)
            if strength > 1.0:        # assume it's a percentage 0-100
                strength /= 100.0
            strength = max(0.0, min(1.0, strength))

            edges.append(Edge(
                supplier_id=sup.id,
                customer_id=cus.id,
                relationship_strength=round(strength, 4),
            ))
            sup.customer_ids.append(cus.id)
            cus.supplier_ids.append(sup.id)

    print(f"[CSV] Loaded {len(edges)} edges from {path} ({skipped} skipped)")
    return edges
