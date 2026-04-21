#!/usr/bin/env python3
"""
app.py — Flask web server for the Supply Chain GNN UI.
    python app.py  →  http://localhost:5000
"""

import json, os, sys
import torch
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.gnn import SupplyChainGNN, DEVICE, DTYPE, to_device
from data.dataset import (
    load_graph, generate_companies, generate_edges, inject_forecasts,
    save_graph, build_tensors, NODE_FEAT_DIM, SECTORS,
)

CHECKPOINT = "models/checkpoint.pt"
GRAPH_FILE  = "data/graph.json"

app = Flask(__name__)

companies = []
edges     = []
model     = None
_cached_tensors = None

# Sector-level forecast defaults (used when a company has no individual forecast)
SECTOR_FORECASTS = {
    "Semiconductors": {"forecast": 0.45, "months": 12},
    "Software":       {"forecast": 0.30, "months": 12},
    "Automotive":     {"forecast": 0.12, "months": 24},
    "Aerospace":      {"forecast": 0.10, "months": 18},
    "Healthcare":     {"forecast": 0.14, "months": 18},
    "Energy":         {"forecast": 0.08, "months": 12},
    "Retail":         {"forecast": 0.10, "months": 12},
    "Financials":     {"forecast": 0.09, "months": 12},
    "Materials":      {"forecast": 0.11, "months": 18},
    "Industrials":    {"forecast": 0.13, "months": 12},
}


def get_forecast(company):
    """Return (forecast, months, is_company_specific)"""
    if company.sector_growth_forecast is not None:
        return company.sector_growth_forecast, company.forecast_horizon_months or 12, True
    sf = SECTOR_FORECASTS.get(company.sector, {})
    return sf.get("forecast", 0.10), sf.get("months", 12), False


def load_or_generate_graph():
    global companies, edges
    if os.path.exists(GRAPH_FILE):
        companies, edges = load_graph(GRAPH_FILE)
    else:
        companies = generate_companies(40)
        edges     = generate_edges(companies, avg_edges_per_node=3)
        inject_forecasts(companies, n_forecasts=8)
        os.makedirs("data", exist_ok=True)
        save_graph(companies, edges, GRAPH_FILE)
    print(f"[Graph] {len(companies)} companies, {len(edges)} edges")


def load_model():
    global model
    hidden, layers = 128, 3
    if os.path.exists(CHECKPOINT):
        ckpt   = torch.load(CHECKPOINT, map_location=DEVICE)
        cfg    = ckpt.get("config", {})
        hidden = cfg.get("hidden", 128)
        layers = cfg.get("layers", 3)
        model  = SupplyChainGNN(NODE_FEAT_DIM, hidden, layers).to(device=DEVICE, dtype=DTYPE)
        model.load_state_dict(ckpt["model_state"])
        print(f"[Model] Loaded hidden={hidden} layers={layers} loss={ckpt.get('best_loss','?')}")
    else:
        print("[Model] No checkpoint — using untrained model")
        model = SupplyChainGNN(NODE_FEAT_DIM, hidden, layers).to(device=DEVICE, dtype=DTYPE)
    model.eval()


def get_tensors():
    global _cached_tensors
    if _cached_tensors is None:
        _cached_tensors = build_tensors(companies, edges)
    return _cached_tensors


def run_inference(shock_overrides=None):
    node_feats, adj, upstream_adj, downstream_adj, shock = get_tensors()
    shock = shock.clone()
    if shock_overrides:
        for cid, val in shock_overrides.items():
            shock[int(cid), 0] = float(val)
    with torch.no_grad():
        preds = model(to_device(node_feats), to_device(upstream_adj),
                      to_device(downstream_adj), to_device(shock))
    return preds.float().cpu().squeeze(-1).tolist()


def company_dict(c, impact):
    forecast, months, specific = get_forecast(c)
    return {
        "id":            c.id,
        "name":          c.name,
        "ticker":        c.ticker,
        "sector":        c.sector,
        "revenue_bn":    c.revenue_bn,
        "margin":        round(c.margin * 100, 2),       # stored as decimal → send as %
        "debt_ratio":    c.debt_ratio,
        "yoy_growth":    round(c.yoy_growth * 100, 2),   # stored as decimal → send as %
        "market_cap_bn": c.market_cap_bn,
        "r_and_d_pct":   round(c.r_and_d_pct * 100, 2),
        "capex_pct":     round(c.capex_pct * 100, 2),
        "suppliers":     c.supplier_ids,
        "customers":     c.customer_ids,
        "forecast":      round(forecast * 100, 1),        # send as % (e.g. 45.0)
        "forecast_months": months,
        "forecast_specific": specific,                    # True = company-level, False = sector default
        "impact":        round(impact, 5),
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/graph")
def api_graph():
    impacts = run_inference()
    nodes = [company_dict(c, impacts[c.id]) for c in companies]
    edges_out = [{"source": e.supplier_id, "target": e.customer_id,
                  "strength": e.relationship_strength} for e in edges]
    return jsonify({"nodes": nodes, "edges": edges_out})


@app.route("/api/company/<int:cid>")
def api_company(cid):
    if cid < 0 or cid >= len(companies):
        return jsonify({"error": "not found"}), 404
    impacts = run_inference()
    c = companies[cid]

    up_strength   = {e.supplier_id: e.relationship_strength for e in edges if e.customer_id == cid}
    down_strength = {e.customer_id: e.relationship_strength for e in edges if e.supplier_id == cid}

    suppliers = [{**company_dict(companies[s], impacts[s]), "strength": up_strength.get(s, 0)}
                 for s in c.supplier_ids if s < len(companies)]
    customers = [{**company_dict(companies[cu], impacts[cu]), "strength": down_strength.get(cu, 0)}
                 for cu in c.customer_ids if cu < len(companies)]

    d = company_dict(c, impacts[c.id])
    d["suppliers"] = suppliers
    d["customers"] = customers
    return jsonify(d)


@app.route("/api/shock", methods=["POST"])
def api_shock():
    body = request.get_json(silent=True) or {}
    shocks = {}

    sector     = body.get("sector")
    sector_val = float(body.get("sector_growth", 0.0)) / 100.0  # UI sends as %
    if sector:
        for c in companies:
            if c.sector == sector:
                shocks[c.id] = sector_val

    for k, v in (body.get("shocks") or {}).items():
        shocks[int(k)] = float(v) / 100.0

    baseline = run_inference()
    new_vals  = run_inference(shocks)

    shocked_ids = set(shocks.keys())
    upstream_set, downstream_set = set(), set()
    for e in edges:
        if e.customer_id in shocked_ids: upstream_set.add(e.supplier_id)
        if e.supplier_id in shocked_ids: downstream_set.add(e.customer_id)

    results = []
    for c in companies:
        delta = new_vals[c.id] - baseline[c.id]
        relation = ("direct" if c.id in shocked_ids else
                    "upstream" if c.id in upstream_set else
                    "downstream" if c.id in downstream_set else "indirect")
        results.append({
            "id": c.id, "name": c.name, "ticker": c.ticker,
            "sector": c.sector, "revenue_bn": c.revenue_bn,
            "impact": round(new_vals[c.id] * 100, 3),
            "delta":  round(delta * 100, 3),
            "relation": relation,
        })

    results.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return jsonify({"results": results})


@app.route("/api/sectors")
def api_sectors():
    impacts = run_inference()
    sector_data = {}
    for c in companies:
        s = c.sector
        f, m, spec = get_forecast(c)
        if s not in sector_data:
            sf = SECTOR_FORECASTS.get(s, {})
            sector_data[s] = {
                "companies": [],
                "avg_impact": 0,
                "total_market_cap": 0,
                "total_revenue": 0,
                "sector_forecast": round(sf.get("forecast", 0.10) * 100, 1),
                "sector_months": sf.get("months", 12),
            }
        sector_data[s]["companies"].append({
            "id": c.id, "name": c.name, "ticker": c.ticker,
            "impact": round(impacts[c.id] * 100, 3),
            "market_cap_bn": c.market_cap_bn,
            "revenue_bn": c.revenue_bn,
        })
        sector_data[s]["total_market_cap"] += c.market_cap_bn
        sector_data[s]["total_revenue"] += c.revenue_bn

    for s in sector_data:
        imps = [x["impact"] for x in sector_data[s]["companies"]]
        sector_data[s]["avg_impact"] = round(sum(imps) / len(imps), 3)
        sector_data[s]["company_count"] = len(sector_data[s]["companies"])

    return jsonify(sector_data)


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").lower()
    if not q: return jsonify([])
    return jsonify([
        {"id": c.id, "name": c.name, "ticker": c.ticker, "sector": c.sector}
        for c in companies
        if q in c.name.lower() or q in c.ticker.lower() or q in c.sector.lower()
    ][:10])


if __name__ == "__main__":
    load_or_generate_graph()
    load_model()
    print("\n[Server] http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
