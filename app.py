#!/usr/bin/env python3
"""
app.py — Flask web server for the Supply Chain GNN UI.

Start:
    python app.py
    → open http://localhost:5000 in your browser

Endpoints
─────────
GET  /                         → main dashboard HTML
GET  /api/graph                → full company graph JSON
GET  /api/company/<id>         → single company details + impact scores
POST /api/shock                → run propagation with custom shock values
GET  /api/sectors              → list sectors + companies per sector
GET  /api/search?q=...         → fuzzy company search
"""

import json
import os
import sys

import torch
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.gnn import SupplyChainGNN, DEVICE, DTYPE, to_device
from data.dataset import (
    load_graph, generate_companies, generate_edges, inject_forecasts,
    save_graph, build_tensors, NODE_FEAT_DIM, SECTORS,
)

# ─── Config ───────────────────────────────────────────────────────────────────

CHECKPOINT = "models/checkpoint.pt"
GRAPH_FILE  = "data/graph.json"

app = Flask(__name__)

# ─── State (loaded once at startup) ───────────────────────────────────────────

companies = []
edges     = []
model     = None
_cached_tensors = None


def load_or_generate_graph():
    global companies, edges
    if os.path.exists(GRAPH_FILE):
        print(f"[Graph] Loading from {GRAPH_FILE}")
        companies, edges = load_graph(GRAPH_FILE)
    else:
        print("[Graph] Generating synthetic graph …")
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
        print(f"[Model] Loading checkpoint from {CHECKPOINT}")
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        cfg  = ckpt.get("config", {})
        hidden = cfg.get("hidden", 128)
        layers = cfg.get("layers", 3)
        model  = SupplyChainGNN(NODE_FEAT_DIM, hidden, layers).to(device=DEVICE, dtype=DTYPE)
        model.load_state_dict(ckpt["model_state"])
        print(f"[Model] Loaded (hidden={hidden}, layers={layers}, best_loss={ckpt.get('best_loss','?')})")
    else:
        print("[Model] No checkpoint found — using untrained model (run train.py first)")
        model = SupplyChainGNN(NODE_FEAT_DIM, hidden, layers).to(device=DEVICE, dtype=DTYPE)
    model.eval()


def get_tensors():
    global _cached_tensors
    if _cached_tensors is None:
        _cached_tensors = build_tensors(companies, edges)
    return _cached_tensors


def run_inference(shock_overrides: dict | None = None) -> list[float]:
    """
    Run GNN inference.
    shock_overrides: {company_id: growth_float} — override specific shocks.
    Returns list of revenue_impact floats, one per company.
    """
    node_feats, adj, upstream_adj, downstream_adj, shock = get_tensors()

    # Apply overrides
    shock = shock.clone()
    if shock_overrides:
        for cid, val in shock_overrides.items():
            shock[int(cid), 0] = float(val)

    nf  = to_device(node_feats)
    ua  = to_device(upstream_adj)
    da  = to_device(downstream_adj)
    sh  = to_device(shock)

    with torch.no_grad():
        preds = model(nf, ua, da, sh)

    return preds.float().cpu().squeeze(-1).tolist()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/graph")
def api_graph():
    impacts = run_inference()

    nodes = []
    for c in companies:
        nodes.append({
            "id":          c.id,
            "name":        c.name,
            "ticker":      c.ticker,
            "sector":      c.sector,
            "revenue_bn":  c.revenue_bn,
            "margin":      c.margin,
            "debt_ratio":  c.debt_ratio,
            "yoy_growth":  c.yoy_growth,
            "market_cap_bn": c.market_cap_bn,
            "r_and_d_pct": c.r_and_d_pct,
            "capex_pct":   c.capex_pct,
            "suppliers":   c.supplier_ids,
            "customers":   c.customer_ids,
            "forecast":    c.sector_growth_forecast,
            "forecast_months": c.forecast_horizon_months,
            "impact":      round(impacts[c.id], 5),
        })

    edge_list = [
        {
            "source":   e.supplier_id,
            "target":   e.customer_id,
            "strength": e.relationship_strength,
        }
        for e in edges
    ]

    return jsonify({"nodes": nodes, "edges": edge_list})


@app.route("/api/company/<int:cid>")
def api_company(cid: int):
    if cid < 0 or cid >= len(companies):
        return jsonify({"error": "not found"}), 404

    impacts = run_inference()
    c = companies[cid]

    suppliers = [
        {"id": s, "name": companies[s].name, "ticker": companies[s].ticker,
         "sector": companies[s].sector, "impact": round(impacts[s], 5)}
        for s in c.supplier_ids if s < len(companies)
    ]
    customers = [
        {"id": cu, "name": companies[cu].name, "ticker": companies[cu].ticker,
         "sector": companies[cu].sector, "impact": round(impacts[cu], 5)}
        for cu in c.customer_ids if cu < len(companies)
    ]

    # Edge strengths for this company
    edge_strengths_up   = {}
    edge_strengths_down = {}
    for e in edges:
        if e.customer_id == cid:
            edge_strengths_up[e.supplier_id]  = e.relationship_strength
        if e.supplier_id == cid:
            edge_strengths_down[e.customer_id] = e.relationship_strength

    for s in suppliers:
        s["strength"] = edge_strengths_up.get(s["id"], 0)
    for cu in customers:
        cu["strength"] = edge_strengths_down.get(cu["id"], 0)

    return jsonify({
        "id":            c.id,
        "name":          c.name,
        "ticker":        c.ticker,
        "sector":        c.sector,
        "revenue_bn":    c.revenue_bn,
        "margin":        c.margin,
        "debt_ratio":    c.debt_ratio,
        "yoy_growth":    c.yoy_growth,
        "market_cap_bn": c.market_cap_bn,
        "r_and_d_pct":   c.r_and_d_pct,
        "capex_pct":     c.capex_pct,
        "forecast":      c.sector_growth_forecast,
        "forecast_months": c.forecast_horizon_months,
        "impact":        round(impacts[c.id], 5),
        "suppliers":     suppliers,
        "customers":     customers,
    })


@app.route("/api/shock", methods=["POST"])
def api_shock():
    """
    Body: {"shocks": {"0": 0.5, "12": 0.3}, "sector": "Semiconductors", "sector_growth": 0.5}
    Returns list of {id, name, impact, delta} sorted by |delta| desc.
    """
    body = request.get_json(silent=True) or {}

    # Build shock map
    shocks = {}

    # Sector-wide shock
    sector      = body.get("sector")
    sector_val  = body.get("sector_growth", 0.0)
    if sector:
        for c in companies:
            if c.sector == sector:
                shocks[c.id] = float(sector_val)

    # Per-company overrides
    for k, v in (body.get("shocks") or {}).items():
        shocks[int(k)] = float(v)

    # Baseline (no shock)
    baseline = run_inference()
    new_vals  = run_inference(shocks)

    results = []
    for c in companies:
        results.append({
            "id":     c.id,
            "name":   c.name,
            "ticker": c.ticker,
            "sector": c.sector,
            "impact": round(new_vals[c.id], 5),
            "delta":  round(new_vals[c.id] - baseline[c.id], 5),
        })

    results.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return jsonify({"results": results, "shocks_applied": shocks})


@app.route("/api/sectors")
def api_sectors():
    sector_data = {}
    impacts = run_inference()
    for c in companies:
        s = c.sector
        if s not in sector_data:
            sector_data[s] = {"companies": [], "avg_impact": 0}
        sector_data[s]["companies"].append({
            "id": c.id, "name": c.name, "ticker": c.ticker,
            "impact": round(impacts[c.id], 5),
        })

    for s in sector_data:
        imps = [x["impact"] for x in sector_data[s]["companies"]]
        sector_data[s]["avg_impact"] = round(sum(imps) / len(imps), 5)

    return jsonify(sector_data)


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").lower()
    if not q:
        return jsonify([])
    results = []
    for c in companies:
        if q in c.name.lower() or q in c.ticker.lower() or q in c.sector.lower():
            results.append({"id": c.id, "name": c.name, "ticker": c.ticker, "sector": c.sector})
    return jsonify(results[:10])


# ─── Startup ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_or_generate_graph()
    load_model()
    print("\n[Server] Starting at http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
