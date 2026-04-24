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



@app.route("/api/supply_chain_stats/<int:cid>")
def api_supply_chain_stats(cid):
    if cid < 0 or cid >= len(companies):
        return jsonify({"error": "not found"}), 404
    c = companies[cid]
    impacts = run_inference()

    def bfs(start_ids, direction, hops=2):
        visited, frontier, results = set(start_ids), set(start_ids), []
        for _ in range(hops):
            nxt = set()
            for e in edges:
                if direction == "up" and e.customer_id in frontier and e.supplier_id not in visited:
                    nxt.add(e.supplier_id); visited.add(e.supplier_id)
                elif direction == "dn" and e.supplier_id in frontier and e.customer_id not in visited:
                    nxt.add(e.customer_id); visited.add(e.customer_id)
            for nid in nxt:
                if nid < len(companies):
                    co = companies[nid]; f, m, spec = get_forecast(co)
                    results.append({"id":co.id,"name":co.name,"ticker":co.ticker,
                        "sector":co.sector,"revenue_bn":co.revenue_bn,
                        "market_cap_bn":co.market_cap_bn,"margin":round(co.margin*100,2),
                        "yoy_growth":round(co.yoy_growth*100,2),
                        "forecast":round(f*100,1),"forecast_months":m,
                        "impact":round(impacts[nid]*100,3)})
            frontier = nxt
        return results

    up2 = bfs([cid], "up", 2)
    dn2 = bfs([cid], "dn", 2)
    all_chain = up2 + dn2
    sup_str = sum(e.relationship_strength for e in edges if e.customer_id == cid)
    cus_str = sum(e.relationship_strength for e in edges if e.supplier_id == cid)
    sec_exp = {}
    for co in all_chain:
        sec_exp[co["sector"]] = sec_exp.get(co["sector"], 0) + co["market_cap_bn"]

    return jsonify({
        "company": company_dict(c, impacts[cid]),
        "upstream_2hop": up2,
        "downstream_2hop": dn2,
        "supplier_concentration": round(sup_str, 3),
        "customer_concentration": round(cus_str, 3),
        "sector_exposure": {k: round(v,1) for k,v in sorted(sec_exp.items(), key=lambda x:-x[1])},
        "chain_avg_forecast": round(sum(co["forecast"] for co in all_chain)/max(len(all_chain),1),1) if all_chain else 0,
        "chain_avg_impact": round(sum(co["impact"] for co in all_chain)/max(len(all_chain),1),3) if all_chain else 0,
    })


import csv, io, tempfile, shutil
from werkzeug.utils import secure_filename

LIVE_CACHE_FILE = "data/live_cache.json"
COMPANIES_CSV   = "companies.csv"
EDGES_CSV       = "edges.csv"


def load_live_cache():
    if os.path.exists(LIVE_CACHE_FILE):
        with open(LIVE_CACHE_FILE) as f:
            return json.load(f)
    return {}


@app.route("/api/stocks")
def api_stocks():
    """Return live stock data merged with company graph data."""
    cache = load_live_cache()
    impacts = run_inference()
    result = []
    for c in companies:
        live = cache.get(c.ticker, {})
        forecast, months, specific = get_forecast(c)
        entry = {
            "id":           c.id,
            "name":         c.name,
            "ticker":       c.ticker,
            "sector":       c.sector,
            "revenue_bn":   c.revenue_bn,
            "market_cap_bn": c.market_cap_bn,
            "margin":       round(c.margin * 100, 2),
            "yoy_growth":   round(c.yoy_growth * 100, 2),
            "impact":       round(impacts[c.id] * 100, 3),
            "forecast":     round(forecast * 100, 1),
            # Live fields (None if not cached)
            "price":        live.get("price"),
            "change_pct":   live.get("change_pct"),
            "market_cap_live": live.get("market_cap_bn"),
            "week52_high":  live.get("52w_high"),
            "week52_low":   live.get("52w_low"),
            "volume":       live.get("volume"),
            "currency":     live.get("currency", "USD"),
            "hist_prices":  live.get("hist_prices", []),
            "quarterly":    live.get("quarterly", {}),
            "fetched_at":   live.get("fetched_at"),
            "live_error":   live.get("error"),
        }
        # Use live market cap if available
        if entry["market_cap_live"]:
            entry["market_cap_bn"] = entry["market_cap_live"]
        result.append(entry)
    return jsonify({
        "stocks": result,
        "cache_count": len(cache),
        "last_update": max((v.get("fetched_at","") for v in cache.values()), default=None),
    })


@app.route("/api/stock_detail/<ticker>")
def api_stock_detail(ticker):
    """Return detailed stock info for one ticker."""
    cache = load_live_cache()
    live = cache.get(ticker.upper(), {})
    c_match = next((c for c in companies if c.ticker == ticker.upper()), None)
    if not c_match:
        return jsonify({"error": "ticker not found in companies"}), 404
    impacts = run_inference()
    forecast, months, specific = get_forecast(c_match)
    return jsonify({
        "id": c_match.id,
        "name": c_match.name,
        "ticker": c_match.ticker,
        "sector": c_match.sector,
        "impact": round(impacts[c_match.id] * 100, 3),
        "forecast": round(forecast * 100, 1),
        **live,
    })


@app.route("/api/update_company_data", methods=["POST"])
def api_update_company_data():
    """
    Manually update a company's financial data from posted JSON.
    Body: { ticker, revenue_bn, margin, yoy_growth, market_cap_bn, ... }
    This edits companies.csv directly so next retrain picks it up.
    """
    body = request.get_json(silent=True) or {}
    ticker = body.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    updatable = ["revenue_bn","margin","yoy_growth","market_cap_bn","r_and_d_pct",
                 "capex_pct","debt_ratio","sector_growth_forecast","forecast_horizon_months"]
    if not os.path.exists(COMPANIES_CSV):
        return jsonify({"error": f"{COMPANIES_CSV} not found"}), 404

    rows, updated, fieldnames = [], False, []
    with open(COMPANIES_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            first = next(iter(row.values()), "")
            if first.strip().startswith("#"):
                rows.append(row); continue
            if row.get("ticker","").strip().upper() == ticker:
                for field in updatable:
                    if field in body and body[field] is not None and field in row:
                        row[field] = str(body[field])
                updated = True
            rows.append(row)

    if not updated:
        return jsonify({"error": f"ticker {ticker} not found in CSV"}), 404

    with open(COMPANIES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Invalidate tensor cache
    global _cached_tensors
    _cached_tensors = None
    return jsonify({"ok": True, "ticker": ticker, "message": "Updated. Retrain to apply changes."})


@app.route("/api/upload_company", methods=["POST"])
def api_upload_company():
    """
    Upload a CSV to add a new company (or update existing).
    Accepts multipart form with:
      file = CSV file (see companies_template.csv for format)
      edges = optional JSON string: [{"supplier":"TICK","customer":"TICK","strength":0.5}]
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    edges_json = request.form.get("edges", "[]")
    try:
        edges_data = json.loads(edges_json)
    except Exception:
        edges_data = []

    # Parse uploaded CSV
    content = f.read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))
    rows = [row for row in reader]
    if not rows:
        return jsonify({"error": "Empty CSV"}), 400

    added, updated_tickers = [], []
    existing = {}
    if os.path.exists(COMPANIES_CSV):
        with open(COMPANIES_CSV, newline="", encoding="utf-8-sig") as cf:
            r = csv.DictReader(cf)
            fieldnames = r.fieldnames or []
            all_rows = list(r)
        existing_tickers = {row.get("ticker","").strip().upper() for row in all_rows
                            if not next(iter(row.values()),"").strip().startswith("#")}
    else:
        fieldnames = ["name","ticker","sector","revenue_bn","margin","debt_ratio",
                      "yoy_growth","market_cap_bn","r_and_d_pct","capex_pct",
                      "sector_growth_forecast","forecast_horizon_months"]
        all_rows = []
        existing_tickers = set()

    for row in rows:
        r = {k.strip().lower(): v.strip() for k, v in row.items()}
        first = next(iter(r.values()), "")
        if first.startswith("#"): continue
        ticker = r.get("ticker", "").upper()
        if not ticker: continue
        new_row = {fn: r.get(fn.lower(), "") for fn in fieldnames}
        new_row["ticker"] = ticker
        if ticker in existing_tickers:
            # Update existing row
            for i, er in enumerate(all_rows):
                if er.get("ticker","").strip().upper() == ticker:
                    for k, v in new_row.items():
                        if v: er[k] = v
                    all_rows[i] = er; break
            updated_tickers.append(ticker)
        else:
            all_rows.append(new_row)
            added.append(ticker)

    # Write back companies.csv
    with open(COMPANIES_CSV, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Append edges
    if edges_data and os.path.exists(EDGES_CSV):
        with open(EDGES_CSV, "a", newline="", encoding="utf-8") as ef:
            from datetime import datetime as dt
            ef.write(f"\n# ── Uploaded edges {dt.now().strftime('%Y-%m-%d')} ──\n")
            for e in edges_data:
                sup = e.get("supplier","").upper()
                cus = e.get("customer","").upper()
                strength = e.get("strength", 0.5)
                if sup and cus:
                    ef.write(f"{sup},{cus},{strength}\n")

    global _cached_tensors
    _cached_tensors = None
    return jsonify({
        "ok": True,
        "added": added,
        "updated": updated_tickers,
        "edges_added": len(edges_data),
        "message": f"Added {len(added)}, updated {len(updated_tickers)} companies. Retrain to apply.",
    })


@app.route("/api/cache_status")
def api_cache_status():
    cache = load_live_cache()
    return jsonify({
        "count": len(cache),
        "tickers": list(cache.keys()),
        "last_update": max((v.get("fetched_at","") for v in cache.values()), default=None),
        "with_errors": [t for t, v in cache.items() if v.get("error")],
    })

if __name__ == "__main__":
    load_or_generate_graph()
    load_model()
    print("\n[Server] http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
