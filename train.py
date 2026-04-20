#!/usr/bin/env python3
"""
train.py — Train the Supply Chain GNN on Intel Arc XPU with BF16.

Synthetic data (default):
    python train.py

Real CSV data:
    python train.py --csv \
        --companies data/examples/companies.csv \
        --edges     data/examples/edges.csv

All flags:
    --epochs 300
    --lr     1e-3
    --hidden 128
    --layers 3
    --save   models/checkpoint.pt
    --graph  data/graph.json
    --csv                             (use CSV loader instead of synthetic data)
    --companies data/companies.csv
    --edges     data/edges.csv
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.gnn import SupplyChainGNN, DEVICE, DTYPE, to_device
from data.dataset import (
    generate_companies, generate_edges, inject_forecasts,
    build_tensors, save_graph, NODE_FEAT_DIM,
)


def synthetic_labels(companies, edges, shock):
    cust_map  = {c.id: c.customer_ids for c in companies}
    supp_map  = {c.id: c.supplier_ids for c in companies}
    shock_vec = [c.sector_growth_forecast or 0.0 for c in companies]
    labels = []
    for i in range(len(companies)):
        own  = shock_vec[i]
        ds = sum(shock_vec[j] for j in cust_map[i]) / max(len(cust_map[i]), 1)
        us = sum(shock_vec[j] for j in supp_map[i]) / max(len(supp_map[i]), 1)
        impact = 0.60 * own + 0.25 * ds + 0.15 * us
        impact += torch.randn(1).item() * 0.02
        labels.append([impact])
    return torch.tensor(labels, dtype=torch.float32)


def train(args):
    print(f"\n{'='*60}")
    print("  Supply Chain GNN — Training")
    print(f"{'='*60}")
    print(f"  Device : {DEVICE}  |  Precision: {'BF16' if DTYPE == torch.bfloat16 else 'FP32'}")
    print(f"  Epochs : {args.epochs}  |  LR: {args.lr}  |  Hidden: {args.hidden}")
    print(f"  Data   : {'CSV' if args.csv else 'Synthetic'}")
    print(f"{'='*60}\n")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    if args.csv:
        print("[1/4] Loading company graph from CSV files …")
        from data.csv_loader import load_companies_from_csv, load_edges_from_csv
        companies = load_companies_from_csv(args.companies)
        edges     = load_edges_from_csv(args.edges, companies)
        if not companies:
            print("ERROR: No companies loaded. Check your CSV.")
            sys.exit(1)
    else:
        print("[1/4] Generating synthetic company graph …")
        companies = generate_companies(n=40)
        edges     = generate_edges(companies, avg_edges_per_node=3)
        inject_forecasts(companies, n_forecasts=8)

    os.makedirs("data", exist_ok=True)
    save_graph(companies, edges, args.graph)
    print(f"      {len(companies)} companies, {len(edges)} edges → {args.graph}")

    node_feats, adj, upstream_adj, downstream_adj, shock = build_tensors(companies, edges)
    node_feats     = to_device(node_feats)
    upstream_adj   = to_device(upstream_adj)
    downstream_adj = to_device(downstream_adj)
    shock          = to_device(shock)
    labels         = to_device(synthetic_labels(companies, edges, shock))

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print("[2/4] Building model …")
    model = SupplyChainGNN(NODE_FEAT_DIM, args.hidden, args.layers).to(device=DEVICE, dtype=DTYPE)
    print(f"      Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.HuberLoss(delta=0.1)

    # ── 3. Training ───────────────────────────────────────────────────────────
    print(f"[3/4] Training for {args.epochs} epochs …\n")
    model.train()
    t0, best_loss = time.time(), float("inf")

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        preds = model(node_feats, upstream_adj, downstream_adj, shock)
        loss  = criterion(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        lv = loss.item()
        if lv < best_loss:
            best_loss = lv

        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"  Epoch {epoch:>5}/{args.epochs}  loss={lv:.6f}  "
                  f"best={best_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"{time.time()-t0:.1f}s")

    print(f"\n  Done in {time.time()-t0:.1f}s  |  best loss={best_loss:.6f}")

    # ── 4. Save ───────────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving → {args.save}")
    os.makedirs(os.path.dirname(args.save) if os.path.dirname(args.save) else ".", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": {"node_feat_dim": NODE_FEAT_DIM, "hidden": args.hidden, "layers": args.layers},
        "best_loss": best_loss,
    }, args.save)
    print("  Run  python app.py  to start the UI.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=300)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--hidden",    type=int,   default=128)
    p.add_argument("--layers",    type=int,   default=3)
    p.add_argument("--save",      type=str,   default="models/checkpoint.pt")
    p.add_argument("--graph",     type=str,   default="data/graph.json")
    p.add_argument("--csv",       action="store_true")
    p.add_argument("--companies", type=str,   default="data/examples/companies.csv")
    p.add_argument("--edges",     type=str,   default="data/examples/edges.csv")
    train(p.parse_args())
