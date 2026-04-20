# Supply Chain Intelligence GNN

Graph Neural Network for supply chain impact propagation.
Runs on **Intel Arc XPU** (B580) with **BF16** precision via PyTorch.

---

## Setup (your existing venv)

```bash
source /home/name/torch-arc/bin/activate
cd supply_chain_gnn
pip install flask        # torch is already in your venv
```

> **Note:** Your `torch-arc` venv already has PyTorch with XPU support.
> The code auto-detects XPU → CUDA → CPU in that order.

---

## Run order

### 1. Train the model

```bash
(torch-arc) ~/supply_chain_gnn$ python train.py
```

Optional flags:
```
--epochs 500     # more epochs = better convergence (default 300)
--lr 5e-4        # learning rate
--hidden 256     # larger model
--layers 4       # more GNN layers = deeper propagation
--save models/checkpoint.pt
```

What happens:
- Generates 40 synthetic companies across 10 sectors
- Builds a directed supply chain graph (~120 edges)
- Injects sector growth forecasts into 8 random companies
- Trains the GNN on XPU with BF16
- Saves checkpoint + graph JSON

### 2. Start the UI server

```bash
(torch-arc) ~/supply_chain_gnn$ python app.py
```

### 3. Open your browser

```
http://localhost:5000
```

---

## UI Tabs

### Graph
- Interactive force-directed graph of all companies
- Node size = GNN impact score
- Node color = sector
- Click any node → detail panel with metrics, upstream suppliers, downstream customers
- Sidebar search: filter by name, ticker, or sector

### Shock Propagation
- Pick a sector (e.g. "Semiconductors")
- Set a growth signal (e.g. +50%)
- Hit **Run Propagation**
- See every company ranked by GNN-predicted revenue impact delta
- Click any result to jump to that company in the graph

### Sectors
- All sectors with avg GNN impact scores
- Click any ticker chip to open that company's detail

---

## Architecture

```
SupplyChainGNN
├── Encoder:       linear(feat_dim → hidden)
├── Upstream GCN:  3× GraphConvLayer  (supplier direction)
├── Downstream GCN:3× GraphConvLayer  (customer direction)
├── Shock proj:    linear(1 → hidden)
└── Head:          linear(hidden*3 → 1)  → revenue_impact
```

Node features (20-dim):
- Revenue, margin, debt ratio, YoY growth, market cap, R&D%, CapEx%
- Supplier count, customer count, growth forecast
- Sector one-hot (10 sectors)

BF16 is used on XPU/CUDA automatically; falls back to FP32 on CPU.

---

## Files

```
supply_chain_gnn/
├── train.py          ← train the GNN
├── app.py            ← Flask UI server
├── models/
│   └── gnn.py        ← SupplyChainGNN, GraphConvLayer
├── data/
│   └── dataset.py    ← Company graph, feature engineering
├── templates/
│   └── index.html    ← Dashboard UI
└── requirements.txt
```
