# run_component_ablation.py
#
# Component Ablation Study — Proves Each MGNN Part is Necessary
# ==============================================================
# Trains 7 ablation variants and records test Macro F1.
# All variants use identical train/val/test splits, epochs, and hardware.
#
# Variants:
#   0. Full MGNN v2 (baseline)
#   1. No bypass path (GNN-only)
#   2. No GNN path (bypass MLP only = deep tabular)
#   3. No Temporal edges (Relation-1 similarity only)
#   4. No Similarity edges (Relation-0 temporal only)
#   5. No focal loss / no class weighting (uniform)
#   6. 1-layer GNN only (single-hop)
#   7. 2-layer GNN (original v1 depth)

import argparse
import json
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score, accuracy_score

p = argparse.ArgumentParser()
p.add_argument("--input-dir",  default="artifacts/phase4_pyg")
p.add_argument("--model-dir",  default="artifacts/phase5_model")
p.add_argument("--output",     default="artifacts/research/component_ablation.json")
p.add_argument("--epochs",     type=int, default=10)
p.add_argument("--batch-size", type=int, default=1024)
args = p.parse_args()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading graph data...")
x          = torch.load(f"{args.input_dir}/x.pt",          map_location="cpu", weights_only=False).float()
y          = torch.load(f"{args.input_dir}/y.pt",          map_location="cpu", weights_only=False).long()
edge_index = torch.load(f"{args.input_dir}/edge_index.pt", map_location="cpu", weights_only=False).long()
edge_type  = torch.load(f"{args.input_dir}/edge_type.pt",  map_location="cpu", weights_only=False).long()

norm    = torch.load(f"{args.model_dir}/feature_normalization.pt", map_location="cpu", weights_only=False)
splits  = torch.load(f"{args.model_dir}/split_indices.pt",         map_location="cpu", weights_only=False)
x       = (x - norm["mean"]) / norm["std"]

N           = x.shape[0]
num_classes = int(y.max().item()) + 1
F_dim       = x.shape[1]
train_idx   = splits["train_idx"]
val_idx     = splits["val_idx"]
test_idx    = splits["test_idx"]

data_full = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)

# Class weights (standard)
cc = torch.bincount(y[train_idx], minlength=num_classes).float().clamp_min(1)
cw_std = (cc.sum() / (cc * num_classes)).pow(0.5)
cw_std = (cw_std / cw_std.mean()).to(DEVICE)

# ─── Flexible model class for ablations ──────────────────────────────────────
class AblationMGNN(nn.Module):
    """
    Configurable MGNN for ablation experiments.
    gnn_layers : int  — 0 (bypass only), 1, 2, or 3
    use_bypass  : bool — include tabular bypass path
    focal_gamma : float — 0.0 = plain CE
    """
    class _SAGEGate(nn.Module):
        def __init__(self, in_c, out_c, num_relations, dropout):
            super().__init__()
            self.convs = nn.ModuleList([pyg_nn.SAGEConv(in_c, out_c) for _ in range(num_relations)])
            self.gate  = nn.Linear(in_c, num_relations)
            self.norm  = nn.LayerNorm(out_c)
            self.drop  = dropout
        def forward(self, h, edge_index, edge_type):
            gw = torch.softmax(self.gate(h), dim=-1)
            nr = len(self.convs)
            outs = []
            for r in range(nr):
                m = edge_type == r
                outs.append(self.convs[r](h, edge_index[:, m]) if m.any()
                            else torch.zeros(h.size(0), self.convs[r].out_channels,
                                             device=h.device, dtype=h.dtype))
            o = sum(gw[:, r:r+1] * outs[r] for r in range(nr))
            return F.relu(self.norm(o))

    def __init__(self, in_c, hidden, out_c, num_relations=2, gnn_layers=3,
                 use_bypass=True, dropout=0.2):
        super().__init__()
        self.gnn_layers = gnn_layers
        self.use_bypass = use_bypass
        self.proj = nn.Sequential(
            nn.Linear(in_c, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        if use_bypass:
            self.bypass = nn.Sequential(
                nn.Linear(in_c, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, out_c))
        if gnn_layers >= 1:
            self.l1 = self._SAGEGate(hidden, hidden, num_relations, dropout)
        if gnn_layers >= 2:
            self.l2 = self._SAGEGate(hidden, hidden, num_relations, dropout)
        if gnn_layers >= 3:
            self.l3 = self._SAGEGate(hidden, hidden, num_relations, dropout)
        self.lin = nn.Linear(hidden, out_c)

    def forward(self, x, edge_index, edge_type):
        bp = self.bypass(x) if self.use_bypass else 0
        h  = self.proj(x)
        if self.gnn_layers >= 1:
            h = h + self.l1(h, edge_index, edge_type)
        if self.gnn_layers >= 2:
            h = h + self.l2(h, edge_index, edge_type)
        if self.gnn_layers >= 3:
            h = h + self.l3(h, edge_index, edge_type)
        return self.lin(h) + bp

def focal_ce(logits, targets, weights, gamma=2.0):
    if gamma <= 0:
        return F.cross_entropy(logits, targets, weight=weights)
    lp = F.log_softmax(logits, dim=1)
    tp = lp.gather(1, targets.unsqueeze(1)).squeeze(1)
    p  = tp.exp().clamp(1e-8, 1-1e-8)
    return (-(1-p)**gamma * weights[targets] * tp).mean()

def run_ablation(variant_name, data_variant, gnn_layers=3, use_bypass=True,
                 num_relations=2, focal_gamma=2.0, uniform_weights=False):
    print(f"\n{'='*50}")
    print(f"Ablation: {variant_name}")
    print(f"{'='*50}")

    model = AblationMGNN(
        in_c=F_dim, hidden=128, out_c=num_classes,
        num_relations=num_relations, gnn_layers=gnn_layers,
        use_bypass=use_bypass, dropout=0.2
    ).to(DEVICE)

    cw = torch.ones(num_classes, device=DEVICE) if uniform_weights else cw_std
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    train_loader = NeighborLoader(data_variant, input_nodes=train_idx,
                                  num_neighbors=[20, 15], batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
    val_loader   = NeighborLoader(data_variant, input_nodes=val_idx,
                                  num_neighbors=[20, 15], batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)
    test_loader  = NeighborLoader(data_variant, input_nodes=test_idx,
                                  num_neighbors=[20, 15], batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)

    best_val_f1 = 0.0
    best_state  = None
    patience    = 5
    no_improve  = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_type)
            loss = focal_ce(out[:batch.batch_size], batch.y[:batch.batch_size], cw, focal_gamma)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            del batch; torch.cuda.empty_cache()

        model.eval()
        preds_v, labels_v = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.edge_type)
                preds_v.extend(out[:batch.batch_size].argmax(1).cpu().numpy())
                labels_v.extend(batch.y[:batch.batch_size].cpu().numpy())
                del batch
        val_f1 = f1_score(labels_v, preds_v, average="macro", zero_division=0)
        print(f"  Epoch {epoch:02d} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = deepcopy(model.state_dict())
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("  Early stop")
                break

    # Test
    model.load_state_dict(best_state)
    model.eval()
    preds_t, labels_t = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.edge_type)
            preds_t.extend(out[:batch.batch_size].argmax(1).cpu().numpy())
            labels_t.extend(batch.y[:batch.batch_size].cpu().numpy())
            del batch

    test_f1  = f1_score(labels_t, preds_t, average="macro", zero_division=0)
    test_acc = accuracy_score(labels_t, preds_t)
    runtime  = time.time() - t_start
    print(f"  -> Test Macro F1: {test_f1:.4f} | Acc: {test_acc:.4f} | {runtime:.0f}s")

    return {
        "variant": variant_name,
        "gnn_layers": gnn_layers,
        "use_bypass": use_bypass,
        "num_relations": num_relations,
        "focal_gamma": focal_gamma,
        "uniform_weights": uniform_weights,
        "best_val_macro_f1": float(best_val_f1),
        "test_macro_f1": float(test_f1),
        "test_accuracy": float(test_acc),
        "runtime_sec": float(runtime),
    }

# ─── Build edge-masked data variants ─────────────────────────────────────────
mask_r0 = edge_type == 0  # temporal edges only
mask_r1 = edge_type == 1  # similarity edges only

data_temporal_only = Data(
    x=x, edge_index=edge_index[:, mask_r0],
    edge_type=edge_type[mask_r0], y=y)
data_similarity_only = Data(
    x=x, edge_index=edge_index[:, mask_r1],
    edge_type=edge_type[mask_r1], y=y)
# For single-relation ablations, remap edge_type to 0
data_temporal_only.edge_type   = torch.zeros(mask_r0.sum(), dtype=torch.long)
data_similarity_only.edge_type = torch.zeros(mask_r1.sum(), dtype=torch.long)

# ─── Run all ablations ────────────────────────────────────────────────────────
results = []

results.append(run_ablation("Full MGNN v2 (baseline)",    data_full, gnn_layers=3, use_bypass=True, num_relations=2, focal_gamma=2.0))
results.append(run_ablation("No bypass (GNN-only)",        data_full, gnn_layers=3, use_bypass=False, num_relations=2, focal_gamma=2.0))
results.append(run_ablation("No GNN (bypass MLP only)",    data_full, gnn_layers=0, use_bypass=True,  num_relations=2, focal_gamma=2.0))
results.append(run_ablation("No temporal edges (sim only)", data_similarity_only, gnn_layers=3, use_bypass=True, num_relations=1, focal_gamma=2.0))
results.append(run_ablation("No similarity edges (tmp only)", data_temporal_only, gnn_layers=3, use_bypass=True, num_relations=1, focal_gamma=2.0))
results.append(run_ablation("No focal loss (uniform CE)",   data_full, gnn_layers=3, use_bypass=True, num_relations=2, focal_gamma=0.0, uniform_weights=True))
results.append(run_ablation("1-layer GNN (single-hop)",    data_full, gnn_layers=1, use_bypass=True, num_relations=2, focal_gamma=2.0))
results.append(run_ablation("2-layer GNN (v1 depth)",      data_full, gnn_layers=2, use_bypass=True, num_relations=2, focal_gamma=2.0))

# ─── Save ─────────────────────────────────────────────────────────────────────
Path(args.output).write_text(json.dumps(results, indent=2))
print(f"\n[DONE] Ablation results saved to {args.output}")
print("\nSummary:")
print(f"{'Variant':<40} {'Test F1':>8}")
print("-" * 50)
for r in results:
    marker = " <- baseline" if "baseline" in r["variant"] else ""
    print(f"{r['variant']:<40} {r['test_macro_f1']:>8.4f}{marker}")
