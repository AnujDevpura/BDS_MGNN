# run_roc_analysis.py
# One-vs-Rest ROC Curves + AUC for all 15 CIC-IDS2017 Classes

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_curve, auc
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--data-dir",  default="artifacts/phase4_pyg")
p.add_argument("--model-dir", default="artifacts/phase5_model")
p.add_argument("--output",    default="artifacts/research/roc_auc.json")
p.add_argument("--batch-size", type=int, default=1024)
args = p.parse_args()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {
    0:"BENIGN", 1:"FTP-Patator", 2:"SSH-Patator", 3:"DoS Slowloris",
    4:"DoS Slowhttptest", 5:"DoS Hulk", 6:"DoS GoldenEye", 7:"Heartbleed",
    8:"Web Attack-Brute Force", 9:"Web Attack-XSS", 10:"Web Attack-SQL Injection",
    11:"Infiltration", 12:"Bot", 13:"PortScan", 14:"DDoS",
}

print("Loading data...")
x          = torch.load(f"{args.data_dir}/x.pt",          map_location="cpu", weights_only=False).float()
y          = torch.load(f"{args.data_dir}/y.pt",          map_location="cpu", weights_only=False).long()
edge_index = torch.load(f"{args.data_dir}/edge_index.pt", map_location="cpu", weights_only=False).long()
edge_type  = torch.load(f"{args.data_dir}/edge_type.pt",  map_location="cpu", weights_only=False).long()
norm       = torch.load(f"{args.model_dir}/feature_normalization.pt", map_location="cpu", weights_only=False)
splits     = torch.load(f"{args.model_dir}/split_indices.pt",         map_location="cpu", weights_only=False)

x = (x - norm["mean"]) / norm["std"]
num_classes = int(y.max().item()) + 1
run_summary = json.loads(Path(f"{args.model_dir}/run_summary.json").read_text()) \
    if Path(f"{args.model_dir}/run_summary.json").exists() else {}
hidden_dim = run_summary.get("hidden_dim", 64)

data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)

# Build model
class _RelGatedRGCNLayer(nn.Module):
    def __init__(self, in_c, out_c, num_relations=2, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([pyg_nn.SAGEConv(in_c, out_c) for _ in range(num_relations)])
        self.gate = nn.Linear(in_c, num_relations)
        self.norm = nn.LayerNorm(out_c)
    def forward(self, h, edge_index, edge_type):
        gate_weights = torch.softmax(self.gate(h), dim=-1)
        outs = []
        for r in range(len(self.convs)):
            mask = edge_type == r
            if mask.any():
                outs.append(self.convs[r](h, edge_index[:, mask]))
            else:
                outs.append(torch.zeros(h.size(0), self.convs[r].out_channels, device=h.device, dtype=h.dtype))
        out = sum(gate_weights[:, r:r+1] * outs[r] for r in range(len(self.convs)))
        return F.relu(self.norm(out))

class _MGNN(nn.Module):
    def __init__(self, in_c, hidden, out_c, num_relations=2):
        super().__init__()
        self.feature_proj = nn.Sequential(nn.Linear(in_c, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.2))
        self.bypass = nn.Sequential(
            nn.Linear(in_c, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, out_c))
        self.layer1 = _RelGatedRGCNLayer(hidden, hidden, num_relations)
        self.layer2 = _RelGatedRGCNLayer(hidden, hidden, num_relations)
        self.layer3 = _RelGatedRGCNLayer(hidden, hidden, num_relations)
        self.lin = nn.Linear(hidden, out_c)
    def forward(self, x, edge_index, edge_type):
        b = self.bypass(x)
        h = self.feature_proj(x)
        h = h + self.layer1(h, edge_index, edge_type)
        h = h + self.layer2(h, edge_index, edge_type)
        h = h + self.layer3(h, edge_index, edge_type)
        return self.lin(h) + b

model = _MGNN(x.shape[1], hidden_dim, num_classes).to(DEVICE)
state = torch.load(f"{args.model_dir}/best_model.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(state, strict=False)
model.eval()

# Run inference on test set
test_loader = NeighborLoader(data, input_nodes=splits["test_idx"],
                             num_neighbors=[20, 15], batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

all_probs, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        out = model(batch.x, batch.edge_index, batch.edge_type)
        probs = F.softmax(out[:batch.batch_size], dim=1).cpu().numpy()
        lbls  = batch.y[:batch.batch_size].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(lbls)
        del batch

all_probs  = np.array(all_probs)   # [N_test, num_classes]
all_labels = np.array(all_labels)  # [N_test]

# One-vs-Rest ROC per class
results = {"system": {"device": DEVICE, "hidden_dim": hidden_dim}, "classes": {}}
for c in range(num_classes):
    y_bin  = (all_labels == c).astype(int)
    y_prob = all_probs[:, c]
    support = int(y_bin.sum())
    if y_bin.sum() == 0:
        results["classes"][str(c)] = {
            "class_name": CLASS_NAMES.get(c, f"Class {c}"),
            "auc": None, "support": 0, "note": "No positive samples in test set"
        }
        continue
    fpr, tpr, thresholds = roc_curve(y_bin, y_prob)
    roc_auc = auc(fpr, tpr)
    # Downsample curve for JSON storage (max 200 points)
    stride = max(1, len(fpr) // 200)
    results["classes"][str(c)] = {
        "class_name": CLASS_NAMES.get(c, f"Class {c}"),
        "auc": float(roc_auc),
        "support": support,
        "fpr": fpr[::stride].tolist(),
        "tpr": tpr[::stride].tolist(),
    }
    print(f"  Class {c:2d} {CLASS_NAMES.get(c,'?'):<30} AUC={roc_auc:.4f}  support={support}")

Path(args.output).write_text(json.dumps(results, indent=2))
print(f"\n[DONE] ROC/AUC saved: {args.output}")
