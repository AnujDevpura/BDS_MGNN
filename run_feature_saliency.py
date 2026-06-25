# run_feature_saliency.py
# Gradient-Based Feature Saliency for MGNN v2
# Computes the average absolute gradient of the loss with respect to input features for each class.

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import torch.nn as nn
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="artifacts/phase4_pyg")
p.add_argument("--model-dir", default="artifacts/phase5_model")
p.add_argument("--output", default="artifacts/research/feature_saliency.json")
p.add_argument("--batch-size", type=int, default=512)
args = p.parse_args()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {
    0:"BENIGN", 1:"FTP-Patator", 2:"SSH-Patator", 3:"DoS Slowloris",
    4:"DoS Slowhttptest", 5:"DoS Hulk", 6:"DoS GoldenEye", 7:"Heartbleed",
    8:"Web Attack-Brute Force", 9:"Web Attack-XSS", 10:"Web Attack-SQL Injection",
    11:"Infiltration", 12:"Bot", 13:"PortScan", 14:"DDoS",
}

# The 78 original features in CIC-IDS2017
# This assumes a standard column ordering, usually found in phase 2 outputs
FEATURE_NAMES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets',
    'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
    'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length',
    'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
    'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
    'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
    'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
    'Idle Max', 'Idle Min'
]

print("Loading data...")
x          = torch.load(f"{args.data_dir}/x.pt",          map_location="cpu", weights_only=False).float()
y          = torch.load(f"{args.data_dir}/y.pt",          map_location="cpu", weights_only=False).long()
edge_index = torch.load(f"{args.data_dir}/edge_index.pt", map_location="cpu", weights_only=False).long()
edge_type  = torch.load(f"{args.data_dir}/edge_type.pt",  map_location="cpu", weights_only=False).long()
norm       = torch.load(f"{args.model_dir}/feature_normalization.pt", map_location="cpu", weights_only=False)
splits     = torch.load(f"{args.model_dir}/split_indices.pt",         map_location="cpu", weights_only=False)

x = (x - norm["mean"]) / norm["std"]
num_classes = int(y.max().item()) + 1
test_idx = splits["test_idx"]

run_summary = json.loads(Path(f"{args.model_dir}/run_summary.json").read_text()) \
    if Path(f"{args.model_dir}/run_summary.json").exists() else {}
hidden_dim = run_summary.get("hidden_dim", 64)

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

# We need gradients with respect to input x
data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)
loader = NeighborLoader(data, input_nodes=test_idx,
                        num_neighbors=[20, 15], batch_size=args.batch_size,
                        shuffle=False, num_workers=0)

print("\nComputing feature saliency via backpropagation...")
saliency_sums = torch.zeros((num_classes, x.shape[1]), device=DEVICE)
class_counts = torch.zeros(num_classes, device=DEVICE)

for batch in loader:
    batch = batch.to(DEVICE)
    batch.x.requires_grad_(True)
    
    out = model(batch.x, batch.edge_index, batch.edge_type)
    logits = out[:batch.batch_size]
    labels = batch.y[:batch.batch_size]
    
    # We compute the gradient of the correct class score w.r.t input features
    score = logits.gather(1, labels.unsqueeze(1)).sum()
    model.zero_grad()
    score.backward()
    
    grads = batch.x.grad[:batch.batch_size].abs()  # Absolute gradients
    
    for i in range(batch.batch_size):
        c = labels[i]
        saliency_sums[c] += grads[i]
        class_counts[c] += 1
    
    del batch

saliency_avg = saliency_sums / class_counts.unsqueeze(1).clamp(min=1)
saliency_avg = saliency_avg.cpu().numpy()

# Ensure we don't index out of bounds if there are more features than names
n_feats = min(len(FEATURE_NAMES), x.shape[1])
features_to_use = FEATURE_NAMES[:n_feats]

results = {"global": {}, "per_class": {}}

# Global average importance
global_avg = saliency_avg.mean(axis=0)
top_global = np.argsort(global_avg)[::-1][:20]
results["global"]["top_features"] = [
    {"feature": features_to_use[i], "importance": float(global_avg[i])} 
    for i in top_global if i < n_feats
]

# Per-class top 10
for c in range(num_classes):
    if class_counts[c].item() == 0:
        continue
    c_avg = saliency_avg[c]
    top_c = np.argsort(c_avg)[::-1][:10]
    results["per_class"][str(c)] = {
        "class_name": CLASS_NAMES.get(c, f"Class {c}"),
        "support": int(class_counts[c].item()),
        "top_features": [
            {"feature": features_to_use[i], "importance": float(c_avg[i])}
            for i in top_c if i < n_feats
        ]
    }

Path(args.output).write_text(json.dumps(results, indent=2))
print(f"[DONE] Feature Saliency saved: {args.output}")
