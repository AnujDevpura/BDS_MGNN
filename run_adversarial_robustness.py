import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from xgboost import XGBClassifier

DROPOUT = 0.2

class MLP(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(features, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, classes),
        )

    def forward(self, x):
        return self.net(x)

class GraphSAGE(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super().__init__()
        self.conv1 = SAGEConv(features, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = torch.nn.Linear(hidden, classes)

    def forward(self, x, edge_index):
        x = F.dropout(F.relu(self.conv1(x, edge_index)), 0.2, self.training)
        return self.head(F.relu(self.conv2(x, edge_index)))

class MGNN(torch.nn.Module):
    class WeightedRGCNConv(torch.nn.Module):
        def __init__(self, in_channels, out_channels, num_relations):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(num_relations, in_channels, out_channels))
            self.root = torch.nn.Linear(in_channels, out_channels, bias=False)
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.reset_parameters()

        def reset_parameters(self):
            torch.nn.init.xavier_uniform_(self.weight)
            self.root.reset_parameters()
            torch.nn.init.zeros_(self.bias)

        def forward(self, x, edge_index, edge_type, edge_weight=None):
            out = self.root(x)
            num_nodes = x.size(0)
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)
            else:
                edge_weight = edge_weight.to(device=x.device, dtype=x.dtype)
            src_all, dst_all = edge_index
            for relation_id in range(self.weight.size(0)):
                rel_mask = edge_type == relation_id
                if not torch.any(rel_mask):
                    continue
                src = src_all[rel_mask]
                dst = dst_all[rel_mask]
                rel_x = x[src] @ self.weight[relation_id]
                rel_w = edge_weight[rel_mask].unsqueeze(1)
                rel_msg = rel_x * rel_w
                rel_out = x.new_zeros((num_nodes, rel_msg.size(1)))
                rel_out.index_add_(0, dst, rel_msg)
                rel_deg = x.new_zeros(num_nodes)
                rel_deg.index_add_(0, dst, edge_weight[rel_mask])
                rel_out = rel_out / rel_deg.clamp_min(1.0).unsqueeze(1)
                out = out + rel_out
            return out + self.bias

    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, use_edge_weight=False, enhanced=False):
        super().__init__()
        conv_cls = self.WeightedRGCNConv if use_edge_weight else pyg_nn.RGCNConv
        self.use_edge_weight = use_edge_weight
        self.feature_proj = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT)
        )
        self.bypass = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        self.conv1 = conv_cls(hidden_channels, hidden_channels, num_relations=num_relations)
        self.conv2 = conv_cls(hidden_channels, hidden_channels, num_relations=num_relations)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.norm1 = torch.nn.LayerNorm(hidden_channels) if enhanced else None
        self.norm2 = torch.nn.LayerNorm(hidden_channels) if enhanced else None

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        out_bypass = self.bypass(x)
        h = self.feature_proj(x)
        if edge_weight is not None and self.use_edge_weight:
            h = self.conv1(h, edge_index, edge_type, edge_weight)
        else:
            h = self.conv1(h, edge_index, edge_type)
        h = self.norm1(h) if self.norm1 is not None else h
        h = F.relu(h)
        h = F.dropout(h, p=DROPOUT, training=self.training)
        residual = h
        if edge_weight is not None and self.use_edge_weight:
            h = self.conv2(h, edge_index, edge_type, edge_weight)
        else:
            h = self.conv2(h, edge_index, edge_type)
        h = self.norm2(h) if self.norm2 is not None else h
        h = F.relu(h + residual) if self.norm2 is not None else F.relu(h)
        out_gnn = self.lin(h)
        return out_gnn + out_bypass

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="artifacts/phase4_pyg")
    p.add_argument("--model-path", default="artifacts/phase5_model/best_model.pt")
    p.add_argument("--output", default="artifacts/research/adversarial_robustness.json")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Graph Data...")
    INPUT_DIR = args.input_dir
    PHASE5_DIR = str(Path(args.model_path).parent)
    
    x = torch.load(f"{INPUT_DIR}/x.pt", map_location="cpu", weights_only=False).float()
    y = torch.load(f"{INPUT_DIR}/y.pt", map_location="cpu", weights_only=False).long()
    edge_index = torch.load(f"{INPUT_DIR}/edge_index.pt", map_location="cpu", weights_only=False).long()
    edge_type = torch.load(f"{INPUT_DIR}/edge_type.pt", map_location="cpu", weights_only=False).long()
    
    try:
        edge_weight = torch.load(f"{INPUT_DIR}/edge_weight.pt", map_location="cpu", weights_only=False).float()
    except Exception:
        edge_weight = None

    normalization = torch.load(f"{PHASE5_DIR}/feature_normalization.pt", map_location="cpu", weights_only=False)
    splits = torch.load(f"{PHASE5_DIR}/split_indices.pt", map_location="cpu", weights_only=False)
    
    x = (x - normalization["mean"]) / normalization["std"]
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)
    if edge_weight is not None:
        data.edge_weight = edge_weight
        
    num_features = x.size(1)
    num_classes = int(y.max().item()) + 1
    
    print("Initializing MGNN Model...")
    mgnn = MGNN(in_channels=num_features, hidden_channels=64, out_channels=num_classes, num_relations=2, use_edge_weight=False, enhanced=False).to(device)
    mgnn.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False), strict=False)
    mgnn.eval()

    print("Training baseline XGBoost...")
    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method="hist", device="cpu", n_jobs=-1)
    train_idx = splits["train_idx"]
    test_idx = splits["test_idx"]
    xgb_model.fit(data.x[train_idx].numpy(), data.y[train_idx].numpy())

    print("Training baseline MLP...")
    mlp = MLP(num_features, 64, num_classes).to(device)
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    mlp.train()
    for _ in range(5):
        optimizer_mlp.zero_grad()
        loss = F.cross_entropy(mlp(data.x[train_idx].to(device)), data.y[train_idx].to(device))
        loss.backward()
        optimizer_mlp.step()
    mlp.eval()

    print("Training baseline GraphSAGE...")
    graphsage = GraphSAGE(num_features, 64, num_classes).to(device)
    optimizer_sage = torch.optim.Adam(graphsage.parameters(), lr=1e-3)
    graphsage.train()
    # Mini-batch training for GraphSAGE
    train_loader = NeighborLoader(data, input_nodes=train_idx, num_neighbors=[20, 15], batch_size=4096, shuffle=True)
    for _ in range(3):
        for batch in train_loader:
            batch = batch.to(device)
            optimizer_sage.zero_grad()
            loss = F.cross_entropy(graphsage(batch.x, batch.edge_index)[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            optimizer_sage.step()
    graphsage.eval()

    print("Running Adversarial Grid Search...")
    noise_levels = np.arange(0.0, 5.5, 0.5)
    
    results = []
    
    clean_test_x = data.x[test_idx].clone()
    y_true = data.y[test_idx].numpy()
    
    for noise in noise_levels:
        print(f"  Evaluating Noise Level: {noise:.1f}")
        noise_tensor = torch.randn_like(clean_test_x) * noise
        noisy_x = clean_test_x + noise_tensor
        
        # XGBoost Inference
        xgb_preds = xgb_model.predict(noisy_x.numpy())
        xgb_f1 = f1_score(y_true, xgb_preds, average="macro", zero_division=0)
        
        # MLP Inference
        with torch.no_grad():
            mlp_preds = mlp(noisy_x.to(device)).argmax(dim=1).cpu().numpy()
        mlp_f1 = f1_score(y_true, mlp_preds, average="macro", zero_division=0)
        
        # Graph Models Inference (requires message passing on noisy nodes)
        data.x[test_idx] = noisy_x
        loader = NeighborLoader(data, input_nodes=test_idx, num_neighbors=[20, 15], batch_size=4096, shuffle=False, num_workers=0)
        
        mgnn_preds = []
        graphsage_preds = []
        with torch.no_grad():
            for subgraph in loader:
                subgraph = subgraph.to(device)
                
                # MGNN
                out_mgnn = mgnn(subgraph.x, subgraph.edge_index, subgraph.edge_type)
                mgnn_preds.extend(out_mgnn[:subgraph.batch_size].argmax(dim=1).cpu().numpy())
                
                # GraphSAGE
                out_sage = graphsage(subgraph.x, subgraph.edge_index)
                graphsage_preds.extend(out_sage[:subgraph.batch_size].argmax(dim=1).cpu().numpy())
                
        mgnn_f1 = f1_score(y_true, mgnn_preds, average="macro", zero_division=0)
        graphsage_f1 = f1_score(y_true, graphsage_preds, average="macro", zero_division=0)
        
        results.append({
            "noise_level": float(noise),
            "mgnn_f1": float(mgnn_f1),
            "xgboost_f1": float(xgb_f1),
            "mlp_f1": float(mlp_f1),
            "graphsage_f1": float(graphsage_f1)
        })
        
    # Restore clean features
    data.x[test_idx] = clean_test_x
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Adversarial research saved to {out_path}")

if __name__ == "__main__":
    main()
