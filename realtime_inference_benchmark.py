import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import RGCNConv


DROPOUT = 0.2

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
        conv_cls = self.WeightedRGCNConv if use_edge_weight else RGCNConv
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


class MLPBaseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)


def parse_args():
    p = argparse.ArgumentParser(description="Held-out real-time MGNN benchmark")
    p.add_argument("--input-dir", default="artifacts/phase4_pyg")
    p.add_argument("--model-path", default="artifacts/phase5_model/best_model.pt")
    p.add_argument("--normalization-path", default="")
    p.add_argument("--split-path", default="")
    p.add_argument("--output", default="artifacts/benchmarks/realtime_inference_report.json")
    p.add_argument("--batch-sizes", default="1,8,32,128,512")
    p.add_argument("--num-neighbors", default="20,15")
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--measure-steps", type=int, default=100)
    p.add_argument("--device", default="")
    p.add_argument("--baseline-hidden", type=int, default=64)
    p.add_argument("--baseline-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def latency_summary(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    return {
        "mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)), "p99_ms": float(np.percentile(arr, 99)),
    }


def benchmark_mgnn(model, data, eval_idx, batch_size, neighbors, warmup, steps, device):
    loader = NeighborLoader(data, input_nodes=eval_idx, num_neighbors=neighbors,
                            batch_size=batch_size, shuffle=False, num_workers=0)
    iterator = iter(loader)
    model.eval()
    inference_ms, end_to_end_ms, y_true, y_pred = [], [], [], []
    total_nodes = 0
    with torch.no_grad():
        for step in range(warmup + steps):
            e2e_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            batch = batch.to(device)
            infer_start = time.perf_counter()
            out = model(
                batch.x,
                batch.edge_index,
                batch.edge_type,
                batch.edge_attr if model.use_edge_weight else None,
            )
            preds = out[:batch.batch_size].argmax(dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_elapsed = (time.perf_counter() - infer_start) * 1000.0
            e2e_elapsed = (time.perf_counter() - e2e_start) * 1000.0
            if step >= warmup:
                inference_ms.append(infer_elapsed)
                end_to_end_ms.append(e2e_elapsed)
                y_pred.extend(preds.cpu().tolist())
                y_true.extend(batch.y[:batch.batch_size].cpu().tolist())
                total_nodes += int(batch.batch_size)
    total_sec = sum(end_to_end_ms) / 1000.0
    return {
        "model_latency": latency_summary(inference_ms),
        "end_to_end_latency": latency_summary(end_to_end_ms),
        "throughput_nodes_per_sec": float(total_nodes / total_sec) if total_sec else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
        "samples_scored": total_nodes,
    }


def train_mlp(model, x, y, train_idx, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loader = torch.utils.data.DataLoader(train_idx, batch_size=4096, shuffle=True,
                                         generator=torch.Generator().manual_seed(42))
    model.train()
    started = time.perf_counter()
    for _ in range(epochs):
        for idx in loader:
            idx = idx.to(device)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[idx]), y[idx]).backward()
            optimizer.step()
    return time.perf_counter() - started


def benchmark_mlp(model, x, y, eval_idx, batch_size, warmup, steps, device):
    model.eval()
    latencies, y_true, y_pred = [], [], []
    with torch.no_grad():
        for step, idx in enumerate(torch.split(eval_idx, batch_size)):
            if step >= warmup + steps:
                break
            idx = idx.to(device)
            started = time.perf_counter()
            preds = model(x[idx]).argmax(dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000.0
            if step >= warmup:
                latencies.append(elapsed)
                y_pred.extend(preds.cpu().tolist())
                y_true.extend(y[idx].cpu().tolist())
    total_nodes = len(y_true)
    total_sec = sum(latencies) / 1000.0
    summary = latency_summary(latencies)
    return {
        "model_latency": summary, "end_to_end_latency": summary,
        "throughput_nodes_per_sec": float(total_nodes / total_sec) if total_sec else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
        "samples_scored": total_nodes,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    input_dir = Path(args.input_dir)
    model_dir = Path(args.model_path).parent
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    neighbors = [int(x) for x in args.num_neighbors.split(",")]
    x = torch.load(input_dir / "x.pt", map_location="cpu", weights_only=False).float()
    y = torch.load(input_dir / "y.pt", map_location="cpu", weights_only=False).long()
    edge_index = torch.load(input_dir / "edge_index.pt", map_location="cpu", weights_only=False).long()
    edge_type = torch.load(input_dir / "edge_type.pt", map_location="cpu", weights_only=False).long()
    edge_weight = torch.load(input_dir / "edge_weight.pt", map_location="cpu", weights_only=False).float()
    norm_path = Path(args.normalization_path) if args.normalization_path else model_dir / "feature_normalization.pt"
    split_path = Path(args.split_path) if args.split_path else model_dir / "split_indices.pt"
    norm = torch.load(norm_path, map_location="cpu", weights_only=False)
    x = (x - norm["mean"]) / norm["std"]
    splits = torch.load(split_path, map_location="cpu", weights_only=False)
    train_idx, test_idx = splits["train_idx"].long(), splits["test_idx"].long()
    data = Data(x=x, y=y, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_weight)
    num_classes = int(y.max()) + 1
    summary_path = model_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    mgnn = MGNN(
        x.size(1), summary.get("hidden_dim", 64), num_classes, 2,
        enhanced=summary.get("enhanced_model", False),
        use_edge_weight=summary.get("use_edge_weight", False),
    ).to(device)
    mgnn.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    x_device, y_device = x.to(device), y.to(device)
    mlp = MLPBaseline(x.size(1), args.baseline_hidden, num_classes).to(device)
    baseline_train_sec = train_mlp(mlp, x_device, y_device, train_idx, args.baseline_epochs, device)
    results = {"mgnn": {}, "mlp_baseline": {}}
    for batch_size in batch_sizes:
        key = f"bs_{batch_size}"
        results["mgnn"][key] = benchmark_mgnn(
            mgnn, data, test_idx, batch_size, neighbors, args.warmup_steps, args.measure_steps, device)
        results["mlp_baseline"][key] = benchmark_mlp(
            mlp, x_device, y_device, test_idx, batch_size, args.warmup_steps, args.measure_steps, device)
    report = {
        "schema_version": "1.0",
        "system": {"device": str(device), "platform": platform.platform(),
                   "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                   "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "dataset": {"nodes": int(data.num_nodes), "edges": int(data.num_edges),
                    "features": int(data.num_features), "classes": num_classes,
                    "held_out_test_nodes": int(test_idx.numel())},
        "benchmark_config": {"batch_sizes": batch_sizes, "num_neighbors": neighbors,
                             "warmup_steps": args.warmup_steps, "measure_steps": args.measure_steps,
                             "baseline_epochs": args.baseline_epochs,
                             "baseline_training_sec": baseline_train_sec},
        "results": results,
        "comparison_protocol": {"primary_latency": "end_to_end_latency.p95_ms",
                                "primary_throughput": "throughput_nodes_per_sec",
                                "primary_quality": "macro_f1",
                                "rule": "Use identical test indices, hardware, batch size, warmup and measurement counts."},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "system", "batch_size", "end_to_end_p50_ms", "end_to_end_p95_ms",
            "end_to_end_p99_ms", "throughput_nodes_per_sec", "accuracy", "macro_f1",
            "device", "nodes", "edges",
        ])
        writer.writeheader()
        for system_name, system_results in results.items():
            for batch_key, metrics in system_results.items():
                writer.writerow({
                    "system": system_name,
                    "batch_size": int(batch_key.removeprefix("bs_")),
                    "end_to_end_p50_ms": metrics["end_to_end_latency"]["p50_ms"],
                    "end_to_end_p95_ms": metrics["end_to_end_latency"]["p95_ms"],
                    "end_to_end_p99_ms": metrics["end_to_end_latency"]["p99_ms"],
                    "throughput_nodes_per_sec": metrics["throughput_nodes_per_sec"],
                    "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                    "device": str(device), "nodes": int(data.num_nodes), "edges": int(data.num_edges),
                })
    print(f"Real-time benchmark report saved: {out}")
    print(f"Comparable CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
