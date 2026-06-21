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
from torch_geometric.nn import GATConv, SAGEConv


class MLP(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(features, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, classes),
        )

    def forward(self, x):
        return self.net(x)


class TemporalCNN(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(features, hidden, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(hidden, hidden, 3, padding=1)
        self.head = torch.nn.Linear(hidden, classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)).mean(dim=2)
        return self.head(x)


class TemporalLSTM(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super().__init__()
        self.lstm = torch.nn.LSTM(features, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1])


class GraphSAGE(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super().__init__()
        self.conv1 = SAGEConv(features, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = torch.nn.Linear(hidden, classes)

    def forward(self, x, edge_index):
        x = F.dropout(F.relu(self.conv1(x, edge_index)), 0.2, self.training)
        return self.head(F.relu(self.conv2(x, edge_index)))


class GAT(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super().__init__()
        per_head = max(8, hidden // 4)
        self.conv1 = GATConv(features, per_head, heads=4, dropout=0.2)
        self.conv2 = GATConv(per_head * 4, hidden, heads=1, dropout=0.2)
        self.head = torch.nn.Linear(hidden, classes)

    def forward(self, x, edge_index):
        x = F.dropout(F.elu(self.conv1(x, edge_index)), 0.2, self.training)
        return self.head(F.elu(self.conv2(x, edge_index)))


def parse_args():
    p = argparse.ArgumentParser(description="Standard IDS model benchmark suite")
    p.add_argument("--input-dir", default="artifacts/phase4_pyg")
    p.add_argument("--phase5-dir", default="artifacts/phase5_model")
    p.add_argument("--realtime-report", default="artifacts/benchmarks/realtime_inference_report.json")
    p.add_argument("--output-dir", default="artifacts/model_benchmarks")
    p.add_argument("--models", default="xgboost,mlp,cnn,lstm,graphsage,gat")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--dense-batch-size", type=int, default=4096)
    p.add_argument("--graph-batch-size", type=int, default=1024)
    p.add_argument("--inference-batch-size", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--sequence-length", type=int, default=8)
    p.add_argument("--neighbors", default="20,15")
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--xgb-estimators", type=int, default=300)
    p.add_argument("--xgb-max-depth", type=int, default=8)
    p.add_argument("--xgb-device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def latency_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    return {k: float(np.percentile(values, p)) for k, p in [("p50_ms", 50), ("p95_ms", 95), ("p99_ms", 99)]}


def score(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def make_temporal_sequences(num_nodes, edge_index, edge_type, edge_weight, length):
    sequence = np.repeat(np.arange(num_nodes, dtype=np.int64)[:, None], length, axis=1)
    mask = edge_type.numpy() == 0
    src = edge_index[0].numpy()[mask]
    dst = edge_index[1].numpy()[mask]
    weights = edge_weight.numpy()[mask]
    order = np.lexsort((-weights, dst))
    src, dst = src[order], dst[order]
    starts = np.r_[0, np.flatnonzero(dst[1:] != dst[:-1]) + 1, len(dst)]
    group_lengths = np.diff(starts)
    ranks = np.arange(len(dst)) - np.repeat(starts[:-1], group_lengths)
    keep = ranks < length - 1
    sequence[dst[keep], ranks[keep]] = src[keep]
    return torch.from_numpy(sequence)


@torch.no_grad()
def evaluate_dense(model, x, y, indices, batch_size, device, input_fn):
    model.eval()
    preds, labels, latencies = [], [], []
    for idx in torch.split(indices, batch_size):
        started = time.perf_counter()
        logits = model(input_fn(idx).to(device))
        prediction = logits.argmax(1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
        preds.extend(prediction.cpu().tolist())
        labels.extend(y[idx].tolist())
    elapsed = sum(latencies) / 1000
    return {**score(labels, preds), **latency_stats(latencies),
            "throughput_nodes_per_sec": len(labels) / elapsed if elapsed else 0.0}


def train_dense(name, model, x, y, train_idx, val_idx, test_idx, class_weights,
                epochs, batch_size, inference_batch, patience, device, input_fn):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    best_f1, stale, best_state = -1.0, 0, None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    generator = torch.Generator().manual_seed(42)
    for epoch in range(epochs):
        model.train()
        permutation = train_idx[torch.randperm(len(train_idx), generator=generator)]
        for idx in torch.split(permutation, batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_fn(idx).to(device))
            loss = F.cross_entropy(logits, y[idx].to(device), weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val = evaluate_dense(model, x, y, val_idx, inference_batch, device, input_fn)
        print(f"{name} epoch={epoch + 1} val_macro_f1={val['macro_f1']:.4f}")
        if val["macro_f1"] > best_f1:
            best_f1, stale = val["macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    train_sec = time.perf_counter() - started
    model.load_state_dict(best_state)
    result = evaluate_dense(model, x, y, test_idx, inference_batch, device, input_fn)
    result.update({"model": name, "training_sec": train_sec, "best_val_macro_f1": best_f1,
                   "epochs_ran": epoch + 1,
                   "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0})
    return result


@torch.no_grad()
def evaluate_graph(model, loader, device):
    model.eval()
    labels, preds, latencies = [], [], []
    iterator = iter(loader)
    while True:
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch = batch.to(device)
        prediction = model(batch.x, batch.edge_index)[:batch.batch_size].argmax(1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000)
        labels.extend(batch.y[:batch.batch_size].cpu().tolist())
        preds.extend(prediction.cpu().tolist())
    elapsed = sum(latencies) / 1000
    return {**score(labels, preds), **latency_stats(latencies),
            "throughput_nodes_per_sec": len(labels) / elapsed if elapsed else 0.0}


def train_graph(name, model, data, train_idx, val_idx, test_idx, class_weights,
                neighbors, batch_size, epochs, patience, device):
    train_loader = NeighborLoader(data, input_nodes=train_idx, num_neighbors=neighbors,
                                  batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = NeighborLoader(data, input_nodes=val_idx, num_neighbors=neighbors,
                                batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = NeighborLoader(data, input_nodes=test_idx, num_neighbors=neighbors,
                                 batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    best_f1, stale, best_state = -1.0, 0, None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index)[:batch.batch_size]
            loss = F.cross_entropy(logits, batch.y[:batch.batch_size], weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val = evaluate_graph(model, val_loader, device)
        print(f"{name} epoch={epoch + 1} val_macro_f1={val['macro_f1']:.4f}")
        if val["macro_f1"] > best_f1:
            best_f1, stale = val["macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    train_sec = time.perf_counter() - started
    model.load_state_dict(best_state)
    result = evaluate_graph(model, test_loader, device)
    result.update({"model": name, "training_sec": train_sec, "best_val_macro_f1": best_f1,
                   "epochs_ran": epoch + 1,
                   "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0})
    return result


def run_xgboost(args, x, y, train_idx, val_idx, test_idx, class_weights):
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=args.xgb_estimators, max_depth=args.xgb_max_depth,
        learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
        device=args.xgb_device, random_state=args.seed, n_jobs=-1,
    )
    weights = class_weights.cpu().numpy()[y[train_idx].numpy()]
    started = time.perf_counter()
    model.fit(x[train_idx].numpy(), y[train_idx].numpy(), sample_weight=weights,
              eval_set=[(x[val_idx].numpy(), y[val_idx].numpy())], verbose=False)
    training_sec = time.perf_counter() - started
    preds, latencies = [], []
    for idx in torch.split(test_idx, args.inference_batch_size):
        batch = x[idx].numpy()
        tick = time.perf_counter()
        preds.extend(model.predict(batch).tolist())
        latencies.append((time.perf_counter() - tick) * 1000)
    elapsed = sum(latencies) / 1000
    return {"model": "XGBoost", **score(y[test_idx].tolist(), preds), **latency_stats(latencies),
            "throughput_nodes_per_sec": len(preds) / elapsed if elapsed else 0.0,
            "training_sec": training_sec, "peak_gpu_mem_gb": None,
            "epochs_ran": args.xgb_estimators, "best_val_macro_f1": None}


def load_rgcn_result(phase5_dir, realtime_report):
    summary = json.loads((phase5_dir / "run_summary.json").read_text())
    realtime = json.loads(Path(realtime_report).read_text())["results"]["mgnn"]["bs_512"]
    latency = realtime["end_to_end_latency"]
    return {"model": "Proposed RGCN", "accuracy": summary["test_accuracy"],
            "macro_f1": summary["test_macro_f1"], "training_sec": summary["runtime_sec"],
            "peak_gpu_mem_gb": summary["peak_gpu_mem_gb"], "epochs_ran": summary["epochs_ran"],
            "best_val_macro_f1": summary["best_val_macro_f1"],
            "p50_ms": latency["p50_ms"], "p95_ms": latency["p95_ms"], "p99_ms": latency["p99_ms"],
            "throughput_nodes_per_sec": realtime["throughput_nodes_per_sec"]}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir, phase5_dir = Path(args.input_dir), Path(args.phase5_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = {x.strip().lower() for x in args.models.split(",")}

    x = torch.load(input_dir / "x.pt", map_location="cpu", weights_only=False).float()
    y = torch.load(input_dir / "y.pt", map_location="cpu", weights_only=False).long()
    edge_index = torch.load(input_dir / "edge_index.pt", map_location="cpu", weights_only=False).long()
    edge_type = torch.load(input_dir / "edge_type.pt", map_location="cpu", weights_only=False).long()
    edge_weight = torch.load(input_dir / "edge_weight.pt", map_location="cpu", weights_only=False).float()
    normalization = torch.load(phase5_dir / "feature_normalization.pt", map_location="cpu", weights_only=False)
    splits = torch.load(phase5_dir / "split_indices.pt", map_location="cpu", weights_only=False)
    x = (x - normalization["mean"]) / normalization["std"]
    train_idx, val_idx, test_idx = splits["train_idx"], splits["val_idx"], splits["test_idx"]
    classes = int(y.max()) + 1
    counts = torch.bincount(y[train_idx], minlength=classes).float().clamp_min(1)
    class_weights = torch.sqrt(counts.sum() / (counts * classes)).to(device)
    results = []

    if "xgboost" in requested:
        results.append(run_xgboost(args, x, y, train_idx, val_idx, test_idx, class_weights))
    if "mlp" in requested:
        results.append(train_dense("MLP", MLP(x.size(1), args.hidden_dim, classes), x, y,
            train_idx, val_idx, test_idx, class_weights, args.epochs, args.dense_batch_size,
            args.inference_batch_size, args.patience, device, lambda idx: x[idx]))

    if requested.intersection({"cnn", "lstm"}):
        sequences = make_temporal_sequences(len(x), edge_index, edge_type, edge_weight, args.sequence_length)
        sequence_input = lambda idx: x[sequences[idx]]
        if "cnn" in requested:
            results.append(train_dense("1D-CNN", TemporalCNN(x.size(1), args.hidden_dim, classes), x, y,
                train_idx, val_idx, test_idx, class_weights, args.epochs, args.dense_batch_size,
                args.inference_batch_size, args.patience, device, sequence_input))
        if "lstm" in requested:
            results.append(train_dense("LSTM", TemporalLSTM(x.size(1), args.hidden_dim, classes), x, y,
                train_idx, val_idx, test_idx, class_weights, args.epochs, args.dense_batch_size,
                args.inference_batch_size, args.patience, device, sequence_input))

    data = Data(x=x, y=y, edge_index=edge_index)
    neighbors = [int(v) for v in args.neighbors.split(",")]
    if "graphsage" in requested:
        results.append(train_graph("GraphSAGE", GraphSAGE(x.size(1), args.hidden_dim, classes), data,
            train_idx, val_idx, test_idx, class_weights, neighbors, args.graph_batch_size,
            args.epochs, args.patience, device))
    if "gat" in requested:
        results.append(train_graph("GAT", GAT(x.size(1), args.hidden_dim, classes), data,
            train_idx, val_idx, test_idx, class_weights, neighbors, args.graph_batch_size,
            args.epochs, args.patience, device))
    results.append(load_rgcn_result(phase5_dir, args.realtime_report))

    report = {
        "protocol": {
            "split": "identical saved train/validation/test indices", "normalization": "training-only statistics",
            "temporal_models": f"incoming temporal-edge sequences, length={args.sequence_length}",
            "homogeneous_models": "same fused graph with relation IDs removed",
            "inference_batch_size": args.inference_batch_size, "seed": args.seed,
        },
        "system": {"device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "platform": platform.platform()},
        "results": results,
    }
    (output_dir / "benchmark_results.json").write_text(json.dumps(report, indent=2))
    fields = ["model", "accuracy", "macro_f1", "training_sec", "peak_gpu_mem_gb",
              "p50_ms", "p95_ms", "p99_ms", "throughput_nodes_per_sec", "epochs_ran"]
    with (output_dir / "benchmark_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Benchmark complete: {output_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
