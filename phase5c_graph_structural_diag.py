import argparse
import json
from collections import defaultdict
from math import log
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Graph structural diagnostics for MGNN research")
    p.add_argument("--input-dir", default="artifacts/phase4_pyg")
    p.add_argument("--output", default="artifacts/metrics/phase5c_graph_structural.json")
    p.add_argument("--max-edges-for-homophily", type=int, default=2000000)
    return p.parse_args()


def union_find_components(num_nodes: int, src: np.ndarray, dst: np.ndarray) -> int:
    parent = np.arange(num_nodes, dtype=np.int64)
    rank = np.zeros(num_nodes, dtype=np.int8)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in zip(src, dst):
        ru = find(int(u))
        rv = find(int(v))
        if ru == rv:
            continue
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1

    roots = set()
    for i in range(num_nodes):
        roots.add(find(i))
    return len(roots)


def entropy_from_counts(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * log(p + 1e-12, 2)
    return float(ent)


def compute_homophily(y: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    if len(src) == 0:
        return 0.0
    return float(np.mean(y[src] == y[dst]))


def compute_degree_stats(num_nodes: int, src: np.ndarray):
    deg = np.bincount(src, minlength=num_nodes).astype(np.float64)
    mean_deg = float(deg.mean())
    std_deg = float(deg.std())
    skew_proxy = float(std_deg / (mean_deg + 1e-9))
    return {
        "degree_mean": mean_deg,
        "degree_std": std_deg,
        "degree_skew_proxy": skew_proxy,
        "degree_p95": float(np.percentile(deg, 95)),
        "degree_p99": float(np.percentile(deg, 99)),
    }


def compute_label_assortativity(y: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    if len(src) == 0:
        return 0.0
    edge_pairs = defaultdict(int)
    out_counts = defaultdict(int)
    in_counts = defaultdict(int)
    m = len(src)
    for u, v in zip(src, dst):
        cu = int(y[u])
        cv = int(y[v])
        edge_pairs[(cu, cv)] += 1
        out_counts[cu] += 1
        in_counts[cv] += 1
    labels = sorted(set(list(out_counts.keys()) + list(in_counts.keys())))
    tr_e = 0.0
    a2 = 0.0
    b2 = 0.0
    for c in labels:
        e_cc = edge_pairs.get((c, c), 0) / m
        tr_e += e_cc
        a = out_counts.get(c, 0) / m
        b = in_counts.get(c, 0) / m
        a2 += a * a
        b2 += b * b
    denom = 1.0 - ((a2 + b2) / 2.0)
    if abs(denom) < 1e-12:
        return 0.0
    return float((tr_e - ((a2 + b2) / 2.0)) / denom)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    y = torch.load(input_dir / "y.pt", map_location="cpu", weights_only=False).numpy()
    edge_index = torch.load(input_dir / "edge_index.pt", map_location="cpu", weights_only=False).numpy()
    edge_type = torch.load(input_dir / "edge_type.pt", map_location="cpu", weights_only=False).numpy()

    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    num_nodes = int(len(y))
    num_edges = int(len(src))

    degree_stats = compute_degree_stats(num_nodes, src)
    components = union_find_components(num_nodes, src, dst)

    rel_vals, rel_counts = np.unique(edge_type, return_counts=True)
    relation_dist = {int(k): int(v) for k, v in zip(rel_vals, rel_counts)}
    relation_entropy = entropy_from_counts(rel_counts.tolist())

    if num_edges > args.max_edges_for_homophily:
        idx = np.random.default_rng(42).choice(num_edges, args.max_edges_for_homophily, replace=False)
        hs = compute_homophily(y, src[idx], dst[idx])
        assort = compute_label_assortativity(y, src[idx], dst[idx])
    else:
        hs = compute_homophily(y, src, dst)
        assort = compute_label_assortativity(y, src, dst)

    result = {
        "nodes": num_nodes,
        "edges": num_edges,
        "connected_components": int(components),
        "homophily": float(hs),
        "label_assortativity": float(assort),
        "relation_entropy": float(relation_entropy),
        "relation_distribution": relation_dist,
        **degree_stats,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
