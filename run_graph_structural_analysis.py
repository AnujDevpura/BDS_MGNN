# run_graph_structural_analysis.py
# Graph Structural Analysis: Degree Distribution, Class Homophily, Connectivity

import argparse, json
from pathlib import Path
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="artifacts/phase4_pyg")
p.add_argument("--model-dir", default="artifacts/phase5_model")
p.add_argument("--output", default="artifacts/research/graph_structural.json")
args = p.parse_args()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0:"BENIGN", 1:"FTP-Patator", 2:"SSH-Patator", 3:"DoS Slowloris",
    4:"DoS Slowhttptest", 5:"DoS Hulk", 6:"DoS GoldenEye", 7:"Heartbleed",
    8:"Web Attack-Brute Force", 9:"Web Attack-XSS", 10:"Web Attack-SQL Injection",
    11:"Infiltration", 12:"Bot", 13:"PortScan", 14:"DDoS",
}

print("Loading graph tensors...")
y          = torch.load(f"{args.data_dir}/y.pt",          map_location="cpu", weights_only=False).long()
edge_index = torch.load(f"{args.data_dir}/edge_index.pt", map_location="cpu", weights_only=False).long()
edge_type  = torch.load(f"{args.data_dir}/edge_type.pt",  map_location="cpu", weights_only=False).long()

N = y.shape[0]
E = edge_index.shape[1]
num_classes = int(y.max().item()) + 1
src, dst = edge_index

print(f"Nodes: {N:,}  Edges: {E:,}  Classes: {num_classes}")

# ─── Degree distribution ─────────────────────────────────────────────────────
results = {"graph": {"nodes": int(N), "edges": int(E), "num_classes": num_classes}}

for r in range(2):
    mask = (edge_type == r)
    rel_src = src[mask]
    rel_dst = dst[mask]
    out_deg = torch.zeros(N, dtype=torch.long)
    in_deg  = torch.zeros(N, dtype=torch.long)
    out_deg.index_add_(0, rel_src, torch.ones_like(rel_src))
    in_deg.index_add_(0, rel_dst,  torch.ones_like(rel_dst))

    total_deg = out_deg + in_deg
    results[f"relation_{r}"] = {
        "name": "Temporal" if r == 0 else "Similarity",
        "edge_count": int(mask.sum()),
        "avg_out_degree": float(out_deg.float().mean()),
        "avg_in_degree":  float(in_deg.float().mean()),
        "max_out_degree": int(out_deg.max()),
        "max_in_degree":  int(in_deg.max()),
        "isolated_nodes": int((total_deg == 0).sum()),
        # Degree histogram (log-spaced bins)
        "degree_histogram": {
            "bins": [0, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 10000],
            "counts_out": [int((out_deg == i).sum()) if i < 5
                           else int(((out_deg >= blo) & (out_deg < bhi)).sum())
                           for i, (blo, bhi) in enumerate(
                               zip([0,1,2,5,10,20,50,100,500,1000],
                                   [1,2,5,10,20,50,100,500,1000,10001]))]
        }
    }

# ─── Class Homophily ─────────────────────────────────────────────────────────
# Fraction of edges where src and dst belong to the same class
# High homophily -> graph edges semantically meaningful

y_np = y.numpy()
homophily_all = float((y_np[src.numpy()] == y_np[dst.numpy()]).mean())

homophily_by_relation = {}
for r in range(2):
    mask = (edge_type == r).numpy()
    src_r, dst_r = src.numpy()[mask], dst.numpy()[mask]
    h = float((y_np[src_r] == y_np[dst_r]).mean()) if len(src_r) > 0 else 0.0
    homophily_by_relation[f"relation_{r}"] = {
        "name": "Temporal" if r == 0 else "Similarity",
        "homophily": h,
        "edges": int(mask.sum()),
    }

results["homophily"] = {
    "overall": homophily_all,
    "by_relation": homophily_by_relation,
    "interpretation": (
        "Homophily = fraction of edges connecting same-class nodes. "
        ">0.5 proves edges carry semantic signal meaningful for classification."
    )
}

# ─── Per-class degree statistics ─────────────────────────────────────────────
all_deg = torch.zeros(N, dtype=torch.long)
all_deg.index_add_(0, src, torch.ones(E, dtype=torch.long))
all_deg.index_add_(0, dst, torch.ones(E, dtype=torch.long))

per_class_stats = {}
for c in range(num_classes):
    mask_c = (y_np == c)
    deg_c  = all_deg.numpy()[mask_c]
    if len(deg_c) > 0:
        per_class_stats[str(c)] = {
            "class_name": CLASS_NAMES.get(c, f"Class {c}"),
            "node_count": int(mask_c.sum()),
            "avg_degree": float(deg_c.mean()),
            "max_degree": int(deg_c.max()),
            "median_degree": float(np.median(deg_c)),
        }

results["per_class_degree"] = per_class_stats

# ─── Save ─────────────────────────────────────────────────────────────────────
Path(args.output).write_text(json.dumps(results, indent=2))
print(f"\n[DONE] Graph structural analysis saved: {args.output}")
print(f"\nOverall homophily: {homophily_all:.4f}")
for r, info in homophily_by_relation.items():
    print(f"  {info['name']:15s} homophily: {info['homophily']:.4f}  ({info['edges']:,} edges)")
