# phase5b_sign_precompute.py
#
# SIGN (Scalable Inception Graph Neural Networks) — Offline Precomputation
# =========================================================================
#
# Computes graph-diffused feature tensors:
#   X^(0) = X                   (raw features,        shape [N, F])
#   X^(1) = A_hat @ X           (1-hop aggregation,   shape [N, F])
#   X^(2) = A_hat^2 @ X         (2-hop aggregation,   shape [N, F])
#   X^(3) = A_hat^3 @ X         (3-hop aggregation,   shape [N, F])
#
# Concatenates to X_sign = [X^(0), X^(1), X^(2), X^(3)]  shape [N, 4*F]
# Saves to:   artifacts/phase4_pyg/x_sign.pt
#
# At inference time, a plain MLP on x_sign[batch_ids] replicates 3-hop
# GNN message passing — NO graph sampling overhead — enabling 300k+ nodes/sec.
#
# A_hat = D^{-1/2} A D^{-1/2}  (symmetric normalisation, self-loops added)
# =========================================================================

import argparse
import time
from pathlib import Path

import torch
import torch.sparse
import numpy as np

# ─── CLI ─────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="SIGN offline graph-diffusion precompute")
p.add_argument("--input-dir",  default="artifacts/phase4_pyg")
p.add_argument("--output-dir", default="artifacts/phase4_pyg")
p.add_argument("--hops",  type=int, default=3,
               help="Number of diffusion hops K (X, AX, A^2X, ..., A^K X)")
p.add_argument("--chunk-size", type=int, default=50_000,
               help="Nodes processed per chunk to stay in RAM")
args = p.parse_args()

INPUT_DIR  = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("SIGN Precomputation — Offline Graph Diffusion")
print("=" * 60)

# ─── Load tensors ─────────────────────────────────────────────────────────────
print("\nLoading graph tensors...")
t0 = time.time()

x = torch.load(INPUT_DIR / "x.pt", map_location="cpu", weights_only=False).float()
edge_index = torch.load(INPUT_DIR / "edge_index.pt", map_location="cpu", weights_only=False).long()
y = torch.load(INPUT_DIR / "y.pt", map_location="cpu", weights_only=False).long()

# Load normalization and apply (same as training)
norm_path = Path("artifacts/phase5_model/feature_normalization.pt")
if norm_path.exists():
    norm = torch.load(norm_path, map_location="cpu", weights_only=False)
    x = (x - norm["mean"]) / norm["std"]
    print("  Applied training-set normalization")

N, F = x.shape
print(f"  Nodes: {N:,}  |  Features: {F}  |  Edges: {edge_index.shape[1]:,}")
print(f"  Loaded in {time.time()-t0:.1f}s")

# ─── Build normalised adjacency matrix A_hat ─────────────────────────────────
print("\nBuilding normalised adjacency A_hat = D^{-1/2} A D^{-1/2} (with self-loops)...")
t_adj = time.time()

# Add self-loops: augment edge_index with [i, i] for all i
self_loop_idx = torch.arange(N, dtype=torch.long)
self_loops = torch.stack([self_loop_idx, self_loop_idx], dim=0)  # [2, N]
edge_index_aug = torch.cat([edge_index, self_loops], dim=1)       # [2, E+N]

# Count degree (including self-loop)
row, col = edge_index_aug
deg = torch.zeros(N, dtype=torch.float32)
deg.index_add_(0, row, torch.ones(edge_index_aug.shape[1], dtype=torch.float32))
# D^{-1/2}
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

# Edge weights: d_i^{-1/2} * d_j^{-1/2}
edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [E+N]

# Build sparse A_hat as a torch.sparse_coo_tensor
A_hat = torch.sparse_coo_tensor(
    indices=edge_index_aug,
    values=edge_weight,
    size=(N, N),
    dtype=torch.float32,
).coalesce()

print(f"  A_hat built in {time.time()-t_adj:.1f}s  |  nnz = {A_hat._nnz():,}")

# ─── Compute diffusion hops ───────────────────────────────────────────────────
print(f"\nComputing {args.hops}-hop diffusion [X, AX, A^2X, ..., A^{args.hops}X]...")
print("  (chunked sparse-dense multiply to stay in RAM)")

diffused = [x.clone()]  # X^(0) = X

current = x.clone()     # will be overwritten each hop
for hop in range(1, args.hops + 1):
    t_hop = time.time()
    # Chunked sparse@dense: A_hat @ current
    result = torch.zeros_like(current)
    chunk = args.chunk_size

    # Convert A_hat to CSR for efficient row-slice
    A_csr = A_hat.to_sparse_csr()
    crow  = A_csr.crow_indices()
    ccol  = A_csr.col_indices()
    cval  = A_csr.values()

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        # Gather all edges that have dst in [start, end)
        src_rows = crow[start:end + 1]  # pointer into ccol / cval
        for local_i, global_i in enumerate(range(start, end)):
            s = int(crow[global_i])
            e = int(crow[global_i + 1])
            if e > s:
                cols = ccol[s:e]
                vals = cval[s:e]
                result[global_i] = (vals.unsqueeze(1) * current[cols]).sum(0)

    diffused.append(result.clone())
    current = result
    print(f"  Hop {hop} done in {time.time()-t_hop:.1f}s")

# ─── Concatenate all hops ─────────────────────────────────────────────────────
print(f"\nConcatenating {len(diffused)} tensors -> X_sign shape [N, {F * len(diffused)}]...")
X_sign = torch.cat(diffused, dim=1)   # [N, F*(K+1)]
print(f"  X_sign: {X_sign.shape}  |  size on disk: {X_sign.nbytes / 1e6:.1f} MB")

# ─── Save ─────────────────────────────────────────────────────────────────────
out_path = OUTPUT_DIR / "x_sign.pt"
torch.save(X_sign, out_path)
print(f"\n[DONE] Saved: {out_path}")

# Save metadata for the student training script
import json
meta = {
    "hops": args.hops,
    "sign_features": int(X_sign.shape[1]),
    "original_features": int(F),
    "nodes": int(N),
    "normalized": norm_path.exists(),
}
(OUTPUT_DIR / "sign_meta.json").write_text(json.dumps(meta, indent=2))
print(f"   Metadata: {OUTPUT_DIR / 'sign_meta.json'}")
print(f"\nTotal runtime: {time.time()-t0:.1f}s")
print("=" * 60)
print("Next step:  python phase5c_sign_student.py")
print("=" * 60)
