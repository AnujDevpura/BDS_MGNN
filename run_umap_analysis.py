# run_umap_analysis.py
# UMAP 2D Embedding Visualization of MGNN Learned Node Representations

import argparse, json, time
from pathlib import Path
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--model-dir",  default="artifacts/phase5_model")
p.add_argument("--data-dir",   default="artifacts/phase4_pyg")
p.add_argument("--output",     default="artifacts/research/umap_embeddings.json")
p.add_argument("--n-samples",  type=int, default=20000,
               help="Max nodes to include in UMAP (for speed)")
p.add_argument("--n-neighbors", type=int, default=15)
p.add_argument("--min-dist",   type=float, default=0.1)
args = p.parse_args()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0:"BENIGN", 1:"FTP-Patator", 2:"SSH-Patator", 3:"DoS Slowloris",
    4:"DoS Slowhttptest", 5:"DoS Hulk", 6:"DoS GoldenEye", 7:"Heartbleed",
    8:"Web Attack-Brute Force", 9:"Web Attack-XSS", 10:"Web Attack-SQL Injection",
    11:"Infiltration", 12:"Bot", 13:"PortScan", 14:"DDoS",
}

# ─── Load embeddings ─────────────────────────────────────────────────────────
embed_path = Path(args.model_dir) / "node_embeddings.pt"
if not embed_path.exists():
    print("ERROR: node_embeddings.pt not found.")
    print("Re-train with --save-embeddings flag:")
    print("  .venv\\Scripts\\python.exe phase5_train_mgnn.py --model-version v2 --save-embeddings")
    raise SystemExit(1)

print("Loading node embeddings...")
embeddings = torch.load(embed_path, map_location="cpu", weights_only=False).numpy()
y = torch.load(f"{args.data_dir}/y.pt", map_location="cpu", weights_only=False).numpy()

print(f"Embeddings shape: {embeddings.shape}")
print(f"Labels: {np.unique(y)}")

# Stratified subsample for UMAP speed
rng = np.random.default_rng(42)
n_total = len(y)
n_samples = min(args.n_samples, n_total)

unique_classes = np.unique(y)
per_class = max(1, n_samples // len(unique_classes))
sample_idx = []
for c in unique_classes:
    cls_idx = np.where(y == c)[0]
    take = min(per_class, len(cls_idx))
    sample_idx.extend(rng.choice(cls_idx, take, replace=False).tolist())
sample_idx = np.array(sample_idx[:n_samples])

X_sub = embeddings[sample_idx]
y_sub = y[sample_idx]
print(f"Subsampled: {len(sample_idx):,} nodes across {len(unique_classes)} classes")

# ─── UMAP ────────────────────────────────────────────────────────────────────
try:
    import umap
except ImportError:
    print("Installing umap-learn...")
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "umap-learn"], check=True)
    import umap

print(f"\nRunning UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist})...")
t0 = time.time()
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=args.n_neighbors,
    min_dist=args.min_dist,
    metric="euclidean",
    random_state=42,
    verbose=False,
)
coords_2d = reducer.fit_transform(X_sub)
print(f"UMAP done in {time.time()-t0:.1f}s")

# ─── Save ────────────────────────────────────────────────────────────────────
output = []
for i, (coord, label) in enumerate(zip(coords_2d, y_sub)):
    output.append({
        "x": float(coord[0]),
        "y": float(coord[1]),
        "class_id": int(label),
        "class_name": CLASS_NAMES.get(int(label), f"Class {label}"),
    })

Path(args.output).write_text(json.dumps(output, indent=1))
print(f"\n[DONE] UMAP embeddings saved: {args.output} ({len(output):,} points)")
