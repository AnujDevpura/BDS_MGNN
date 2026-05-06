# phase5_diagnostics.py

# =========================================================
# MGNN PHASE 5.5
# DIAGNOSTICS + EVALUATION VALIDATION
# =========================================================

import numpy as np
import pandas as pd
import torch

from collections import Counter

# =========================================================
# CONFIG
# =========================================================

INPUT_DIR = "artifacts/phase4_pyg"

TOP_DUPLICATES_TO_SHOW = 10

# =========================================================
# LOAD TENSORS
# =========================================================

print("Loading tensors...")

x = torch.load(
    f"{INPUT_DIR}/x.pt",
    map_location="cpu"
)

y = torch.load(
    f"{INPUT_DIR}/y.pt",
    map_location="cpu"
)

edge_index = torch.load(
    f"{INPUT_DIR}/edge_index.pt",
    map_location="cpu"
)

edge_type = torch.load(
    f"{INPUT_DIR}/edge_type.pt",
    map_location="cpu"
)

print("Tensors loaded")

# =========================================================
# BASIC STATS
# =========================================================

num_nodes = x.shape[0]
num_features = x.shape[1]
num_edges = edge_index.shape[1]

print("\n==============================")
print("GRAPH STATS")
print("==============================")

print("\nNodes:", num_nodes)
print("Features:", num_features)
print("Edges:", num_edges)

# =========================================================
# LABEL DISTRIBUTION
# =========================================================

print("\n==============================")
print("LABEL DISTRIBUTION")
print("==============================")

labels = y.numpy()

label_counts = Counter(labels)

for label, count in label_counts.items():

    percentage = 100 * count / len(labels)

    print(
        f"Class {label}: "
        f"{count} nodes "
        f"({percentage:.2f}%)"
    )

# =========================================================
# EDGE TYPE DISTRIBUTION
# =========================================================

print("\n==============================")
print("EDGE TYPE DISTRIBUTION")
print("==============================")

edge_types = edge_type.numpy()

edge_type_counts = Counter(edge_types)

for edge_t, count in edge_type_counts.items():

    edge_name = (
        "temporal"
        if edge_t == 0
        else "similarity"
    )

    percentage = 100 * count / len(edge_types)

    print(
        f"{edge_name}: "
        f"{count} edges "
        f"({percentage:.2f}%)"
    )

# =========================================================
# FEATURE STATISTICS
# =========================================================

print("\n==============================")
print("FEATURE STATISTICS")
print("==============================")

x_np = x.numpy()

print("\nFeature matrix shape:", x_np.shape)

print("\nGlobal stats:")

print("Min:", np.min(x_np))
print("Max:", np.max(x_np))
print("Mean:", np.mean(x_np))
print("Std:", np.std(x_np))

# =========================================================
# NAN / INF CHECK
# =========================================================

print("\n==============================")
print("NAN / INF CHECK")
print("==============================")

nan_count = np.isnan(x_np).sum()
inf_count = np.isinf(x_np).sum()

print("\nNaN values:", nan_count)
print("Inf values:", inf_count)

# =========================================================
# DUPLICATE FEATURE CHECK
# =========================================================

print("\n==============================")
print("DUPLICATE FEATURE CHECK")
print("==============================")

print("\nHashing feature vectors...")

feature_hashes = []

for row in x_np:

    feature_hashes.append(
        hash(row.tobytes())
    )

hash_counts = Counter(feature_hashes)

duplicate_groups = {
    h: c
    for h, c in hash_counts.items()
    if c > 1
}

num_duplicate_groups = len(duplicate_groups)

num_duplicate_nodes = sum(
    duplicate_groups.values()
)

print("\nDuplicate feature groups:",
      num_duplicate_groups)

print("Nodes involved in duplicates:",
      num_duplicate_nodes)

if num_duplicate_groups > 0:

    print("\nLargest duplicate groups:")

    largest = sorted(
        duplicate_groups.values(),
        reverse=True
    )[:TOP_DUPLICATES_TO_SHOW]

    for i, size in enumerate(largest):

        print(
            f"Group {i+1}: "
            f"{size} identical nodes"
        )

# =========================================================
# FEATURE VARIANCE CHECK
# =========================================================

print("\n==============================")
print("LOW VARIANCE FEATURES")
print("==============================")

feature_variances = np.var(
    x_np,
    axis=0
)

low_variance = np.where(
    feature_variances < 1e-8
)[0]

print("\nLow variance feature count:",
      len(low_variance))

if len(low_variance) > 0:

    print("Low variance feature indices:")

    print(low_variance.tolist())

# =========================================================
# FEATURE-LABEL CORRELATION
# =========================================================

print("\n==============================")
print("FEATURE-LABEL CORRELATION")
print("==============================")

correlations = []

y_np = y.numpy()

for i in range(num_features):

    feat = x_np[:, i]

    # avoid constant features
    if np.std(feat) < 1e-8:

        correlations.append(0)

        continue

    corr = np.corrcoef(
        feat,
        y_np
    )[0, 1]

    if np.isnan(corr):
        corr = 0

    correlations.append(abs(corr))

correlations = np.array(correlations)

top_corr_idx = np.argsort(
    correlations
)[::-1][:10]

print("\nTop correlated features:")

for idx in top_corr_idx:

    print(
        f"Feature {idx}: "
        f"{correlations[idx]:.4f}"
    )

# =========================================================
# GRAPH CONNECTIVITY CHECK
# =========================================================

print("\n==============================")
print("GRAPH CONNECTIVITY")
print("==============================")

src_nodes = edge_index[0].numpy()
dst_nodes = edge_index[1].numpy()

unique_graph_nodes = len(
    set(src_nodes).union(set(dst_nodes))
)

print("\nNodes appearing in graph:",
      unique_graph_nodes)

isolated_nodes = (
    num_nodes -
    unique_graph_nodes
)

print("Potential isolated nodes:",
      isolated_nodes)

# =========================================================
# DEGREE STATISTICS
# =========================================================

print("\n==============================")
print("DEGREE STATISTICS")
print("==============================")

degree_counter = Counter(src_nodes)

degrees = np.array(
    list(degree_counter.values())
)

print("\nMin degree:", degrees.min())
print("Max degree:", degrees.max())
print("Mean degree:", degrees.mean())
print("Median degree:", np.median(degrees))

# =========================================================
# SUMMARY
# =========================================================

print("\n==============================")
print("DIAGNOSTIC SUMMARY")
print("==============================")

print("\nPotential issues detected:")

issues = []

# imbalance
largest_class_pct = (
    max(label_counts.values())
    / num_nodes
)

if largest_class_pct > 0.95:

    issues.append(
        "Extreme class imbalance"
    )

# duplicates
if num_duplicate_nodes > 1000:

    issues.append(
        "Large number of duplicate features"
    )

# leakage
if correlations.max() > 0.95:

    issues.append(
        "Possible feature-label leakage"
    )

# isolated nodes
if isolated_nodes > 0:

    issues.append(
        "Graph contains isolated nodes"
    )

if len(issues) == 0:

    print("\nNo major issues detected.")

else:

    for issue in issues:

        print("-", issue)

print("\nDiagnostics complete.")