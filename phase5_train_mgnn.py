# phase5_train_mgnn.py

# =========================================================
# MGNN PHASE 5
# FINAL MULTICLASS HETEROGENEOUS RGCN TRAINING
# =========================================================

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from sklearn.model_selection import train_test_split

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# =========================================================
# CONFIG
# =========================================================

INPUT_DIR = "artifacts/phase4_pyg"
MODEL_OUTPUT = "artifacts/phase5_model"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# -----------------------------
# TRAINING
# -----------------------------

BATCH_SIZE = 1024

NUM_NEIGHBORS = [20, 15]

HIDDEN_DIM = 64

NUM_EPOCHS = 10

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 1e-5

DROPOUT = 0.2

GRAD_CLIP = 1.0

NUM_RELATIONS = 2

PRINT_EVERY = 20

EARLY_STOPPING_PATIENCE = 2

def parse_args():
    p = argparse.ArgumentParser(description="MGNN Phase 5 training")
    p.add_argument("--input-dir", default=INPUT_DIR)
    p.add_argument("--output-dir", default=MODEL_OUTPUT)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--metrics-output", default="")
    p.add_argument("--init-model", default="")
    p.add_argument("--num-classes", type=int, default=0)
    p.add_argument("--normalization-input", default="")
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    p.add_argument("--neighbors", default="20,15")
    p.add_argument("--sampling-alpha", type=float, default=0.0)
    p.add_argument("--focal-gamma", type=float, default=0.0)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--enhanced-model", action="store_true")
    p.add_argument("--use-edge-weight", action="store_true")
    p.add_argument("--finetune-epochs", type=int, default=0)
    p.add_argument("--finetune-learning-rate", type=float, default=0.00015)
    p.add_argument("--finetune-sampling-alpha", type=float, default=0.2)
    p.add_argument("--finetune-focal-gamma", type=float, default=1.0)
    p.add_argument("--finetune-class-weight-power", type=float, default=0.6)
    # ── v2 flags ──────────────────────────────────────────────────────────────
    p.add_argument("--model-version", choices=["v1", "v2"], default="v1",
                   help="v2: 3-layer RGCN, hidden=128, relation gating, focal γ=2.0")
    p.add_argument("--save-embeddings", action="store_true",
                   help="Save final GNN hidden states for UMAP / distillation")
    return p.parse_args()

# =========================================================
# LOAD TENSORS
# =========================================================

args = parse_args()
INPUT_DIR = args.input_dir
MODEL_OUTPUT = args.output_dir
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
HIDDEN_DIM = args.hidden_dim
DROPOUT = args.dropout
LEARNING_RATE = args.learning_rate
NUM_NEIGHBORS = [int(value) for value in args.neighbors.split(",")]

Path(MODEL_OUTPUT).mkdir(parents=True, exist_ok=True)

print("Loading PyG tensors...")

x = torch.load(
    f"{INPUT_DIR}/x.pt",
    map_location="cpu",
    weights_only=False
)

y = torch.load(
    f"{INPUT_DIR}/y.pt",
    map_location="cpu",
    weights_only=False
)

edge_index = torch.load(
    f"{INPUT_DIR}/edge_index.pt",
    map_location="cpu",
    weights_only=False
)

edge_weight = torch.load(
    f"{INPUT_DIR}/edge_weight.pt",
    map_location="cpu",
    weights_only=False
)

edge_type = torch.load(
    f"{INPUT_DIR}/edge_type.pt",
    map_location="cpu",
    weights_only=False
)

print("Tensors loaded")

# =========================================================
# FEATURE NORMALIZATION
# =========================================================

x = x.float()

# =========================================================
# BUILD GRAPH
# =========================================================

print("\nBuilding PyG graph...")

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_weight,
    edge_type=edge_type,
    y=y
)

print(data)

# =========================================================
# STRATIFIED SPLITS
# =========================================================

print("\nCreating stratified train/val/test splits...")

num_nodes = data.num_nodes

indices = np.arange(num_nodes)

labels = y.numpy()

label_values, label_frequencies = np.unique(labels, return_counts=True)
can_stratify = len(label_values) > 1 and int(label_frequencies.min()) >= 10

# -----------------------------
# TRAIN / TEMP
# -----------------------------

train_idx, temp_idx = train_test_split(

    indices,

    test_size=0.2,

    stratify=labels if can_stratify else None,

    random_state=42

)

# -----------------------------
# VAL / TEST
# -----------------------------

temp_labels = labels[temp_idx]
temp_values, temp_frequencies = np.unique(temp_labels, return_counts=True)
can_stratify_temp = len(temp_values) > 1 and int(temp_frequencies.min()) >= 2

val_idx, test_idx = train_test_split(

    temp_idx,

    test_size=0.5,

    stratify=temp_labels if can_stratify_temp else None,

    random_state=42

)

# -----------------------------
# CONVERT TO TENSORS
# -----------------------------

train_idx = torch.tensor(
    train_idx,
    dtype=torch.long
)

val_idx = torch.tensor(
    val_idx,
    dtype=torch.long
)

test_idx = torch.tensor(
    test_idx,
    dtype=torch.long
)

# -----------------------------
# MASKS
# -----------------------------

train_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)

val_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)

test_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

torch.save(
    {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx},
    Path(MODEL_OUTPUT) / "split_indices.pt",
)

print("Train nodes:",
      train_mask.sum().item())

print("Val nodes:",
      val_mask.sum().item())

print("Test nodes:",
      test_mask.sum().item())

# Fit preprocessing only on training nodes to avoid validation/test leakage.
print("\nNormalizing node features from training statistics...")
if args.normalization_input:
    normalization = torch.load(args.normalization_input, map_location="cpu", weights_only=False)
    feature_mean = normalization["mean"]
    feature_std = normalization["std"]
else:
    feature_mean = x[train_idx].mean(dim=0)
    feature_std = x[train_idx].std(dim=0)
    feature_std[feature_std == 0] = 1.0
x = (x - feature_mean) / feature_std
data.x = x
print("Feature normalization complete")

# =========================================================
# CLASS WEIGHTS
# =========================================================

print("\nComputing class weights...")

num_classes = args.num_classes if args.num_classes > 0 else int(y.max().item()) + 1
if int(y.max().item()) >= num_classes:
    raise ValueError("--num-classes must be greater than the maximum label ID")

class_counts = torch.bincount(
    y[train_idx],
    minlength=num_classes
).float()

print("Class counts:")
print(class_counts)

# -----------------------------
# SQRT INVERSE FREQUENCY
# -----------------------------

safe_class_counts = class_counts.clamp_min(1.0)

def build_class_weights(power):
    weights = (class_counts.sum() / (safe_class_counts * num_classes)).pow(power)
    weights = weights / weights.mean()
    return weights.to(DEVICE)

class_weights = build_class_weights(args.class_weight_power)
finetune_class_weights = build_class_weights(args.finetune_class_weight_power)

print("\nClass weights:")
print(class_weights)
if args.finetune_epochs > 0:
    print("Finetune class weights:")
    print(finetune_class_weights)

# =========================================================
# NEIGHBOR LOADERS
# =========================================================

print("\nCreating NeighborLoaders...")

def build_train_input_nodes(alpha):
    train_input_nodes = train_idx
    if alpha > 0:
        sampling_weights = safe_class_counts[y[train_idx]].pow(-alpha)
        generator = torch.Generator().manual_seed(42)
        sampled_positions = torch.multinomial(
            sampling_weights,
            num_samples=len(train_idx),
            replacement=True,
            generator=generator,
        )
        train_input_nodes = train_idx[sampled_positions]
        print(f"Balanced seed sampling enabled: alpha={alpha}")
    return train_input_nodes

train_input_nodes = build_train_input_nodes(args.sampling_alpha)
finetune_input_nodes = build_train_input_nodes(args.finetune_sampling_alpha)

train_loader = NeighborLoader(
    data,
    input_nodes=train_input_nodes,
    num_neighbors=NUM_NEIGHBORS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

finetune_loader = NeighborLoader(
    data,
    input_nodes=finetune_input_nodes,
    num_neighbors=NUM_NEIGHBORS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = NeighborLoader(
    data,
    input_nodes=data.val_mask,
    num_neighbors=NUM_NEIGHBORS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = NeighborLoader(
    data,
    input_nodes=data.test_mask,
    num_neighbors=NUM_NEIGHBORS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("NeighborLoaders ready")

# =========================================================
# MGNN MODEL v1 (backward-compatible)
# =========================================================

class MGNN(torch.nn.Module):
    """MGNN v1 — 2-layer Hybrid Residual RGCN (original architecture)."""

    class WeightedRGCNConv(torch.nn.Module):
        def __init__(self, in_channels, out_channels, num_relations):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.empty(num_relations, in_channels, out_channels)
            )
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
                edge_weight = torch.ones(
                    edge_index.size(1), device=x.device, dtype=x.dtype)
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

    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        conv_cls = self.WeightedRGCNConv if args.use_edge_weight else pyg_nn.RGCNConv
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
        self.norm1 = torch.nn.LayerNorm(hidden_channels) if args.enhanced_model else None
        self.norm2 = torch.nn.LayerNorm(hidden_channels) if args.enhanced_model else None

    def forward(self, x, edge_index, edge_type, edge_weight=None, return_embeddings=False):
        out_bypass = self.bypass(x)
        h = self.feature_proj(x)
        h = self.conv1(h, edge_index, edge_type) if edge_weight is None else \
            self.conv1(h, edge_index, edge_type, edge_weight)
        h = self.norm1(h) if self.norm1 is not None else h
        h = F.relu(h)
        h = F.dropout(h, p=DROPOUT, training=self.training)
        residual = h
        h = self.conv2(h, edge_index, edge_type) if edge_weight is None else \
            self.conv2(h, edge_index, edge_type, edge_weight)
        h = self.norm2(h) if self.norm2 is not None else h
        h = F.relu(h + residual) if self.norm2 is not None else F.relu(h)
        if return_embeddings:
            return self.lin(h) + out_bypass, h
        return self.lin(h) + out_bypass


# =========================================================
# MGNN MODEL v2 — Upgraded Architecture
# =========================================================

class MGNNv2(torch.nn.Module):
    """
    MGNN v2 — Hybrid Residual Multi-Relational GNN (upgraded).

    Improvements over v1:
    - 3 RGCN layers (3-hop message passing) with residuals at every layer
    - Hidden dim 128 (vs 64) — richer node representations
    - Relation-Gating Attention: learned soft gate over Relation-0 (temporal)
      and Relation-1 (similarity), per node at each layer
    - LayerNorm always-on after every convolution
    - Bypass (tabular MLP) path unchanged — safety net guarantee
    - return_embeddings flag for UMAP / SIGN distillation
    """

    class _RelGatedRGCNLayer(torch.nn.Module):
        """
        One layer of relation-gated message passing.
        For each relation, aggregates neighbours separately, then combines
        with a per-node soft attention gate learned via a small linear head.
        """
        def __init__(self, in_channels, out_channels, num_relations, dropout):
            super().__init__()
            self.num_relations = num_relations
            self.dropout = dropout
            # Separate GNN per relation (allows different aggregation per edge type)
            self.convs = torch.nn.ModuleList([
                pyg_nn.SAGEConv(in_channels, out_channels, aggr="mean")
                for _ in range(num_relations)
            ])
            # Per-node attention gate: maps hidden -> num_relations scores
            self.gate = torch.nn.Linear(in_channels, num_relations, bias=True)
            self.norm = torch.nn.LayerNorm(out_channels)

        def forward(self, h, edge_index, edge_type):
            # Compute per-relation soft gate weights for each node
            gate_weights = torch.softmax(self.gate(h), dim=-1)  # [N, R]

            # Aggregate each relation separately
            rel_outs = []
            for r in range(self.num_relations):
                mask = (edge_type == r)
                if mask.any():
                    ei_r = edge_index[:, mask]
                    rel_outs.append(self.convs[r](h, ei_r))  # [N, out]
                else:
                    rel_outs.append(torch.zeros(
                        h.size(0), self.convs[r].out_channels,
                        device=h.device, dtype=h.dtype
                    ))

            # Gated combination: sum over relations weighted by gate
            out = sum(gate_weights[:, r:r+1] * rel_outs[r]
                      for r in range(self.num_relations))
            out = self.norm(out)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            return out

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_relations, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        # Feature projection (maps raw features -> hidden space)
        self.feature_proj = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )

        # Bypass MLP (tabular path — safety net guarantee)
        self.bypass = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, out_channels)
        )

        # 3 relation-gated RGCN layers
        self.layer1 = self._RelGatedRGCNLayer(
            hidden_channels, hidden_channels, num_relations, dropout)
        self.layer2 = self._RelGatedRGCNLayer(
            hidden_channels, hidden_channels, num_relations, dropout)
        self.layer3 = self._RelGatedRGCNLayer(
            hidden_channels, hidden_channels, num_relations, dropout)

        # Classification head
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type, edge_weight=None,
                return_embeddings=False):
        # Bypass (tabular) path
        out_bypass = self.bypass(x)

        # GNN path
        h = self.feature_proj(x)

        # Layer 1
        h1 = self.layer1(h, edge_index, edge_type)
        h = h + h1                      # residual

        # Layer 2
        h2 = self.layer2(h, edge_index, edge_type)
        h = h + h2                      # residual

        # Layer 3
        h3 = self.layer3(h, edge_index, edge_type)
        h = h + h3                      # residual

        out_gnn = self.lin(h)

        if return_embeddings:
            return out_gnn + out_bypass, h
        return out_gnn + out_bypass

# =========================================================
# MODEL INIT
# =========================================================

num_features = data.num_features

# Apply v2 defaults when --model-version v2 is requested
if args.model_version == "v2":
    if args.hidden_dim == HIDDEN_DIM:   # user didn't override -> use v2 default
        HIDDEN_DIM = 128
    if args.focal_gamma == 0.0:         # user didn't override -> enable focal loss
        args.focal_gamma = 2.0
    if args.finetune_focal_gamma == 1.0:
        args.finetune_focal_gamma = 2.0
    if args.sampling_alpha == 0.0:
        args.sampling_alpha = 0.3       # mild minority oversampling
    if args.finetune_epochs == 0:
        args.finetune_epochs = 5        # automatic finetune pass
    print("[MGNNv2] Activated: hidden=128, focal_gamma=2.0, sampling_alpha=0.3, finetune_epochs=5")

if args.model_version == "v2":
    model = MGNNv2(
        in_channels=num_features,
        hidden_channels=HIDDEN_DIM,
        out_channels=num_classes,
        num_relations=NUM_RELATIONS,
        dropout=DROPOUT
    )
else:
    model = MGNN(
        in_channels=num_features,
        hidden_channels=HIDDEN_DIM,
        out_channels=num_classes,
        num_relations=NUM_RELATIONS
    )

model = model.to(DEVICE)

print(f"\nModel v{args.model_version} initialized (hidden={HIDDEN_DIM})")
print(model)

if args.init_model:
    print(f"\nLoading init model weights from: {args.init_model}")
    state = torch.load(args.init_model, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state, strict=False)

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


def classification_loss(logits, targets, weights, focal_gamma):
    if focal_gamma <= 0:
        return F.cross_entropy(logits, targets, weight=weights)
    log_probs = F.log_softmax(logits, dim=1)
    target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    target_probs = target_log_probs.exp().clamp(min=1e-8, max=1.0 - 1e-8)
    focal_factor = (1.0 - target_probs).clamp(min=1e-8, max=1.0).pow(focal_gamma)
    return (-weights[targets] * focal_factor * target_log_probs).mean()

# =========================================================
# TRAIN FUNCTION
# =========================================================

def train_epoch(loader, weights, focal_gamma):

    model.train()

    total_loss = 0

    for step, batch in enumerate(loader):

        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        out = model(
            batch.x,
            batch.edge_index,
            batch.edge_type,
            batch.edge_attr if args.use_edge_weight else None,
        )

        loss = classification_loss(
            out[:batch.batch_size],
            batch.y[:batch.batch_size],
            weights,
            focal_gamma,
        )

        if not torch.isfinite(loss):
            print(f"Step {step} | Non-finite loss encountered, skipping batch")
            del batch
            torch.cuda.empty_cache()
            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        total_loss += loss.item()

        if step % PRINT_EVERY == 0:

            print(
                f"Step {step} | "
                f"Loss: {loss.item():.4f}"
            )

        del batch

        torch.cuda.empty_cache()

    return total_loss / max(len(loader), 1)

# =========================================================
# EVALUATION
# =========================================================

@torch.no_grad()
def evaluate(loader):

    model.eval()

    all_preds = []
    all_labels = []

    for batch in loader:

        batch = batch.to(DEVICE)

        out = model(
            batch.x,
            batch.edge_index,
            batch.edge_type,
            batch.edge_attr if args.use_edge_weight else None,
        )

        preds = (
            out[:batch.batch_size]
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        labels = (
            batch.y[:batch.batch_size]
            .cpu()
            .numpy()
        )

        all_preds.extend(preds)
        all_labels.extend(labels)

        del batch

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    return (
        acc,
        macro_f1,
        np.array(all_labels),
        np.array(all_preds)
    )

# =========================================================
# TRAIN LOOP
# =========================================================

run_start = time.time()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

print("\nStarting MGNN training...")

best_val_f1 = -1.0

epochs_without_improvement = 0

history = []
phase_name = "base"

for epoch in range(1, NUM_EPOCHS + 1):

    loss = train_epoch(train_loader, class_weights, args.focal_gamma)

    val_acc, val_f1, _, _ = evaluate(
        val_loader
    )

    history.append({

        "epoch": epoch,
        "phase": phase_name,

        "loss": float(loss),

        "val_acc": float(val_acc),

        "val_macro_f1": float(val_f1)

    })

    print(
        f"\nEpoch {epoch:02d}"
        f" | Loss: {loss:.4f}"
        f" | Val Acc: {val_acc:.4f}"
        f" | Val Macro F1: {val_f1:.4f}"
    )

    # -----------------------------------------------------
    # SAVE BEST MODEL
    # -----------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            f"{MODEL_OUTPUT}/best_model.pt"
        )

        print("Best model saved")

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement for "
            f"{epochs_without_improvement} epoch(s)"
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print("\nEarly stopping triggered")

            break

    gc.collect()

    torch.cuda.empty_cache()

if args.finetune_epochs > 0:
    print("\nStarting minority-aware finetune...")
    model.load_state_dict(
        torch.load(
            f"{MODEL_OUTPUT}/best_model.pt",
            map_location=DEVICE,
            weights_only=False
        )
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.finetune_learning_rate,
        weight_decay=WEIGHT_DECAY
    )
    phase_name = "finetune"
    epochs_without_improvement = 0

    for finetune_epoch in range(1, args.finetune_epochs + 1):
        loss = train_epoch(
            finetune_loader,
            finetune_class_weights,
            args.finetune_focal_gamma,
        )

        val_acc, val_f1, _, _ = evaluate(val_loader)

        history.append({
            "epoch": NUM_EPOCHS + finetune_epoch,
            "phase": phase_name,
            "loss": float(loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_f1)
        })

        print(
            f"\nFinetune Epoch {finetune_epoch:02d}"
            f" | Loss: {loss:.4f}"
            f" | Val Acc: {val_acc:.4f}"
            f" | Val Macro F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(
                model.state_dict(),
                f"{MODEL_OUTPUT}/best_model.pt"
            )
            print("Best model saved")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for "
                f"{epochs_without_improvement} finetune epoch(s)"
            )
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("\nFinetune early stopping triggered")
                break

        gc.collect()
        torch.cuda.empty_cache()

# =========================================================
# FINAL TEST
# =========================================================

print("\nLoading best model...")

model.load_state_dict(

    torch.load(
        f"{MODEL_OUTPUT}/best_model.pt",
        map_location=DEVICE,
        weights_only=False
    )

)

test_acc, test_f1, test_labels, test_preds = evaluate(
    test_loader
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    test_labels,
    test_preds
)

np.save(
    f"{MODEL_OUTPUT}/confusion_matrix.npy",
    cm
)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nGenerating classification report...")

report = classification_report(
    test_labels,
    test_preds,
    zero_division=0,
    output_dict=True
)

with open(
    f"{MODEL_OUTPUT}/classification_report.json",
    "w"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )

# =========================================================
# SAVE TRAIN HISTORY
# =========================================================

with open(
    f"{MODEL_OUTPUT}/training_history.json",
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=2
    )

# =========================================================
# FINAL RESULTS
# =========================================================

print("\n==============================")
print("PHASE 5 COMPLETE")
print("==============================")

print("\nDevice:",
      DEVICE)

print("\nNodes:",
      data.num_nodes)

print("Edges:",
      data.num_edges)

print("\nFeatures:",
      data.num_features)

print("Classes:",
      num_classes)

print("\nBest Validation Macro F1:",
      round(best_val_f1, 4))

print("Test Accuracy:",
      round(test_acc, 4))

print("Test Macro F1:",
      round(test_f1, 4))

print("\nModel saved to:")
print(MODEL_OUTPUT)

print("\nSaved artifacts:")
print("- best_model.pt")
print("- confusion_matrix.npy")
print("- classification_report.json")
print("- training_history.json")

print("\nFinal MGNN training complete.")

total_runtime_sec = time.time() - run_start
peak_gpu_mem_gb = 0.0
if torch.cuda.is_available():
    peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

# =========================================================
# SAVE NODE EMBEDDINGS (for UMAP / SIGN distillation)
# =========================================================

if args.save_embeddings:
    print("\nSaving node embeddings (full graph forward pass)...")
    model.eval()
    all_embeds = []
    embed_loader = NeighborLoader(
        data,
        input_nodes=torch.arange(data.num_nodes),
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
    )
    with torch.no_grad():
        for batch in embed_loader:
            batch = batch.to(DEVICE)
            _, emb = model(
                batch.x, batch.edge_index, batch.edge_type,
                return_embeddings=True
            )
            all_embeds.append(emb[:batch.batch_size].cpu())
            del batch
    all_embeds = torch.cat(all_embeds, dim=0)  # [N, hidden_dim]
    embed_path = Path(MODEL_OUTPUT) / "node_embeddings.pt"
    torch.save(all_embeds, embed_path)
    print(f"Node embeddings saved: {embed_path} — shape {all_embeds.shape}")
    del all_embeds

summary = {
    "model_version": args.model_version,
    "nodes": int(data.num_nodes),
    "edges": int(data.num_edges),
    "features": int(data.num_features),
    "classes": int(num_classes),
    "best_val_macro_f1": float(best_val_f1),
    "test_accuracy": float(test_acc),
    "test_macro_f1": float(test_f1),
    "runtime_sec": float(total_runtime_sec),
    "peak_gpu_mem_gb": float(peak_gpu_mem_gb),
    "epochs_ran": int(len(history)),
    "hidden_dim": HIDDEN_DIM,
    "neighbors": NUM_NEIGHBORS,
    "sampling_alpha": args.sampling_alpha,
    "focal_gamma": args.focal_gamma,
    "class_weight_power": args.class_weight_power,
    "enhanced_model": args.enhanced_model,
    "use_edge_weight": args.use_edge_weight,
    "finetune_epochs": args.finetune_epochs,
    "finetune_learning_rate": args.finetune_learning_rate,
    "finetune_sampling_alpha": args.finetune_sampling_alpha,
    "finetune_focal_gamma": args.finetune_focal_gamma,
    "finetune_class_weight_power": args.finetune_class_weight_power,
    "save_embeddings": args.save_embeddings,
}

summary_path = Path(MODEL_OUTPUT) / "run_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
torch.save(
    {"mean": feature_mean.cpu(), "std": feature_std.cpu()},
    Path(MODEL_OUTPUT) / "feature_normalization.pt",
)
if args.metrics_output:
    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
