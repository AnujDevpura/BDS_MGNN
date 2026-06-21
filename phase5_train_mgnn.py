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

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from sklearn.model_selection import train_test_split

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import RGCNConv

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
    return p.parse_args()

# =========================================================
# LOAD TENSORS
# =========================================================

args = parse_args()
INPUT_DIR = args.input_dir
MODEL_OUTPUT = args.output_dir
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

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

class_weights = torch.sqrt(

    class_counts.sum()

    /

    (safe_class_counts * num_classes)

)

class_weights = class_weights.to(DEVICE)

print("\nClass weights:")
print(class_weights)

# =========================================================
# NEIGHBOR LOADERS
# =========================================================

print("\nCreating NeighborLoaders...")

train_loader = NeighborLoader(
    data,
    input_nodes=data.train_mask,
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
# MGNN MODEL
# =========================================================

class MGNN(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_relations
    ):
        super().__init__()

        self.conv1 = RGCNConv(
            in_channels,
            hidden_channels,
            num_relations=num_relations
        )

        self.conv2 = RGCNConv(
            hidden_channels,
            hidden_channels,
            num_relations=num_relations
        )

        self.lin = torch.nn.Linear(
            hidden_channels,
            out_channels
        )

    def forward(
        self,
        x,
        edge_index,
        edge_type
    ):

        x = self.conv1(
            x,
            edge_index,
            edge_type
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training
        )

        x = self.conv2(
            x,
            edge_index,
            edge_type
        )

        x = F.relu(x)

        x = self.lin(x)

        return x

# =========================================================
# MODEL INIT
# =========================================================

num_features = data.num_features

model = MGNN(
    in_channels=num_features,
    hidden_channels=HIDDEN_DIM,
    out_channels=num_classes,
    num_relations=NUM_RELATIONS
)

model = model.to(DEVICE)

print("\nModel initialized")
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

# =========================================================
# TRAIN FUNCTION
# =========================================================

def train():

    model.train()

    total_loss = 0

    for step, batch in enumerate(train_loader):

        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        out = model(
            batch.x,
            batch.edge_index,
            batch.edge_type
        )

        loss = F.cross_entropy(

            out[:batch.batch_size],

            batch.y[:batch.batch_size],

            weight=class_weights

        )

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

    return total_loss / len(train_loader)

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
            batch.edge_type
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

for epoch in range(1, NUM_EPOCHS + 1):

    loss = train()

    val_acc, val_f1, _, _ = evaluate(
        val_loader
    )

    history.append({

        "epoch": epoch,

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

summary = {
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
