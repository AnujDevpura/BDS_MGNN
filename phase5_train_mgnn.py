# phase5_train_mgnn.py

# =========================================================
# MGNN PHASE 5
# FINAL MULTICLASS HETEROGENEOUS RGCN TRAINING
# =========================================================

import gc
import json
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

# =========================================================
# OUTPUT DIR
# =========================================================

Path(MODEL_OUTPUT).mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# LOAD TENSORS
# =========================================================

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

print("\nNormalizing node features...")

x = x.float()

feature_mean = x.mean(dim=0)

feature_std = x.std(dim=0)

# prevent divide-by-zero
feature_std[feature_std == 0] = 1.0

x = (x - feature_mean) / feature_std

print("Feature normalization complete")

print("Feature mean (global):",
      round(x.mean().item(), 6))

print("Feature std (global):",
      round(x.std().item(), 6))

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

# -----------------------------
# TRAIN / TEMP
# -----------------------------

train_idx, temp_idx = train_test_split(

    indices,

    test_size=0.2,

    stratify=labels,

    random_state=42

)

# -----------------------------
# VAL / TEST
# -----------------------------

temp_labels = labels[temp_idx]

val_idx, test_idx = train_test_split(

    temp_idx,

    test_size=0.5,

    stratify=temp_labels,

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

print("Train nodes:",
      train_mask.sum().item())

print("Val nodes:",
      val_mask.sum().item())

print("Test nodes:",
      test_mask.sum().item())

# =========================================================
# CLASS WEIGHTS
# =========================================================

print("\nComputing class weights...")

num_classes = int(y.max().item()) + 1

class_counts = torch.bincount(
    y,
    minlength=num_classes
).float()

print("Class counts:")
print(class_counts)

# -----------------------------
# SQRT INVERSE FREQUENCY
# -----------------------------

class_weights = torch.sqrt(

    class_counts.sum()

    /

    (class_counts * num_classes)

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

print("\nStarting MGNN training...")

best_val_f1 = 0

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