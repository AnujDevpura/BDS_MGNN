# train_phase5_mgnn.py
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleMGNNFallback(nn.Module):
    # Fallback if you don't import your own MGNN class
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bin_head = nn.Linear(hidden_dim, 1)  # binary logits

        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight=edge_weight))
        x2 = x2 + x1  # residual
        logits = self.bin_head(x2).squeeze(-1)
        return logits


def make_masks(num_nodes: int, train_ratio=0.7, val_ratio=0.15, device="cpu"):
    perm = torch.randperm(num_nodes, device=device)
    tr_end = int(train_ratio * num_nodes)
    va_end = int((train_ratio + val_ratio) * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    train_mask[perm[:tr_end]] = True
    val_mask[perm[tr_end:va_end]] = True
    test_mask[perm[va_end:]] = True
    return train_mask, val_mask, test_mask


@torch.no_grad()
def eval_binary(model, data: Data, mask: torch.Tensor):
    model.eval()
    logits = model(
        data.x,
        data.edge_index,
        data.edge_weight if hasattr(data, "edge_weight") else None,
    )
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()

    y_true = data.y[mask].long().cpu().numpy()
    y_pred = preds[mask].cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return acc, p, r, f1


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data: Data = torch.load(args.input, map_location="cpu", weights_only=False)
    data = data.to(device)

    if not hasattr(data, "x") or data.x is None:
        raise ValueError("Input graph is missing node features `x`.")
    if not hasattr(data, "y") or data.y is None:
        raise ValueError("Input graph is missing labels `y`.")

    # Numeric stabilization for CICIDS-derived features
    data.x = torch.nan_to_num(data.x, nan=0.0, posinf=1e6, neginf=-1e6)
    data.x = torch.clamp(data.x, -1e6, 1e6)

    # Ensure binary labels are valid longs
    data.y = data.y.long()
    data.y = torch.where(data.y > 0, torch.ones_like(data.y), torch.zeros_like(data.y))

    # If masks absent, create them
    if not hasattr(data, "train_mask") or data.train_mask is None:
        train_mask, val_mask, test_mask = make_masks(
            data.num_nodes,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            device=device,
        )
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

    # Replace with your own MGNN import if available
    model = SimpleMGNNFallback(
        in_dim=data.num_features,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Handle class imbalance
    y_train = data.y[data.train_mask].float()
    pos = y_train.sum().item()
    neg = y_train.numel() - pos
    pos_weight = torch.tensor([(neg / max(pos, 1.0))], device=device)

    best_val_f1 = -1.0
    best_state = None
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(
            data.x,
            data.edge_index,
            data.edge_weight if hasattr(data, "edge_weight") else None,
        )

        loss = F.binary_cross_entropy_with_logits(
            logits[data.train_mask],
            data.y[data.train_mask].float(),
            pos_weight=pos_weight,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss encountered. Check feature scaling and graph data quality."
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        tr = eval_binary(model, data, data.train_mask)
        va = eval_binary(model, data, data.val_mask)

        if va[3] > best_val_f1:
            best_val_f1 = va[3]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        print(
            f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
            f"Train F1 {tr[3]:.4f} | Val F1 {va[3]:.4f}"
        )

        if patience_ctr >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    te = eval_binary(model, data, data.test_mask)
    print("\n=== Final Test Metrics (Binary) ===")
    print(f"Accuracy : {te[0]:.4f}")
    print(f"Precision: {te[1]:.4f}")
    print(f"Recall   : {te[2]:.4f}")
    print(f"F1-score : {te[3]:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model_best.pt")
    torch.save(data.cpu(), out_dir / "data_with_masks.pt")
    print(f"\nSaved model: {out_dir / 'model_best.pt'}")
    print(f"Saved data : {out_dir / 'data_with_masks.pt'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="artifacts/phase5_data_datab.pt")
    p.add_argument("--output-dir", type=str, default="artifacts/train_out")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
