"""Model Training and Validation Pipeline for Bird Species Classification.

Features:
  - torch.nn.CrossEntropyLoss for multi-class classification
  - AdamW optimizer with configurable weight decay and learning rate
  - ReduceLROnPlateau learning rate scheduler
  - Early stopping tracking validation loss
  - Full epoch-by-epoch metric logging to results/training_history.csv
  - Checkpoint saving for best validation performance
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import sys
import time
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import set_seed, load_config, get_device, save_checkpoint, save_json
from src.dataset import create_dataloaders
from src.model import build_model


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after given patience."""

    def __init__(self, patience: int = 4, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_loss_epoch = -1
        self.early_stop = False
        self.stopped_epoch = None
        self.stopping_reason = None

    def __call__(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_loss_epoch = epoch
            self.counter = 0
            return True  # Improvement occurred
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch
                self.stopping_reason = (
                    f"Validation loss did not improve by at least {self.min_delta} for "
                    f"{self.patience} consecutive epochs after reaching minimum {self.best_loss:.4f} "
                    f"at epoch {self.best_loss_epoch}."
                )
            return False  # No improvement


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:
    """Train model for a single epoch. Dropout and BatchNorm are in training mode."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (correct / total) * 100.0 if total > 0 else 0.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """Evaluate model on validation data. Dropout is deactivated via model.eval()."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (correct / total) * 100.0 if total > 0 else 0.0
    return epoch_loss, epoch_acc


def run_training(config_path: str = "configs/config.yaml") -> dict:
    """Run full training and validation loop using configuration parameters."""
    config = load_config(config_path)
    seed = config["project"]["seed"]
    set_seed(seed)

    device = get_device(config["training"].get("device", "auto"))
    processed_dir = os.path.join(PROJECT_ROOT, config["dataset"]["processed_dir"])

    print("[Training] Building DataLoaders...")
    train_loader, val_loader, _, selected_species, class_to_idx = create_dataloaders(
        processed_dir, config
    )
    print(f"[Training] Data loaded: {len(train_loader.dataset)} training samples, "
          f"{len(val_loader.dataset)} validation samples.")

    model = build_model(config).to(device)

    # Optimization setup
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    epochs = int(config["training"]["epochs"])
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # LR Scheduler
    sched_cfg = config["training"].get("scheduler", {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg.get("mode", "min"),
        factor=float(sched_cfg.get("factor", 0.5)),
        patience=int(sched_cfg.get("patience", 2)),
        min_lr=float(sched_cfg.get("min_lr", 1e-5))
    )

    # Early stopping
    es_cfg = config["training"].get("early_stopping", {})
    early_stopping = EarlyStopping(
        patience=int(es_cfg.get("patience", 4)),
        min_delta=float(es_cfg.get("min_delta", 0.001))
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = -1
    checkpoint_path = os.path.join(PROJECT_ROOT, config["paths"]["best_checkpoint"])
    history_records = []

    print("\n" + "=" * 70)
    print(f"Beginning Training for {epochs} Epochs on {device}")
    print("=" * 70)

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        improved = early_stopping(val_loss, epoch)
        elapsed = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "learning_rate": current_lr,
            "time_seconds": round(elapsed, 2)
        }
        history_records.append(record)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}% | "
            f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s"
        )

        # Save checkpoint if best validation loss achieved
        if improved or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "class_names": selected_species,
                "class_to_idx": class_to_idx,
                "config": config
            }
            save_checkpoint(checkpoint, checkpoint_path)
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")

        if es_cfg.get("enabled", True) and early_stopping.early_stop:
            print(f"\n[Early Stopping] Triggered after {epoch} epochs. Reason: {early_stopping.stopping_reason}")
            break

    if not early_stopping.early_stop:
        early_stopping.stopping_reason = (
            f"Completed all {epochs} scheduled epochs without triggering early stopping "
            f"(final patience counter: {early_stopping.counter}/{early_stopping.patience})."
        )

    total_duration = time.time() - start_total_time
    print("=" * 70)
    print(f"Training completed in {total_duration / 60:.2f} minutes.")
    print(f"Best Validation Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.2f}%")
    print(f"Stopping Reason: {early_stopping.stopping_reason}")
    print("=" * 70)

    # Save history to CSV
    history_df = pd.DataFrame(history_records)
    history_path = os.path.join(PROJECT_ROOT, config["paths"]["history_file"])
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    history_df.to_csv(history_path, index=False)
    print(f"[Training] Saved epoch training history to {history_path}")

    # Record and save training summary including explicit early stopping outcome
    training_summary = {
        "total_epochs_scheduled": epochs,
        "epochs_completed": len(history_records),
        "early_stopping_enabled": bool(es_cfg.get("enabled", True)),
        "early_stopping_triggered": bool(early_stopping.early_stop),
        "stopped_epoch": early_stopping.stopped_epoch,
        "stopping_reason": early_stopping.stopping_reason,
        "patience": early_stopping.patience,
        "min_delta": early_stopping.min_delta,
        "final_patience_counter": early_stopping.counter,
        "best_epoch": best_epoch,
        "best_val_loss": round(float(best_val_loss), 4),
        "best_val_acc": round(float(best_val_acc), 2),
        "total_training_duration_seconds": round(total_duration, 2)
    }
    summary_path = os.path.join(
        PROJECT_ROOT,
        config["paths"].get("training_summary_file", "results/training_summary.json")
    )
    save_json(training_summary, summary_path)
    print(f"[Training] Saved training summary to {summary_path}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "history": history_records,
        "training_summary": training_summary
    }


if __name__ == "__main__":
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    run_training(cfg_arg)
