"""
ClarifySign - ISL Recognition Model Training Pipeline
Trains a Bidirectional LSTM neural network on processed skeletal landmark sequences.
Includes signer-leakage safeguards, reproducible random seeding,
stratified train/val/test splits, early stopping, and training metadata logging.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    MODEL_PATH,
    LABELS_PATH,
    TRAINING_METADATA_PATH,
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    DENSE_HIDDEN_DIM,
    DROPOUT_RATE,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED
)
from core.recognizer import ISLBiLSTM


class LandmarkDataset(Dataset):
    """PyTorch Dataset for landmark feature sequences."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(data_dir):
    """Loads all processed .npy sequences and labels."""
    X = []
    y = []
    file_metadata = []

    classes = sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])
    for cls in classes:
        cls_dir = Path(data_dir) / cls
        for f in cls_dir.glob("*.npy"):
            try:
                arr = np.load(f)
                if arr.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
                    X.append(arr)
                    y.append(cls)
                    file_metadata.append(f.name)
            except Exception as e:
                print(f"Skipping corrupted file {f}: {e}")

    return np.asarray(X, dtype=np.float32), np.asarray(y), file_metadata, classes


def split_data_safely(X, y_encoded, file_names, test_size=0.15, val_size=0.15, seed=RANDOM_SEED):
    """
    Splits data into train, val, and test partitions with signer leakage safeguard.

    When signer IDs are present in filenames, performs a SIGNER-DISJOINT split:
        - test_signer  : held completely out; the model never saw this signer during training
        - val_signer   : held for early stopping and calibration
        - train_signers: all remaining signers

    An assertion enforces that no signer ID appears in more than one partition.
    If signer IDs are absent, falls back to stratified random split with a clear warning.
    """
    import re
    # Check if signer metadata is present in filenames (e.g. 'SignerA', 'Signer_1', 'S01')
    signers = []
    for fn in file_names:
        parts = fn.split("_")
        signer = next((p for p in parts if p.lower().startswith("signer") or re.match(r"^s\d+$", p.lower())), None)
        signers.append(signer)

    has_signers = all(s is not None for s in signers) and len(set(signers)) >= 3

    if has_signers:
        unique_signers = sorted(list(set(signers)))
        print(f"[SAFEGUARD] Signer metadata detected: {unique_signers}. Performing signer-independent split.")
        test_signer = unique_signers[-1]
        val_signer = unique_signers[-2]
        train_signers = unique_signers[:-2]

        train_idx = [i for i, s in enumerate(signers) if s in train_signers]
        val_idx   = [i for i, s in enumerate(signers) if s == val_signer]
        test_idx  = [i for i, s in enumerate(signers) if s == test_signer]
        leakage_warning = None

        # === HARD ASSERTION: signer-disjoint check ===
        train_signers_in_split = set(signers[i] for i in train_idx)
        val_signers_in_split   = set(signers[i] for i in val_idx)
        test_signers_in_split  = set(signers[i] for i in test_idx)
        leakage_train_val  = train_signers_in_split & val_signers_in_split
        leakage_train_test = train_signers_in_split & test_signers_in_split
        leakage_val_test   = val_signers_in_split   & test_signers_in_split
        if leakage_train_val or leakage_train_test or leakage_val_test:
            raise AssertionError(
                "[SIGNER LEAKAGE DETECTED] The signer-disjoint split has been violated.\n"
                f"  train ∩ val:  {leakage_train_val}\n"
                f"  train ∩ test: {leakage_train_test}\n"
                f"  val   ∩ test: {leakage_val_test}\n"
                "Training halted to prevent data leakage and misleading accuracy numbers."
            )
        print(f"[SAFEGUARD] Signer-disjoint assertion PASSED: train={train_signers_in_split}, "
              f"val={{{val_signer}}}, test={{{test_signer}}}")
    else:
        leakage_warning = (
            "WARNING: Explicit signer IDs not detected across all samples. "
            "Performing stratified video-level split (70% train, 15% val, 15% test). "
            "Risk: Possible signer leakage across splits."
        )
        print(f"\n[SAFEGUARD WARNING] {leakage_warning}\n")

        idx = np.arange(len(X))
        tr_val_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            stratify=y_encoded,
            random_state=seed
        )
        rel_val_size = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            tr_val_idx,
            test_size=rel_val_size,
            stratify=y_encoded[tr_val_idx],
            random_state=seed
        )

    return train_idx, val_idx, test_idx, has_signers, leakage_warning




def train_model(
    data_dir=PROCESSED_DATA_DIR,
    model_out=MODEL_PATH,
    labels_out=LABELS_PATH,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    seed=RANDOM_SEED
):
    """Executes full reproducible training workflow."""
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Hardware device detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dev_name = "Apple Silicon (MPS)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dev_name = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        dev_name = "Host CPU"

    print(f"Executing training on: {dev_name}")

    # Load data
    X, y, file_names, raw_classes = load_dataset(data_dir)
    if len(X) == 0:
        raise RuntimeError(f"No valid .npy sequences found in {data_dir}. Run prepare_dataset.py first.")

    print(f"Loaded {len(X)} total sequences across {len(raw_classes)} classes.")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes_list = le.classes_.tolist()
    num_classes = len(classes_list)

    # Save label mapping
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_out, "w", encoding="utf-8") as f:
        json.dump(classes_list, f, indent=2)
    print(f"Saved class label mapping to {labels_out}")

    # Split dataset
    train_idx, val_idx, test_idx, has_signers, leakage_warning = split_data_safely(X, y_encoded, file_names, seed=seed)
    print(f"Split sizes -> Train: {len(train_idx)} | Validation: {len(val_idx)} | Test: {len(test_idx)}")

    train_loader = DataLoader(LandmarkDataset(X[train_idx], y_encoded[train_idx]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(LandmarkDataset(X[val_idx], y_encoded[val_idx]), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(LandmarkDataset(X[test_idx], y_encoded[test_idx]), batch_size=batch_size, shuffle=False)

    # Instantiate model
    model = ISLBiLSTM(
        input_dim=FEATURE_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS,
        num_classes=num_classes,
        dense_dim=DENSE_HIDDEN_DIM,
        dropout=DROPOUT_RATE
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    print("\n--- Starting Training ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == batch_y).sum().item()
            train_total += len(batch_y)

        train_loss /= max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4)
        })

        if epoch % 5 == 0 or epoch == 1 or val_loss < best_val_loss:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch} (patience={EARLY_STOPPING_PATIENCE}).")
                break

    training_time = time.time() - start_time

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model checkpoint
    model_out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": FEATURE_DIM,
        "hidden_dim": LSTM_HIDDEN_DIM,
        "num_layers": LSTM_NUM_LAYERS,
        "num_classes": num_classes,
        "dense_dim": DENSE_HIDDEN_DIM,
        "labels": classes_list
    }
    torch.save(checkpoint, model_out)
    print(f"\nSaved trained model weights to: {model_out}")

    # Evaluate on held-out Test set
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_top3_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            test_loss += loss.item() * len(batch_y)

            # Top-1
            preds = torch.argmax(logits, dim=-1)
            test_correct += (preds == batch_y).sum().item()

            # Top-3
            _, top3_preds = torch.topk(logits, k=min(3, num_classes), dim=-1)
            for i in range(len(batch_y)):
                if batch_y[i] in top3_preds[i]:
                    test_top3_correct += 1

            test_total += len(batch_y)

    test_loss /= max(1, test_total)
    test_top1_acc = test_correct / max(1, test_total)
    test_top3_acc = test_top3_correct / max(1, test_total)

    print(f"\n==========================================")
    print(f"FINAL TEST SET EVALUATION RESULTS:")
    print(f"Test Loss:        {test_loss:.4f}")
    print(f"Test Top-1 Acc:   {test_top1_acc*100:.2f}%")
    print(f"Test Top-3 Acc:   {test_top3_acc*100:.2f}%")
    print(f"Training Time:    {training_time:.2f} seconds")
    print(f"==========================================\n")

    # Record full verified training metadata
    metadata = {
        "model_architecture": "BiLSTM + LayerNorm + Dual Temporal Pooling + GELU Dense Head",
        "feature_dim": FEATURE_DIM,
        "sequence_length": SEQUENCE_LENGTH,
        "num_classes": num_classes,
        "classes": classes_list,
        "total_dataset_samples": len(X),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "hardware_device": dev_name,
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lstm_hidden_dim": LSTM_HIDDEN_DIM,
            "lstm_num_layers": LSTM_NUM_LAYERS,
            "dropout": DROPOUT_RATE,
            "seed": seed
        },
        "verified_metrics": {
            "best_validation_loss": round(float(best_val_loss), 4),
            "test_loss": round(float(test_loss), 4),
            "test_top1_accuracy": round(float(test_top1_acc), 4),
            "test_top3_accuracy": round(float(test_top3_acc), 4),
            "training_duration_seconds": round(training_time, 2)
        },
        "signer_leakage_safeguard": {
            "signer_independent_split": has_signers,
            "warning": leakage_warning
        },
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_file": "models/isl_bilstm.pt",
        "labels_file": "models/labels.json"
    }

    with open(TRAINING_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved verified training metadata to: {TRAINING_METADATA_PATH}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train ClarifySign ISL Recognition Model")
    parser.add_argument("--data", default=str(PROCESSED_DATA_DIR), help="Path to processed sequences")
    parser.add_argument("--out", default=str(MODEL_PATH), help="Output PyTorch model path")
    parser.add_argument("--labels", default=str(LABELS_PATH), help="Output class labels path")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    train_model(
        data_dir=Path(args.data),
        model_out=Path(args.out),
        labels_out=Path(args.labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()
