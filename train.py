"""
Train a simple 1D-convolutional classifier on the toy motif dataset.

Architecture:
- one-hot encode DNA (A,C,G,T) into 4 channels
- Conv1d(in_channels=4, out_channels=32, kernel_size=10)
- ReLU
- flatten and Linear to a single logit
- sigmoid for binary binding prediction

Data:
- Loads sequences from ./data/dataset.json
- Shared ancestor-family train/validation/test manifest via run_advisor_training.py
- Records validation accuracy, AUROC, and confusion counts; reserves test metrics
- Exports ordered probabilities with sample IDs and dataset/split fingerprints
"""

import json
import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, random_split

RANDOM_SEED = 0
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8

ALPHABET = ["A", "C", "G", "T"]
CHAR_TO_INDEX = {c: i for i, c in enumerate(ALPHABET)}
SEQ_LEN = 100

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.json")
CONFUSION_PNG = os.path.join(BASE_DIR, "confusion_matrix.png")
STATS_JSON = os.path.join(BASE_DIR, "stats.json")

# New outputs for L2X (teacher predictions)
PRED_PROBS_NPY = os.path.join(BASE_DIR, "cnn_predictions.npy")          # shape (N,)
PRED_PROBS_L2X_NPY = os.path.join(BASE_DIR, "cnn_predictions_l2x.npy")  # shape (N, 2)


def one_hot_encode(seq: str) -> torch.Tensor:
    """Convert a DNA string to a (4, L) one-hot tensor."""
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for pos, ch in enumerate(seq):
        idx = CHAR_TO_INDEX.get(ch, None)
        if idx is not None:
            arr[idx, pos] = 1.0
    return torch.from_numpy(arr)


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[str], labels: List[int]) -> None:
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        label = float(self.labels[idx])
        return one_hot_encode(seq), torch.tensor(label, dtype=torch.float32)


class ConvClassifier(nn.Module):
    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=10)
        conv_out_len = seq_len - 10 + 1  # valid padding
        self.classifier = nn.Linear(32 * conv_out_len, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        logit = self.classifier(x)
        return logit.squeeze(1)


def load_data(path: str) -> SequenceDataset:
    """Load canonical positive-then-negative row order without shuffling identity."""
    from data_contract import load_dataset
    dataset = load_dataset(path)
    return SequenceDataset(dataset.sequences, dataset.labels.tolist())


def train(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits: List[float] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x).cpu()
            all_logits.extend(logits.tolist())
            all_labels.extend(batch_y.tolist())
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    labels_np = np.array(all_labels, dtype=int)
    acc = accuracy_score(labels_np, preds)
    try:
        auroc = roc_auc_score(labels_np, probs)
    except ValueError:
        auroc = float("nan")  # only one class present
    cm = confusion_matrix(labels_np, preds, labels=[0, 1])
    return acc, auroc, cm


def plot_confusion_matrix(cm: np.ndarray, path: str) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def get_all_predictions(model: nn.Module, dataset: Dataset, device: torch.device) -> np.ndarray:
    """
    Return P(Y=1|X) for every example in `dataset` in dataset order.
    Output shape: (N,)
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_probs: List[float] = []
    with torch.no_grad():
        for batch_x, _batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
    return np.array(all_probs, dtype=np.float32)


def main():
    # Keep the supplied classifier/utilities; use one authoritative training/export entry point.
    from run_advisor_training import main as shared_split_main
    shared_split_main()


if __name__ == "__main__":
    main()
