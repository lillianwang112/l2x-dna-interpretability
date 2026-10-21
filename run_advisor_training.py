"""Train the mentor-provided teacher on a shared family split and export verified predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import torch
from torch.utils.data import DataLoader, Dataset

from data_contract import load_dataset, load_split, write_artifact
from train import ConvClassifier, one_hot_encode


class RowDataset(Dataset):
    """Keep row identity attached to every tensor, including during shuffling."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        return (one_hot_encode(self.dataset.sequences[index]),
                torch.tensor(self.dataset.labels[index], dtype=torch.float32),
                self.dataset.sample_ids[index])


def predict_rows(model, dataset, indices, batch_size=64):
    indices = np.asarray(indices, dtype=int)
    loader = DataLoader(RowDataset(dataset, indices), batch_size=batch_size, shuffle=False)
    probabilities, row_ids = [], []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for inputs, _, ids in loader:
            p1 = torch.sigmoid(model(inputs.to(device))).cpu().numpy()
            probabilities.append(np.stack([1.0 - p1, p1], axis=1))
            row_ids.extend(ids)
    if row_ids != dataset.sample_ids[indices].tolist():
        raise ValueError("Teacher inference rows do not match the requested sample IDs.")
    result = np.concatenate(probabilities, axis=0)
    if not np.isfinite(result).all():
        raise ValueError("Teacher produced nonfinite probabilities.")
    return result


def export_predictions(model, dataset, split, path, batch_size=64, seed=0):
    indices = np.arange(len(dataset))
    probabilities = predict_rows(model, dataset, indices, batch_size)
    reverse = predict_rows(model, dataset, indices[::-1], batch_size)[::-1]
    if not np.allclose(probabilities, reverse, rtol=1e-5, atol=1e-6):
        raise ValueError("Independent reverse-order inference did not reproduce the aligned export.")
    fingerprint = write_artifact(path, "teacher_predictions", probabilities, dataset, split, seed=seed)
    return probabilities, {
        "prediction_fingerprint": fingerprint,
        "alignment_check": {"method": "independent reverse-order inference, then restored row order",
                            "rows_verified": len(dataset),
                            "max_absolute_difference": float(np.max(np.abs(probabilities - reverse)))},
    }


def run_teacher(dataset_path, split_path, out_dir, epochs=10, batch_size=64, seed=0, lr=1e-3):
    if epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("epochs, batch_size, and learning rate must be positive.")
    dataset = load_dataset(dataset_path)
    split = load_split(split_path, dataset)
    output = Path(out_dir)
    names = ["teacher.pt", "teacher_predictions.npz", "teacher_report.json"]
    if any((output / name).exists() for name in names):
        raise FileExistsError("Teacher output already exists; choose a new output directory to preserve it.")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    # CPU is the checked configuration; no implicit device or backend changes.
    model = ConvClassifier(seq_len=len(dataset.sequences[0])).to("cpu")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(RowDataset(dataset, split["train"]), batch_size=batch_size,
                              shuffle=True, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels, _ in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(inputs)
        validation = predict_rows(model, dataset, split["validation"], batch_size)
        truth = dataset.labels[split["validation"]]
        row = {"epoch": epoch + 1, "train_loss": total_loss / len(split["train"]),
               "validation_accuracy": float(accuracy_score(truth, validation.argmax(axis=1))),
               "validation_auroc": float(roc_auc_score(truth, validation[:, 1]))}
        history.append(row)
        print(json.dumps(row), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "teacher.pt"
    torch.save(model.state_dict(), checkpoint)
    probabilities, export = export_predictions(model, dataset, split, output / "teacher_predictions.npz",
                                                batch_size=batch_size, seed=seed)
    val_idx = split["validation"]
    report = {"dataset_fingerprint": dataset.fingerprint, "split_fingerprint": split.fingerprint,
              "seed": seed, "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
              "training_population": "train families only", "evaluation_population": "validation families",
              "test_metrics_evaluated": False,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "partition_sizes": {name: len(idx) for name, idx in split.indices.items()},
              "history": history,
              "validation_confusion_matrix": confusion_matrix(dataset.labels[val_idx],
                                                                probabilities[val_idx].argmax(axis=1), labels=[0, 1]).tolist(),
              **export}
    (output / "teacher_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_json", default="data/dataset.json")
    parser.add_argument("--split_json", default="data/family_split.json")
    parser.add_argument("--out_dir", default="outputs/teacher")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)
    run_teacher(args.dataset_json, args.split_json, args.out_dir, args.epochs, args.batch_size, args.seed, args.lr)


if __name__ == "__main__":
    main()
