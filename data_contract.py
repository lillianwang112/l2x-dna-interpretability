"""Shared row identity, family-held-out partitions, and verified array artifacts."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


PARTITIONS = ("train", "validation", "test")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass
class Dataset:
    sequences: list
    labels: np.ndarray
    family_ids: np.ndarray
    sample_ids: np.ndarray
    fingerprint: str
    motif: str

    def __len__(self):
        return len(self.sequences)


@dataclass
class Split:
    indices: dict
    fingerprint: str
    payload: dict

    def __getitem__(self, name):
        return self.indices[name]


def load_dataset(path):
    payload = json.loads(Path(path).read_text())
    sequences, labels, families, ids = [], [], [], []
    for prefix, label in [("positive", 1), ("negative", 0)]:
        try:
            group = payload[prefix + "_sequences"]
            group_families = payload[prefix + "_family_ids"]
            group_ids = payload[prefix + "_sample_ids"]
        except KeyError as error:
            raise ValueError("Dataset requires explicit family and sample IDs; regenerate metadata with simulation.py.") from error
        if not all(isinstance(x, list) for x in [group, group_families, group_ids]):
            raise ValueError("Dataset sequences and identity fields must be lists.")
        if not len(group) == len(group_families) == len(group_ids):
            raise ValueError("Each sequence must have a corresponding family ID and sample ID.")
        sequences.extend(group)
        labels.extend([label] * len(group))
        families.extend(group_families)
        ids.extend(group_ids)
    if not sequences or any(not isinstance(s, str) or not s or set(s) - set("ACGT") for s in sequences):
        raise ValueError("Expected nonempty uppercase DNA sequences.")
    if len({len(s) for s in sequences}) != 1:
        raise ValueError("All sequences must have the same length.")
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in families):
        raise ValueError("Family IDs must be nonnegative integers.")
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Sample IDs must be unique nonempty strings.")
    motif = payload.get("hyperparameters", {}).get("motif", "AAAAA")
    if not isinstance(motif, str) or not motif or set(motif) - set("ACGT"):
        raise ValueError("Invalid motif.")
    if any(bool(label) != (motif in seq) for label, seq in zip(labels, sequences)):
        raise ValueError("Dataset labels disagree with motif presence.")
    rows = list(zip(ids, families, labels, sequences))
    return Dataset(sequences, np.array(labels, dtype=int), np.array(families, dtype=int),
                   np.array(ids), _digest({"rows": rows, "motif": motif}), motif)


def _validate_partitions(dataset, partitions):
    if set(partitions) != set(PARTITIONS):
        raise ValueError("Split must contain train, validation, and test partitions.")
    indices = {}
    family_sets, sequence_sets = [], []
    all_indices = []
    for name in PARTITIONS:
        values = partitions[name]
        if not isinstance(values, list) or not values or any(type(v) is not int for v in values):
            raise ValueError("Split indices must be nonempty integer lists.")
        if len(set(values)) != len(values) or min(values) < 0 or max(values) >= len(dataset):
            raise ValueError("Split indices must be unique and within the dataset.")
        idx = np.array(values, dtype=int)
        if set(dataset.labels[idx]) != {0, 1}:
            raise ValueError(f"Both classes are required in the {name} partition.")
        indices[name] = idx
        all_indices.extend(values)
        family_sets.append(set(dataset.family_ids[idx].tolist()))
        sequence_sets.append({dataset.sequences[i] for i in values})
    if sorted(all_indices) != list(range(len(dataset))):
        raise ValueError("Partitions must cover every row exactly once without overlap.")
    for i in range(3):
        for j in range(i):
            if family_sets[i] & family_sets[j]:
                raise ValueError("An ancestor family appears in more than one partition.")
            if sequence_sets[i] & sequence_sets[j]:
                raise ValueError("An identical DNA sequence appears in more than one partition.")
    return indices


def make_split(dataset, seed=42, train_fraction=0.6, validation_fraction=0.2):
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Training/validation fractions must be positive and leave a test partition.")
    families = np.unique(dataset.family_ids)
    shuffled = np.random.default_rng(seed).permutation(families)
    n_train = int(len(shuffled) * train_fraction)
    n_val = int(len(shuffled) * validation_fraction)
    if n_train < 1 or n_val < 1 or n_train + n_val >= len(shuffled):
        raise ValueError("Not enough families for the requested three-way split.")
    groups = [shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]]
    partitions = {name: np.flatnonzero(np.isin(dataset.family_ids, group)).tolist()
                  for name, group in zip(PARTITIONS, groups)}
    indices = _validate_partitions(dataset, partitions)
    payload = {"schema_version": 1, "kind": "ancestor_family_holdout", "seed": int(seed),
               "dataset_fingerprint": dataset.fingerprint, "partitions": partitions,
               "train_fraction": train_fraction, "validation_fraction": validation_fraction}
    payload["fingerprint"] = _digest(payload)
    return Split(indices, payload["fingerprint"], payload)


def load_split(path, dataset):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 1 or payload.get("kind") != "ancestor_family_holdout":
        raise ValueError("Unsupported split format.")
    if payload.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("Split fingerprint does not match the ordered dataset.")
    fingerprint = payload.get("fingerprint")
    if _digest({k: v for k, v in payload.items() if k != "fingerprint"}) != fingerprint:
        raise ValueError("Split manifest checksum mismatch.")
    return Split(_validate_partitions(dataset, payload["partitions"]), fingerprint, payload)


def _array_digest(values, sample_ids, metadata):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(values, dtype="<f4").tobytes())
    digest.update(_digest({"sample_ids": sample_ids.tolist(), "metadata": metadata}).encode())
    return digest.hexdigest()


def write_artifact(path, kind, values, dataset, split, *, seed, teacher_fingerprint=None):
    if split.payload.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("The shared split belongs to another dataset.")
    if kind not in {"teacher_predictions", "selection_scores"}:
        raise ValueError("Unknown artifact kind.")
    values = np.asarray(values, dtype=np.float32)
    _validate_values(kind, values, dataset)
    metadata = {"schema_version": 1, "kind": kind, "dataset_fingerprint": dataset.fingerprint,
                "split_fingerprint": split.fingerprint, "seed": int(seed),
                "train_indices": split["train"].tolist()}
    if kind == "selection_scores" and (not isinstance(teacher_fingerprint, str) or len(teacher_fingerprint) != 64):
        raise ValueError("Selection scores require the teacher artifact fingerprint.")
    if teacher_fingerprint is not None:
        metadata["teacher_fingerprint"] = teacher_fingerprint
    fingerprint = _array_digest(values, dataset.sample_ids, metadata)
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Verified artifacts must use .npz files.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, values=values, sample_ids=dataset.sample_ids,
                        metadata=np.array(json.dumps(metadata, sort_keys=True)),
                        fingerprint=np.array(fingerprint))
    return fingerprint


def _validate_values(kind, values, dataset):
    expected = (len(dataset), 2) if kind == "teacher_predictions" else (len(dataset), len(dataset.sequences[0]))
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"Expected finite {kind} with shape {expected}.")
    if kind == "teacher_predictions" and (np.any(values < 0) or
                                           not np.allclose(values.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Teacher probabilities must be nonnegative and sum to one.")


def load_artifact(path, kind, dataset, split):
    if split.payload.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("The shared split belongs to another dataset.")
    if Path(path).suffix != ".npz":
        raise ValueError("Unverified array: use an identity-bound .npz export from the shared-split pipeline.")
    with np.load(path, allow_pickle=False) as artifact:
        try:
            values = np.array(artifact["values"], dtype=np.float32)
            sample_ids = np.array(artifact["sample_ids"])
            metadata = json.loads(str(artifact["metadata"].item()))
            fingerprint = str(artifact["fingerprint"].item())
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError("Invalid identity-bound artifact.") from error
    if metadata.get("schema_version") != 1 or metadata.get("kind") != kind:
        raise ValueError("Artifact kind/version does not match the requested data.")
    if metadata.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("Artifact dataset fingerprint mismatch.")
    if metadata.get("split_fingerprint") != split.fingerprint:
        raise ValueError("Artifact split fingerprint mismatch.")
    if not np.array_equal(sample_ids, dataset.sample_ids):
        raise ValueError("Artifact sample IDs are missing, reordered, or do not match the dataset.")
    if kind == "selection_scores" and (not isinstance(metadata.get("teacher_fingerprint"), str)
                                      or len(metadata["teacher_fingerprint"]) != 64):
        raise ValueError("Selection artifact has no valid teacher fingerprint.")
    if metadata.get("train_indices") != split["train"].tolist():
        raise ValueError("Artifact training rows do not match the shared training partition.")
    if _array_digest(values, sample_ids, metadata) != fingerprint:
        raise ValueError("Artifact values/metadata checksum mismatch.")
    _validate_values(kind, values, dataset)
    return values, {**metadata, "fingerprint": fingerprint}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/dataset.json")
    parser.add_argument("--output", default="data/family_split.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    dataset = load_dataset(args.dataset)
    split = make_split(dataset, args.seed)
    output = Path(args.output)
    if output.exists():
        previous = load_split(output, dataset)
        if previous.fingerprint != split.fingerprint:
            parser.error("A different split already exists at this path; use a new output path.")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(split.payload, indent=2) + "\n")
    print(json.dumps({"dataset_fingerprint": dataset.fingerprint, "split_fingerprint": split.fingerprint,
                      "partitions": {name: {"sequences": len(idx), "families": len(set(dataset.family_ids[idx])),
                                            "positive": int(dataset.labels[idx].sum()),
                                            "negative": int((dataset.labels[idx] == 0).sum())}
                                     for name, idx in split.indices.items()}}, indent=2))


if __name__ == "__main__":
    main()
