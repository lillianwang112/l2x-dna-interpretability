"""Measure feature selections on the synthetic motif task.

All metrics are computed from the supplied scores. The command-line report verifies
row identity and evaluates the named shared partition. Metrics are descriptive.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from data_contract import load_dataset, load_split, load_artifact


def load_dataset_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    positives = data["positive_sequences"]
    negatives = data["negative_sequences"]
    if not isinstance(positives, list) or not isinstance(negatives, list):
        raise ValueError("Positive and negative sequences must be lists.")
    return positives + negatives, np.array([1] * len(positives) + [0] * len(negatives))


def select_topk(scores, k):
    """Select exactly k positions; equal scores prefer the lower position index."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not scores.shape[0] or not scores.shape[1]:
        raise ValueError("Scores must be a nonempty (examples, positions) array.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must contain only finite values.")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer.")
    if not 1 <= k <= scores.shape[1]:
        raise ValueError(f"k must be between 1 and {scores.shape[1]}.")
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def motif_windows(seq, motif="AAAAA"):
    """Return inclusive windows, including overlapping motif occurrences."""
    if not motif:
        raise ValueError("Motif must not be empty.")
    return [(i, i + len(motif) - 1) for i in range(len(seq) - len(motif) + 1)
            if seq[i:i + len(motif)] == motif]


def check_hit(selected_indices, true_windows):
    return any(start <= index <= end for index in selected_indices
               for start, end in true_windows)


def is_contiguous(indices):
    positions = sorted(set(indices))
    return (len(positions) == len(indices)
            and all(b - a == 1 for a, b in zip(positions, positions[1:])))


def _mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def evaluate_selection(sequences, labels, scores, k=5, motif="AAAAA", trials=100, seed=0):
    """Evaluate one explicitly supplied population; missing classes have null metrics."""
    selected = select_topk(scores, k)
    if len(sequences) != len(selected):
        raise ValueError("Score rows must match the number of sequences.")
    length = np.asarray(scores).shape[1]
    if any(not isinstance(s, str) or len(s) != length or set(s) - set("ACGT")
           for s in sequences):
        raise ValueError("Sequences must be equally long uppercase A/C/G/T strings matching score width.")
    if not motif or set(motif) - set("ACGT"):
        raise ValueError("Motif must be a nonempty uppercase A/C/G/T string.")
    labels = np.asarray(labels)
    if labels.shape != (len(sequences),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be one binary value per sequence.")
    if not isinstance(trials, (int, np.integer)) or trials < 1:
        raise ValueError("trials must be a positive integer.")
    windows = [motif_windows(s, motif) for s in sequences]
    if any(bool(w) != bool(y) for w, y in zip(windows, labels)):
        raise ValueError("Labels disagree with motif presence in the supplied sequences.")
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    motif_positions = []
    for occurrence in windows:
        motif_positions.append({p for start, end in occurrence for p in range(start, end + 1)})

    hits, precisions, recalls, full_windows, expected_random = [], [], [], [], []
    for i in positive:
        chosen = set(selected[i].tolist())
        truth = motif_positions[i]
        overlap = len(chosen & truth)
        hits.append(bool(overlap))
        precisions.append(overlap / k)
        recalls.append(overlap / len(truth))
        full_windows.append(any(set(range(a, b + 1)) <= chosen for a, b in windows[i]))
        # Exact probability that a uniformly random k-subset hits the union of motif bases.
        available_nonmotif = length - len(truth)
        miss = math.comb(available_nonmotif, k) if available_nonmotif >= k else 0
        expected_random.append(1.0 - miss / math.comb(length, k))

    rng = np.random.default_rng(seed)
    random_rates = []
    if len(positive):
        for _ in range(trials):
            random_hits = [check_hit(rng.choice(length, k, replace=False), windows[i]) for i in positive]
            random_rates.append(float(np.mean(random_hits)))
    model_hit_rate = _mean_or_none(hits)
    random_expected = _mean_or_none(expected_random)

    def selected_a_frequency(indices):
        return _mean_or_none([sum(sequences[i][j] == "A" for j in selected[i]) / k for i in indices])

    return {
        "schema_version": 1,
        "n_sequences": len(sequences),
        "sequence_length": length,
        "n_positive": int(len(positive)),
        "n_negative": int(len(negative)),
        "k": int(k),
        "motif": motif,
        "tie_rule": "lower position index first",
        "positive_any_motif_overlap_rate": model_hit_rate,
        "positive_selected_position_precision": _mean_or_none(precisions),
        "positive_motif_position_recall": _mean_or_none(recalls),
        "positive_full_motif_window_recovery_rate": _mean_or_none(full_windows),
        "random_baseline": {
            "expected_any_overlap_rate": random_expected,
            "monte_carlo_mean": _mean_or_none(random_rates),
            "monte_carlo_std": float(np.std(random_rates)) if random_rates else None,
            "trials": int(trials),
            "seed": int(seed),
            "model_minus_expected_random": (model_hit_rate - random_expected) if len(positive) else None,
        },
        "selection_content": {
            "positive_selected_a_frequency": selected_a_frequency(positive),
            "negative_selected_a_frequency": selected_a_frequency(negative),
            "negative_background_a_frequency": _mean_or_none([sequences[i].count("A") / length for i in negative]),
            "negative_contiguous_selection_rate": _mean_or_none([is_contiguous(selected[i]) for i in negative]),
        },
        "interpretation": "Descriptive selection diagnostics; random-baseline gap is not a significance test.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/dataset.json")
    parser.add_argument("--scores", default="outputs/l2x/selection_scores.npz")
    parser.add_argument("--split_json", default="data/family_split.json")
    parser.add_argument("--partition", choices=["validation", "test"], default="validation")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--motif", default="AAAAA")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/selection_metrics.json")
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.dataset)
        split = load_split(args.split_json, dataset)
        scores, metadata = load_artifact(args.scores, "selection_scores", dataset, split)
        indices = split[args.partition]
        if args.motif != dataset.motif:
            raise ValueError("Requested motif does not match the dataset motif.")
        result = evaluate_selection([dataset.sequences[i] for i in indices], dataset.labels[indices],
                                    scores[indices], args.k, args.motif, args.trials, args.seed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    result["population"] = args.partition + " ancestor families"
    result["dataset_fingerprint"] = dataset.fingerprint
    result["split_fingerprint"] = split.fingerprint
    result["selection_fingerprint"] = metadata["fingerprint"]
    result["teacher_fingerprint"] = metadata.get("teacher_fingerprint")
    result["inputs"] = {"dataset": str(args.dataset), "scores": str(args.scores)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
