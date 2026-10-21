"""Lock an evaluation protocol, then evaluate saved models without fitting them."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from compare_mask_training import array_hash, exact_mask, infer, prediction_metrics
from data_contract import _digest, load_artifact, load_dataset, load_split
from dna_l2x import build_l2x_model, load_dataset_json
from run_diagnostics import get_approximator
from validation_checks import evaluate_selection, motif_windows, select_topk


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    keys = [key for key, value in rows[0].items() if isinstance(value, float) or value is None]
    result = {}
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = {"mean": float(np.mean(values)) if values else None,
                       "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    return result


def lock_protocol(root, path, *, dataset_path="data/dataset.json", split_path="data/family_split.json",
                  teacher_path="outputs/teacher/teacher_predictions.npz", checkpoint_dir="outputs/fidelity",
                  comparison_dir="outputs/mask_comparison", budgets=tuple(range(1, 11)), seeds=(0, 1, 2),
                  reference_k=5, random_trials=100):
    root, path = Path(root).resolve(), Path(path)
    if path.exists():
        raise FileExistsError("Protocol exists; retain it and choose a new path for a new reproduction.")
    if reference_k not in budgets or not budgets or not seeds or random_trials < 1:
        raise ValueError("Require a reference budget, seeds, and positive random-trial count.")
    dataset = load_dataset(root / dataset_path)
    split = load_split(root / split_path, dataset)
    teacher, meta = load_artifact(root / teacher_path, "teacher_predictions", dataset, split)
    files = [dataset_path, split_path, teacher_path]
    for k in budgets:
        for seed in seeds:
            folder = f"{checkpoint_dir}/k{k}_seed{seed}"
            files.extend([f"{folder}/l2x.weights.h5", f"{folder}/selection_scores.npz"])
    for seed in seeds:
        for mode in ["soft", "hard"]:
            files.append(f"{comparison_dir}/seed{seed}_{mode}/approximator.weights.h5")
    # Examples are chosen by sample ID and label before any test performance is computed.
    test_rows = sorted(split["test"].tolist(), key=lambda i: dataset.sample_ids[i])
    example_rows = [i for i in test_rows if dataset.labels[i] == 1][:2]
    example_rows += [i for i in test_rows if dataset.labels[i] == 0][:1]
    train_teacher_classes = np.bincount(teacher[split["train"]].argmax(1), minlength=2)
    protocol = {
        "schema_version": 1, "kind": "frozen_checkpoint_evaluation", "locked_before_test_metrics": True,
        "inputs": {"dataset": dataset_path, "split": split_path, "teacher": teacher_path,
                   "checkpoints": checkpoint_dir, "comparison": comparison_dir},
        "dataset_fingerprint": dataset.fingerprint, "split_fingerprint": split.fingerprint,
        "teacher_fingerprint": meta["fingerprint"], "budgets": list(budgets), "seeds": list(seeds),
        "reference_k": reference_k, "reference_budget_rationale": "known motif length, chosen before test evaluation",
        "partitions": ["validation", "test"], "temperature": .5, "batch_size": 128,
        "mask": "exact-k binary; lower position index resolves ties", "random_trials": random_trials,
        "checkpoint_selection": "all recorded final-epoch models; no best-run selection",
        "comparison": "both equal-budget continuation arms with teacher and selectors fixed",
        "constant_teacher_class": int(train_teacher_classes.argmax()),
        "constant_label_class": int(np.bincount(dataset.labels[split["train"]], minlength=2).argmax()),
        "probability_baseline": "mean teacher probabilities on training rows",
        "metrics": ["any motif overlap", "complete motif window", "position precision and recall",
                    "raw and balanced teacher agreement", "class-specific agreement", "teacher KL",
                    "teacher cross entropy", "probability MAE", "true-label accuracy",
                    "teacher label AUROC", "remove-mask approximator diagnostics"],
        "uncertainty": "mean and sample SD over explainer seeds with one teacher; no confidence intervals",
        "example_rule": "first two motif-positive and first negative test sample IDs in lexicographic order; original reference-k seed-0 model",
        "example_sample_ids": dataset.sample_ids[example_rows].tolist(),
        "artifact_sha256": {str(name): sha256(root / name) for name in files},
        "source_sha256": {p.name: sha256(p) for p in sorted(root.glob("*.py"))},
    }
    protocol["fingerprint"] = _digest(protocol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def load_protocol(root, path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "frozen_checkpoint_evaluation":
        raise ValueError("Unsupported evaluation protocol.")
    if _digest({k: v for k, v in protocol.items() if k != "fingerprint"}) != protocol.get("fingerprint"):
        raise ValueError("Protocol checksum mismatch.")
    for group in ["artifact_sha256", "source_sha256"]:
        for name, digest in protocol[group].items():
            if sha256(Path(root) / name) != digest:
                raise ValueError(f"Locked evaluation input changed: {name}")
    return protocol


def evaluate(root, protocol_path, output_path):
    root, output_path = Path(root).resolve(), Path(output_path)
    if output_path.exists():
        raise FileExistsError("Evaluation report exists; preserve the recorded result.")
    protocol = load_protocol(root, protocol_path)
    inputs = protocol["inputs"]
    dataset = load_dataset(root / inputs["dataset"])
    split = load_split(root / inputs["split"], dataset)
    teacher, meta = load_artifact(root / inputs["teacher"], "teacher_predictions", dataset, split)
    if (dataset.fingerprint, split.fingerprint, meta["fingerprint"]) != (
            protocol["dataset_fingerprint"], protocol["split_fingerprint"], protocol["teacher_fingerprint"]):
        raise ValueError("Evaluation identity does not match the locked protocol.")
    _, x, _ = load_dataset_json(str(root / inputs["dataset"]), len(dataset.sequences[0]))
    prior = teacher[split["train"]].mean(0, dtype=np.float64)
    prior /= prior.sum()
    report = {"schema_version": 1, "protocol_fingerprint": protocol["fingerprint"],
              "dataset_fingerprint": dataset.fingerprint, "split_fingerprint": split.fingerprint,
              "teacher_fingerprint": meta["fingerprint"], "seeds": protocol["seeds"],
              "reference_k": protocol["reference_k"], "test_metrics_evaluated": True,
              "training_teacher_prior": prior.tolist(), "partitions": {}, "examples": []}
    for partition in protocol["partitions"]:
        idx = split[partition]
        labels, t = dataset.labels[idx], teacher[idx]
        confusion = np.zeros((2, 2), dtype=int)
        np.add.at(confusion, (labels, t.argmax(1)), 1)
        report["partitions"][partition] = {
            "n_sequences": len(idx), "n_families": len(np.unique(dataset.family_ids[idx])),
            "n_positive": int(labels.sum()), "n_negative": int((labels == 0).sum()),
            "sample_ids_fingerprint": _digest(dataset.sample_ids[idx].tolist()),
            "teacher": {"true_label_accuracy": float(np.mean(t.argmax(1) == labels)),
                        "true_label_auroc": float(roc_auc_score(labels, t[:, 1])),
                        "true_vs_predicted_confusion": confusion.tolist(),
                        "predicted_class_counts": np.bincount(t.argmax(1), minlength=2).tolist()},
            "baselines": {"constant_teacher_class": protocol["constant_teacher_class"],
                          "constant_class_teacher_agreement": float(np.mean(t.argmax(1) == protocol["constant_teacher_class"])),
                          "constant_class_balanced_agreement": .5 if len(np.unique(t.argmax(1))) == 2 else None,
                          "training_label_majority_accuracy": float(np.mean(labels == protocol["constant_label_class"])),
                          "training_teacher_prior": prediction_metrics(t, np.tile(prior, (len(idx), 1)), labels)},
            "sweep_runs": [], "mask_comparison_runs": [],
        }
    batch = protocol["batch_size"]
    all_eval_idx = np.concatenate([split[p] for p in protocol["partitions"]])
    progress = root / "outputs/final_evaluation/progress.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    for k in protocol["budgets"]:
        for seed in protocol["seeds"]:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            model, selector = build_l2x_model(x.shape[1], k, protocol["temperature"], .001)
            model.optimizer.build(model.trainable_variables)
            folder = root / inputs["checkpoints"] / f"k{k}_seed{seed}"
            model.load_weights(folder / "l2x.weights.h5")
            initial_weights = array_hash(model.get_weights())
            scores, score_meta = load_artifact(folder / "selection_scores.npz", "selection_scores", dataset, split)
            if score_meta["teacher_fingerprint"] != meta["fingerprint"] or score_meta["seed"] != seed:
                raise ValueError("Selector artifact has a different teacher or seed.")
            observed = infer(selector, x[all_eval_idx], batch).squeeze(-1)
            if not np.allclose(observed, scores[all_eval_idx], rtol=0, atol=1e-6):
                raise ValueError("Checkpoint selector does not reproduce the saved score artifact.")
            q = get_approximator(model)
            masks = {}
            for partition in protocol["partitions"]:
                idx = split[partition]
                hard = exact_mask(scores[idx, :, None], k)
                masks[partition] = hard
                kept = infer(q, x[idx] * hard, batch)
                removed = infer(q, x[idx] * (1 - hard), batch)
                selection = evaluate_selection([dataset.sequences[i] for i in idx], dataset.labels[idx], scores[idx],
                                               k, dataset.motif, protocol["random_trials"], seed)
                row = {"k": k, "seed": seed, "selection_fingerprint": score_meta["fingerprint"],
                       "selection": selection, "keep": prediction_metrics(teacher[idx], kept, dataset.labels[idx]),
                       "remove": prediction_metrics(teacher[idx], removed, dataset.labels[idx])}
                report["partitions"][partition]["sweep_runs"].append(row)
                if partition == "test" and k == protocol["reference_k"] and seed == protocol["seeds"][0]:
                    for sample_id in protocol["example_sample_ids"]:
                        local = np.flatnonzero(dataset.sample_ids[idx] == sample_id)
                        if len(local) != 1:
                            raise ValueError("Illustrative sample is not uniquely identified in the test set.")
                        j = int(local[0]); i = int(idx[j])
                        report["examples"].append({"sample_id": sample_id, "sequence": dataset.sequences[i],
                            "label": int(dataset.labels[i]), "k": k, "seed": seed,
                            "scores": scores[i].tolist(), "selected_positions": select_topk(scores[i:i+1], k)[0].tolist(),
                            "motif_windows": motif_windows(dataset.sequences[i], dataset.motif),
                            "teacher_positive_probability": float(teacher[i, 1]),
                            "keep_positive_probability": float(kept[j, 1])})
            if array_hash(model.get_weights()) != initial_weights:
                raise ValueError("Evaluation mutated model weights.")
            if k == protocol["reference_k"]:
                for mode in ["soft", "hard"]:
                    weights = root / inputs["comparison"] / f"seed{seed}_{mode}/approximator.weights.h5"
                    q.load_weights(weights)
                    for partition in protocol["partitions"]:
                        idx = split[partition]
                        p = infer(q, x[idx] * masks[partition], batch)
                        report["partitions"][partition]["mask_comparison_runs"].append(
                            {"seed": seed, "mode": mode, "metrics": prediction_metrics(teacher[idx], p, dataset.labels[idx])})
            progress.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"Evaluated saved k={k}, seed={seed}", flush=True)
    for partition, population in report["partitions"].items():
        summaries = []
        for k in protocol["budgets"]:
            rows = [r for r in population["sweep_runs"] if r["k"] == k]
            localization = [{key: r["selection"][key] for key in ["positive_any_motif_overlap_rate",
                            "positive_full_motif_window_recovery_rate", "positive_selected_position_precision",
                            "positive_motif_position_recall"]} for r in rows]
            summaries.append({"k": k, "n_seeds": len(rows), "localization": summarize(localization),
                              "keep": summarize([r["keep"] for r in rows]),
                              "remove": summarize([r["remove"] for r in rows]),
                              "uniform_random_any_overlap": rows[0]["selection"]["random_baseline"]["expected_any_overlap_rate"]})
        population["sweep_summary"] = summaries
        population["mask_comparison_summary"] = {
            mode: summarize([r["metrics"] for r in population["mask_comparison_runs"] if r["mode"] == mode])
            for mode in ["soft", "hard"]}
        deltas = []
        for seed in protocol["seeds"]:
            soft, hard = [next(r["metrics"] for r in population["mask_comparison_runs"] if r["seed"] == seed and r["mode"] == mode)
                          for mode in ["soft", "hard"]]
            deltas.append({"seed": seed, **{key: hard[key] - soft[key] for key in hard
                                           if isinstance(hard[key], float) and soft[key] is not None}})
        population["paired_hard_minus_soft"] = {"per_seed": deltas, "summary": summarize(deltas)}
    load_protocol(root, protocol_path)  # Recheck source and artifact hashes after inference.
    report["status"] = "complete"
    report["weights_and_inputs_unchanged"] = True
    report["uncertainty"] = "Mean and sample SD across three explainer seeds with one fixed teacher; not confidence intervals."
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="results/evaluation_protocol.json")
    parser.add_argument("--output", default="results/final_evaluation.json")
    parser.add_argument("--lock_protocol", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    if args.lock_protocol:
        protocol = lock_protocol(root, args.protocol)
        print("Locked protocol:", protocol["fingerprint"])
    else:
        report = evaluate(root, args.protocol, args.output)
        print("Saved", args.output, "—", report["status"])


if __name__ == "__main__":
    main()
