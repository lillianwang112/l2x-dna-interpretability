"""Feature-budget experiments with explicit, descriptive teacher-agreement metrics."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Flatten
from keras.models import Model
from tqdm import tqdm

from dna_l2x import load_dataset_json, load_teacher, build_l2x_model
from validation_checks import select_topk, evaluate_selection
from data_contract import load_dataset, load_split, load_artifact, write_artifact


def _classes(probabilities, n):
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (n, 2) or not np.isfinite(probabilities).all():
        raise ValueError("Expected finite two-class probabilities with shape (examples, 2).")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Class probabilities must be nonnegative and sum to one.")
    return probabilities.argmax(axis=1)


def baseline_metrics(y_true, teacher):
    """Keep label accuracy separate from agreement with teacher predictions."""
    y_true = np.asarray(y_true)
    if y_true.ndim != 2 or y_true.shape[0] == 0:
        raise ValueError("Expected a nonempty array of true class probabilities.")
    truth = _classes(y_true, len(y_true))
    predicted = _classes(teacher, len(y_true))
    majority = int(np.bincount(predicted, minlength=2).argmax())
    return {
        "teacher_label_accuracy": float(np.mean(truth == predicted)),
        "teacher_majority_class": majority,
        "majority_teacher_class_agreement": float(np.mean(predicted == majority)),
        "perfect_teacher_agreement": 1.0,
    }


def get_approximator(model):
    """Reuse the existing approximator layers and weights, bypassing the selector.

    The current architecture has one Flatten layer immediately after masking.
    A different architecture requires an explicit adapter, never a fallback score.
    """
    flatten_layers = [layer for layer in model.layers if isinstance(layer, Flatten)]
    if len(flatten_layers) != 1:
        raise ValueError("Expected one Flatten entry point for the masked-input approximator.")
    try:
        approximator = Model(inputs=flatten_layers[0].input, outputs=model.output)
    except (ValueError, TypeError) as error:
        raise ValueError("Could not isolate the masked-input approximator.") from error
    if len(approximator.input_shape) != 3 or approximator.output_shape[1:] != (2,):
        raise ValueError("Expected a sequence-input, two-class approximator.")
    return approximator


def evaluate_masked_fidelity(model, selector, X, teacher, k, batch_size=128):
    """Compare exact-k keep/remove masks through the same learned approximator.

    Both masks use the same deterministic selection. These are descriptive tests
    of the approximator, not direct interventions on the original teacher.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or not len(X) or X.shape[2] != 4 or not np.isfinite(X).all():
        raise ValueError("Expected finite DNA inputs with shape (examples, positions, 4).")
    teacher_class = _classes(teacher, len(X))
    logits = np.asarray(selector.predict(X, batch_size=batch_size, verbose=0))
    if logits.shape != (len(X), X.shape[1], 1):
        raise ValueError("Selector output must have shape (examples, positions, 1).")
    selected = select_topk(logits[:, :, 0], k)
    keep_mask = np.zeros((len(X), X.shape[1], 1), dtype=np.float32)
    keep_mask[np.arange(len(X))[:, None], selected, 0] = 1.0
    approximator = get_approximator(model)
    if tuple(approximator.input_shape[1:]) != tuple(X.shape[1:]):
        raise ValueError("Approximator input shape does not match the supplied DNA inputs.")
    keep = approximator.predict(X * keep_mask, batch_size=batch_size, verbose=0)
    remove = approximator.predict(X * (1.0 - keep_mask), batch_size=batch_size, verbose=0)
    return {
        "keep_selected_teacher_agreement": float(np.mean(_classes(keep, len(X)) == teacher_class)),
        "remove_selected_teacher_agreement": float(np.mean(_classes(remove, len(X)) == teacher_class)),
    }


def summarize_runs(runs):
    if not runs:
        raise ValueError("No completed runs to summarize.")
    summary = []
    for k in sorted({run["k"] for run in runs}):
        group = [run for run in runs if run["k"] == k]
        row = {"k": k, "n_seeds": len(group)}
        for metric in ["keep_selected_teacher_agreement", "remove_selected_teacher_agreement"]:
            values = np.array([run[metric] for run in group], dtype=float)
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError("Agreement values must be finite and between zero and one.")
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        summary.append(row)
    return summary


def run_diagnostics(dataset_path, teacher_path, out_dir, max_k=10, n_seeds=3,
                    epochs=15, batch_size=128, split_path="data/family_split.json",
                    evaluation_partition="validation", show=False):
    dataset = load_dataset(dataset_path)
    split = load_split(split_path, dataset)
    if evaluation_partition not in {"validation", "test"}:
        raise ValueError("Evaluation partition must be validation or test.")
    sequences, X, y_true = load_dataset_json(dataset_path, seq_len=len(dataset.sequences[0]))
    teacher, teacher_metadata = load_artifact(teacher_path, "teacher_predictions", dataset, split)
    _classes(teacher, len(X))
    if len(X) < 2 or not 1 <= max_k <= X.shape[1]:
        raise ValueError("Need at least two examples and 1 <= max_k <= sequence length.")
    if min(n_seeds, epochs, batch_size) < 1:
        raise ValueError("n_seeds, epochs, and batch_size must be positive.")
    train_indices, val_indices = split["train"], split["validation"]
    evaluation_indices = split[evaluation_partition]
    X_train, T_train = X[train_indices], teacher[train_indices]
    X_val, T_val = X[val_indices], teacher[val_indices]
    X_eval, T_eval = X[evaluation_indices], teacher[evaluation_indices]
    baselines = baseline_metrics(y_true[evaluation_indices], T_eval)
    output = Path(out_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Diagnostic output is not empty; choose a new directory to preserve prior runs.")
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for k in tqdm(range(1, max_k + 1), desc="Feature budgets"):
        for seed in range(n_seeds):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            try:
                model, selector = build_l2x_model(seq_len=X.shape[1], k=k, tau=0.5, lr=1e-3)
                history = model.fit(X_train, T_train, validation_data=(X_val, T_val),
                                    epochs=epochs, batch_size=batch_size, verbose=0)
                metrics = evaluate_masked_fidelity(model, selector, X_eval, T_eval, k, batch_size)
                scores = selector.predict(X, batch_size=batch_size, verbose=0).squeeze(-1)
                run_dir = output / f"k{k}_seed{seed}"
                score_fingerprint = write_artifact(run_dir / "selection_scores.npz", "selection_scores",
                                                   scores, dataset, split, seed=seed,
                                                   teacher_fingerprint=teacher_metadata["fingerprint"])
                model.save_weights(run_dir / "l2x.weights.h5")
                selection_metrics = evaluate_selection([dataset.sequences[i] for i in evaluation_indices],
                                                       dataset.labels[evaluation_indices], scores[evaluation_indices],
                                                       k=k, motif=dataset.motif, seed=seed)
            except Exception as error:
                raise RuntimeError(f"Diagnostic run failed for k={k}, seed={seed}; no fallback metric recorded.") from error
            runs.append({
                "k": k, "seed": seed, **metrics,
                "selection_metrics": selection_metrics, "selection_fingerprint": score_fingerprint,
                "selection_scores_file": str(run_dir.relative_to(output) / "selection_scores.npz"),
                "history": {name: [float(v) for v in values] for name, values in history.history.items()},
            })
            (run_dir / "run_metrics.json").write_text(json.dumps(runs[-1], indent=2, allow_nan=False) + "\n")
            print(f"k={k}, seed={seed}: keep={metrics['keep_selected_teacher_agreement']:.4f}, "
                  f"remove={metrics['remove_selected_teacher_agreement']:.4f}", flush=True)
    summary = summarize_runs(runs)
    report = {
        "schema_version": 1,
        "inputs": {"dataset": str(dataset_path), "teacher": str(teacher_path)},
        "parameters": {"max_k": max_k, "n_seeds": n_seeds, "epochs": epochs,
                       "batch_size": batch_size},
        "dataset_fingerprint": dataset.fingerprint, "split_fingerprint": split.fingerprint,
        "teacher_fingerprint": teacher_metadata["fingerprint"],
        "evaluation_population": evaluation_partition + " ancestor families",
        "evaluation_indices": evaluation_indices.tolist(),
        "test_metrics_evaluated": evaluation_partition == "test",
        "train_indices": train_indices.tolist(), "validation_indices": val_indices.tolist(),
        "test_indices": split["test"].tolist(),
        "tie_rule": "lower position index first",
        "masking": "exact-k keep/remove masks evaluated by the same learned approximator",
        "baselines": baselines, "runs": runs, "summary": summary,
    }
    # Save raw per-seed metrics and histories so figures can be rebuilt without retraining.
    (output / "fidelity_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    ks = [row["k"] for row in summary]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, [r["keep_selected_teacher_agreement_mean"] for r in summary], "o-", label="Keep selected bases")
    ax.plot(ks, [r["remove_selected_teacher_agreement_mean"] for r in summary], "x-", label="Remove selected bases")
    ax.axhline(1.0, color="gray", linestyle="--", label="Perfect teacher agreement")
    ax.axhline(baselines["majority_teacher_class_agreement"], color="gray", linestyle=":",
               label="Majority-teacher-class baseline")
    ax.set(xlabel="k (selected positions)", ylabel="Class agreement with teacher",
           title="Approximator agreement under keep/remove masks", xticks=ks, ylim=(0, 1.03))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "diagnostics_fidelity.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_json", default="data/dataset.json")
    parser.add_argument("--teacher_npz", default="outputs/teacher/teacher_predictions.npz")
    parser.add_argument("--split_json", default="data/family_split.json")
    parser.add_argument("--out_dir", default="outputs/fidelity")
    parser.add_argument("--max_k", type=int, default=10)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--evaluation_partition", choices=["validation", "test"], default="validation")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    run_diagnostics(args.dataset_json, args.teacher_npz, args.out_dir, args.max_k,
                    args.n_seeds, args.epochs, args.batch_size, args.split_json, args.evaluation_partition, args.show)


if __name__ == "__main__":
    main()
