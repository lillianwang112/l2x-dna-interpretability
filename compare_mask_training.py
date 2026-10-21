"""Compare soft/hard approximator training with a fixed teacher and selectors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_contract import load_artifact, load_dataset, load_split
from dna_l2x import build_l2x_model, load_dataset_json
from run_diagnostics import _classes, get_approximator
from validation_checks import select_topk


def array_hash(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def exact_mask(logits, k):
    selected = select_topk(np.asarray(logits).squeeze(-1), k)
    mask = np.zeros_like(logits, dtype=np.float32)
    mask[np.arange(len(mask))[:, None], selected, 0] = 1
    return mask


def infer(model, x, batch_size):
    return np.concatenate([np.asarray(model(x[i:i + batch_size], training=False))
                           for i in range(0, len(x), batch_size)])


def prediction_metrics(target, prediction, labels):
    teacher_class = _classes(target, len(labels))
    predicted = _classes(prediction, len(labels))
    p = np.asarray(target, dtype=np.float64)
    q = np.asarray(prediction, dtype=np.float64)
    confusion = np.zeros((2, 2), dtype=int)
    np.add.at(confusion, (teacher_class, predicted), 1)
    per_class = [float(confusion[c, c] / confusion[c].sum()) if confusion[c].sum() else None
                 for c in range(2)]
    return {
        "teacher_agreement": float(np.mean(teacher_class == predicted)),
        "teacher_negative_agreement": per_class[0], "teacher_positive_agreement": per_class[1],
        "balanced_teacher_agreement": float(np.mean(per_class)) if None not in per_class else None,
        "teacher_vs_approximator_confusion": confusion.tolist(),
        "true_label_accuracy": float(np.mean(np.asarray(labels) == predicted)),
        "teacher_cross_entropy": float(-np.mean(np.sum(p * np.log(np.clip(q, 1e-7, 1)), axis=1))),
        "teacher_kl": float(np.mean(np.sum(p * (np.log(np.clip(p, 1e-7, 1)) -
                                                     np.log(np.clip(q, 1e-7, 1))), axis=1))),
        "positive_probability_mae": float(np.mean(np.abs(p[:, 1] - q[:, 1]))),
    }


def epoch_order(n_rows, seed, epoch):
    return np.random.default_rng(np.random.SeedSequence([81000, seed, epoch])).permutation(n_rows)


def train_branch(approximator, sampler, x, target, logits, *, mode, k,
                 epochs, batch_size, learning_rate, seed, on_epoch=None):
    """Only update Q; the selector logits are detached, fixed arrays in both arms."""
    if mode not in {"soft", "hard"}:
        raise ValueError("Training mode must be soft or hard.")
    if min(epochs, batch_size) < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Require positive epochs, batch size, and learning rate.")
    tf.random.set_seed(91000 + seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    hard = exact_mask(logits, k)
    orders = []
    history = []

    @tf.function(reduce_retracing=True)
    def step(masked, teacher):
        with tf.GradientTape() as tape:
            prediction = approximator(masked, training=True)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(teacher, prediction))
        gradients = tape.gradient(loss, approximator.trainable_variables)
        if any(g is None for g in gradients):
            raise ValueError("Missing approximator gradient.")
        optimizer.apply_gradients(zip(gradients, approximator.trainable_variables))
        agreement = tf.reduce_sum(tf.cast(tf.argmax(prediction, 1) == tf.argmax(teacher, 1), tf.float32))
        return loss, agreement

    for epoch in range(epochs):
        order = epoch_order(len(x), seed, epoch)
        orders.append(array_hash([order]))
        loss_sum, correct = 0., 0.
        for start in range(0, len(x), batch_size):
            rows = order[start:start + batch_size]
            mask = (np.asarray(sampler(logits[rows], training=True)) if mode == "soft" else hard[rows])
            loss, agreement = step(x[rows] * mask, target[rows])
            loss_sum += float(loss) * len(rows)
            correct += float(agreement)
        row = {"epoch": epoch + 1, "train_cross_entropy": loss_sum / len(x),
               "train_teacher_agreement": correct / len(x)}
        if on_epoch is not None:
            row["validation_exact_k"] = on_epoch()
        history.append(row)
    return {"history": history, "batch_order_hashes": orders,
            "optimizer_steps": int(optimizer.iterations.numpy())}


def compare(dataset_path, split_path, teacher_path, checkpoint_dir, output_dir, *,
            k=5, seeds=(0, 1, 2), epochs=15, batch_size=128, learning_rate=.001, soft_draws=10):
    if min(epochs, batch_size, soft_draws) < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid training/evaluation budget.")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers.")
    dataset = load_dataset(dataset_path)
    split = load_split(split_path, dataset)
    teacher, teacher_meta = load_artifact(teacher_path, "teacher_predictions", dataset, split)
    _, x, _ = load_dataset_json(str(dataset_path), len(dataset.sequences[0]))
    if not 1 <= k <= x.shape[1]:
        raise ValueError("Invalid k.")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Comparison directory is not empty; preserve existing results.")
    checkpoints = Path(checkpoint_dir)
    # Verify every source artifact before starting either arm.
    sources = {}
    for seed in seeds:
        folder = checkpoints / f"k{k}_seed{seed}"
        scores, meta = load_artifact(folder / "selection_scores.npz", "selection_scores", dataset, split)
        if meta["teacher_fingerprint"] != teacher_meta["fingerprint"] or meta["seed"] != seed:
            raise ValueError("Checkpoint scores do not match this teacher/seed.")
        sources[seed] = {"scores": scores, "selection_fingerprint": meta["fingerprint"],
                         "checkpoint_sha256": hashlib.sha256((folder / "l2x.weights.h5").read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    train, val = split["train"], split["validation"]
    config = {"k": k, "seeds": list(seeds), "epochs": epochs, "batch_size": batch_size,
              "learning_rate": learning_rate, "temperature": .5, "soft_evaluation_draws": soft_draws,
              "modes": ["soft", "hard"], "initialization": "same saved Q weights; fresh Adam in each arm",
              "frozen_components": ["teacher", "selector"], "evaluation_partition": "validation",
              "checkpoint_selection": "final epoch only", "test_metrics_evaluated": False,
              "batch_order_seed_rule": "SeedSequence([81000, checkpoint_seed, zero_based_epoch])",
              "training_mask_seed_rule": "91000 + checkpoint_seed",
              "evaluation_mask_seed_rule": "101000 + 100*checkpoint_seed + draw_index",
              "dataset_fingerprint": dataset.fingerprint, "split_fingerprint": split.fingerprint,
              "teacher_fingerprint": teacher_meta["fingerprint"],
              "train_indices": train.tolist(), "validation_indices": val.tolist(),
              "sources": {str(s): {key: value for key, value in source.items() if key != "scores"}
                          for s, source in sources.items()},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    report = {"configuration": config, "runs": []}
    for seed in seeds:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model, selector = build_l2x_model(x.shape[1], k, .5, learning_rate)
        model.optimizer.build(model.trainable_variables)
        checkpoint = checkpoints / f"k{k}_seed{seed}" / "l2x.weights.h5"
        model.load_weights(checkpoint)
        initial_weights = [v.copy() for v in model.get_weights()]
        selector_hash = array_hash(selector.get_weights())
        q = get_approximator(model)
        q_initial_hash = array_hash(q.get_weights())
        # Inference only on training and validation rows; no test metric computation.
        logits_train = infer(selector, x[train], batch_size)
        logits_val = infer(selector, x[val], batch_size)
        for idx, logits in [(train, logits_train), (val, logits_val)]:
            if not np.allclose(logits[:, :, 0], sources[seed]["scores"][idx], atol=1e-6, rtol=0):
                raise ValueError("Checkpoint does not reproduce its verified selector artifact.")
        hard_val = exact_mask(logits_val, k)
        baseline = prediction_metrics(teacher[val], infer(q, x[val] * hard_val, batch_size), dataset.labels[val])
        order_hashes = None
        for mode in ["soft", "hard"]:
            model.set_weights(initial_weights)
            if array_hash(q.get_weights()) != q_initial_hash:
                raise ValueError("Paired arms must start with identical approximator weights.")
            evaluate_hard = lambda: prediction_metrics(teacher[val], infer(q, x[val] * hard_val, batch_size), dataset.labels[val])
            trained = train_branch(q, model.get_layer("sample_concrete"), x[train], teacher[train], logits_train,
                                   mode=mode, k=k, epochs=epochs, batch_size=batch_size,
                                   learning_rate=learning_rate, seed=seed, on_epoch=evaluate_hard)
            if order_hashes is None:
                order_hashes = trained["batch_order_hashes"]
            if order_hashes != trained["batch_order_hashes"] or array_hash(selector.get_weights()) != selector_hash:
                raise ValueError("Batch pairing or frozen-selector invariant failed.")
            soft_metrics = []
            for draw in range(soft_draws):
                tf.random.set_seed(101000 + 100*seed + draw)
                mask = np.asarray(model.get_layer("sample_concrete")(logits_val, training=True))
                soft_metrics.append(prediction_metrics(teacher[val], infer(q, x[val] * mask, batch_size), dataset.labels[val]))
            folder = output / f"seed{seed}_{mode}"
            folder.mkdir()
            q.save_weights(folder / "approximator.weights.h5")
            row = {"seed": seed, "mode": mode, "baseline_exact_k": baseline,
                   "final_exact_k": evaluate_hard(), "final_soft_draws": soft_metrics,
                   "initial_approximator_hash": q_initial_hash,
                   "final_approximator_hash": array_hash(q.get_weights()),
                   "frozen_selector_hash": selector_hash, "selector_unchanged": True, **trained}
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != sources[seed]["checkpoint_sha256"]:
                raise ValueError("Source checkpoint changed.")
            (folder / "metrics.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
            report["runs"].append(row)
            (output / "comparison_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"seed": seed, "mode": mode, "final_exact_k": row["final_exact_k"]}), flush=True)
    report["status"] = "complete"
    (output / "comparison_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_json", default="data/dataset.json")
    parser.add_argument("--split_json", default="data/family_split.json")
    parser.add_argument("--teacher_npz", default="outputs/teacher/teacher_predictions.npz")
    parser.add_argument("--checkpoint_dir", default="outputs/fidelity")
    parser.add_argument("--out_dir", default="outputs/mask_comparison")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--soft_draws", type=int, default=10)
    args = parser.parse_args()
    compare(args.dataset_json, args.split_json, args.teacher_npz, args.checkpoint_dir, args.out_dir,
            k=args.k, seeds=args.seeds, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.lr, soft_draws=args.soft_draws)


if __name__ == "__main__":
    main()
