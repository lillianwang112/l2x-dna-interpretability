import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import (
    Layer, Input, Conv1D, Dense, Flatten, GlobalMaxPooling1D, Multiply, Reshape, Lambda
)
from keras.optimizers import Adam
from data_contract import load_dataset, load_split, load_artifact, write_artifact
from validation_checks import select_topk, evaluate_selection



# Config

ALPHABET = "ACGT"
MAP = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
}
MOTIF = "AAAAA"


def load_dataset_json(path: str, seq_len: int = 100) -> Tuple[List[str], np.ndarray, np.ndarray]:
    dataset = load_dataset(path)
    if any(len(s) != seq_len for s in dataset.sequences):
        raise ValueError("Sequence length does not match seq_len; no silent padding or truncation is performed.")
    X = np.asarray([[MAP[ch] for ch in seq] for seq in dataset.sequences], dtype=np.float32)
    y_true = np.eye(2, dtype=np.float32)[dataset.labels]
    return dataset.sequences, X, y_true


def load_teacher(path: str, n: int, *, dataset=None, split=None) -> np.ndarray:
    if dataset is None or split is None:
        raise ValueError("Teacher loading requires the verified dataset and shared split.")
    teacher, _ = load_artifact(path, "teacher_predictions", dataset, split)
    if len(teacher) != n:
        raise ValueError("Teacher row count does not match the DNA input.")
    return teacher


def motif_windows(seq: str, motif: str = MOTIF):
    m = len(motif)
    out = []
    for start in range(0, len(seq) - m + 1):
        if seq[start:start + m] == motif:
            out.append((start, start + m - 1))
    return out


class SampleConcrete(Layer):
    def __init__(self, k: int, tau0: float, **kwargs):
        super().__init__(**kwargs)
        self.k = int(k)
        self.tau0 = float(tau0)

    def call(self, logits, training=None):
        # logits: (batch, L, 1)
        logits_ = tf.transpose(logits, perm=[0, 2, 1])  # (batch, 1, L)
        L = tf.shape(logits_)[2]

        # soft sample (train)
        uniform = tf.random.uniform(
            shape=(tf.shape(logits_)[0], self.k, L),
            minval=1e-10,
            maxval=1.0,
            dtype=logits.dtype,
        )
        gumbel = -tf.math.log(-tf.math.log(uniform))
        noisy_logits = (gumbel + logits_) / self.tau0
        samples = tf.nn.softmax(noisy_logits, axis=-1)   # (batch, k, L)
        samples = tf.reduce_max(samples, axis=1)         # (batch, L)
        samples = tf.expand_dims(samples, axis=-1)       # (batch, L, 1)

        # hard top-k (eval)
        logits_flat = tf.squeeze(logits, axis=-1)        # (batch, L)
        topk_vals = tf.math.top_k(logits_flat, k=self.k, sorted=False).values
        thresh = tf.reduce_min(topk_vals, axis=-1, keepdims=True)
        hard = tf.cast(logits_flat >= thresh, tf.float32)
        hard = tf.expand_dims(hard, axis=-1)             # (batch, L, 1)

        if training is None:
            training = False
        training = tf.cast(training, tf.bool)

        return tf.cond(training, lambda: samples, lambda: hard)

    def compute_output_shape(self, input_shape):
        return input_shape


def build_l2x_model(seq_len: int, k: int, tau: float, lr: float):
    x_in = Input(shape=(seq_len, 4), name="dna_input")

    # Fully Convolutional Selector
    h = Conv1D(32, 7, padding="same", activation="relu")(x_in)
    h = Conv1D(32, 7, padding="same", activation="relu")(h)
    
    # Directly output (Batch, L, 1) logits without squashing
    logits_T = Conv1D(1, 1, padding="same", activation=None)(h)


    mask = SampleConcrete(k=k, tau0=tau, name="sample_concrete")(logits_T)
    mask4 = Lambda(lambda m: tf.repeat(m, repeats=4, axis=2), name='mask_repeat4')(mask)
    x_masked = Multiply()([x_in, mask4])

    # Approximator Q(Y | X_S)
    q = Flatten()(x_masked)
    q = Dense(256, activation="relu")(q)
    q = Dense(128, activation="relu")(q)
    y_out = Dense(2, activation="softmax", name="preds")(q)

    model = Model(inputs=x_in, outputs=y_out, name="l2x_dna")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    selector_only = Model(inputs=x_in, outputs=logits_T, name="selector_only")
    return model, selector_only
# Visualization Helpers 
def print_colored_genome(seq, top_indices):
    """Prints DNA with selected bases highlighted in Green."""
    GREEN = '\033[92m'
    RESET = '\033[0m'
    out = ""
    for i, char in enumerate(seq):
        if i in top_indices:
            out += f"{GREEN}[{char}]{RESET}"
        else:
            out += char.lower() # Background in lowercase
    print(out)

def print_ascii_histogram(offsets):
    """Prints a text-based histogram of selection offsets."""
    if not offsets:
        print("No positive sequences found to histogram.")
        return
    
    # We look at a window around the motif start
    bins = np.arange(-10, 16) # From -10 to +15 relative to start
    hist, _ = np.histogram(offsets, bins=bins)
    
    max_h = max(hist) if len(hist) > 0 else 1
    
    print(f"\n{'Offset':<6} | {'Count':<6} | Distribution")
    print("-" * 40)
    for i, count in enumerate(hist):
        offset = bins[i]
        bar_len = int((count / max_h) * 40)
        bar = "#" * bar_len
        
        # Label the motif region
        label = ""
        if 0 <= offset < 5:
            label = " <--- MOTIF"
            
        print(f"{offset:<6} | {count:<6} | {bar}{label}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit DNA L2X using verified teacher predictions and a shared family split.")
    parser.add_argument("--dataset_json", default="data/dataset.json")
    parser.add_argument("--split_json", default="data/family_split.json")
    parser.add_argument("--teacher_npz", default="outputs/teacher/teacher_predictions.npz")
    parser.add_argument("--out_dir", default="outputs/l2x")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluation_partition", choices=["validation", "test"], default="validation")
    args = parser.parse_args(argv)
    dataset = load_dataset(args.dataset_json)
    split = load_split(args.split_json, dataset)
    teacher, teacher_metadata = load_artifact(args.teacher_npz, "teacher_predictions", dataset, split)
    _, X, _ = load_dataset_json(args.dataset_json, len(dataset.sequences[0]))
    if not 1 <= args.k <= X.shape[1] or min(args.epochs, args.batch_size) < 1:
        parser.error("Require 1 <= k <= sequence length and positive epochs/batch_size.")
    if not np.isfinite(args.tau) or args.tau <= 0 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("Temperature and learning rate must be positive and finite.")
    output = os.path.join(args.out_dir, "selection_scores.npz")
    if os.path.exists(output):
        raise FileExistsError("L2X output exists; choose a new output directory to preserve it.")
    os.makedirs(args.out_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    model, selector = build_l2x_model(seq_len=X.shape[1], k=args.k, tau=args.tau, lr=args.lr)
    train_idx, val_idx = split["train"], split["validation"]
    history = model.fit(X[train_idx], teacher[train_idx], validation_data=(X[val_idx], teacher[val_idx]),
                        epochs=args.epochs, batch_size=args.batch_size, verbose=1)
    scores = selector.predict(X, batch_size=args.batch_size, verbose=0).squeeze(-1)
    selection_fingerprint = write_artifact(output, "selection_scores", scores, dataset, split, seed=args.seed,
                                         teacher_fingerprint=teacher_metadata["fingerprint"])
    model.save_weights(os.path.join(args.out_dir, "l2x.weights.h5"))
    indices = split[args.evaluation_partition]
    metrics = evaluate_selection([dataset.sequences[i] for i in indices], dataset.labels[indices],
                                 scores[indices], args.k, dataset.motif, seed=args.seed)
    summary = {"dataset_fingerprint": dataset.fingerprint, "split_fingerprint": split.fingerprint,
               "teacher_fingerprint": teacher_metadata["fingerprint"],
               "selection_fingerprint": selection_fingerprint,
               "evaluation_population": args.evaluation_partition + " ancestor families",
               "evaluation_indices": indices.tolist(), "seed": args.seed, "k": args.k,
               "history": {key: [float(v) for v in values] for key, values in history.history.items()},
               "selection_metrics": metrics}
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    for i in [i for i in indices if dataset.labels[i] == 1][:3]:
        print(f"Sample {dataset.sample_ids[i]}")
        print_colored_genome(dataset.sequences[i], select_topk(scores[i:i + 1], args.k)[0])
    print("Saved identity-bound scores and metrics for", summary["evaluation_population"])


if __name__ == "__main__":
    main()
