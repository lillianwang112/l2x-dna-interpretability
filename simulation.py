"""
Small evolutionary toy world for motif discovery and shortcut-learning experiments.

Generative story:
- K ancestor sequences of length L are sampled uniformly over {A,C,G,T}.
- A subset of ancestors (default 50%) each get exactly one embedded motif (default "AAAAA") at a random position.
- Descendants are created by mutating each ancestral base independently with rate mu.
- Labels are set by true motif presence after mutation (de novo motif creation counts as positive).

Running:
    python simulation.py
Outputs:
- data/dataset.json with two lists of sequences: positives and negatives.
- A printed summary of counts and hyperparameters.
"""

import json
import math
import os
import random
from typing import Dict, List, Tuple

# --- Hyperparameters (feel free to edit here) ---
RANDOM_SEED: int = 0
ALPHABET: str = "ACGT"
SEQUENCE_LENGTH: int = 100
MOTIF: str = "AAAAA"
NUM_FAMILIES: int = 100
DESCENDANTS_PER_FAMILY: int = 100
MUTATION_RATE: float = 0.1  # per-base probability of changing to a different base
MOTIF_ANCESTOR_FRACTION: float = 0.5  # fraction of ancestors that receive the motif

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.json")


def random_base_except(base: str) -> str:
    """Return a random base from ALPHABET that is not `base`."""
    choices = [b for b in ALPHABET if b != base]
    return random.choice(choices)


def mutate_sequence(seq: str, mu: float) -> str:
    """Independently mutate each position with probability mu."""
    mutated = []
    for base in seq:
        if random.random() < mu:
            mutated.append(random_base_except(base))
        else:
            mutated.append(base)
    return "".join(mutated)


def embed_motif(seq: str, motif: str, start: int) -> str:
    """Overwrite `seq` starting at `start` with `motif`."""
    end = start + len(motif)
    return seq[:start] + motif + seq[end:]


def has_motif(seq: str, motif: str) -> bool:
    """Check if motif appears contiguously anywhere in seq."""
    return motif in seq


def generate_ancestor(with_motif: bool) -> Tuple[str, int]:
    """Create one ancestor; optionally place the motif at a random start position."""
    ancestor = "".join(random.choice(ALPHABET) for _ in range(SEQUENCE_LENGTH))
    if with_motif:
        start = random.randint(0, SEQUENCE_LENGTH - len(MOTIF))
        ancestor = embed_motif(ancestor, MOTIF, start)
    else:
        start = -1  # indicates no motif embedded
    return ancestor, start


def simulate() -> Tuple[Dict[str, List[str]], int]:
    """Mentor-provided generator, extended to record family/sample IDs without new RNG draws."""
    random.seed(RANDOM_SEED)
    positives: List[str] = []
    negatives: List[str] = []
    positive_family_ids, negative_family_ids = [], []
    positive_sample_ids, negative_sample_ids = [], []

    num_motif_families = max(1, math.ceil(NUM_FAMILIES * MOTIF_ANCESTOR_FRACTION))
    motif_families = set(random.sample(range(NUM_FAMILIES), num_motif_families))

    for family_idx in range(NUM_FAMILIES):
        ancestor, motif_start = generate_ancestor(with_motif=family_idx in motif_families)

        for descendant_idx in range(DESCENDANTS_PER_FAMILY):
            descendant = mutate_sequence(ancestor, MUTATION_RATE)
            label = has_motif(descendant, MOTIF)

            if label:
                positives.append(descendant)
                positive_family_ids.append(family_idx)
                positive_sample_ids.append(f"family-{family_idx:03d}/descendant-{descendant_idx:03d}")
            else:
                negatives.append(descendant)
                negative_family_ids.append(family_idx)
                negative_sample_ids.append(f"family-{family_idx:03d}/descendant-{descendant_idx:03d}")

    return {
        "positive_sequences": positives, "negative_sequences": negatives,
        "positive_family_ids": positive_family_ids, "negative_family_ids": negative_family_ids,
        "positive_sample_ids": positive_sample_ids, "negative_sample_ids": negative_sample_ids,
    }, num_motif_families


def main() -> None:
    dataset, num_motif_families = simulate()

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    payload = {
        "hyperparameters": {
            "random_seed": RANDOM_SEED,
            "alphabet": ALPHABET,
            "sequence_length": SEQUENCE_LENGTH,
            "motif": MOTIF,
            "num_families": NUM_FAMILIES,
            "descendants_per_family": DESCENDANTS_PER_FAMILY,
            "mutation_rate": MUTATION_RATE,
            "motif_ancestor_fraction": MOTIF_ANCESTOR_FRACTION,
        },
        **dataset,
        "meta": {
            "num_motif_families": num_motif_families,
        },
    }

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    pos_count = len(dataset["positive_sequences"])
    neg_count = len(dataset["negative_sequences"])
    print(f"Wrote {DATA_PATH}")
    print(
        f"Positives: {pos_count} | Negatives: {neg_count} | "
        f"Total: {pos_count + neg_count}"
    )
    print(
        "Hyperparameters: "
        f"K={NUM_FAMILIES}, N={DESCENDANTS_PER_FAMILY}, mu={MUTATION_RATE}, "
        f"L={SEQUENCE_LENGTH}, motif={MOTIF}, motif_families={num_motif_families}"
    )


if __name__ == "__main__":
    main()
