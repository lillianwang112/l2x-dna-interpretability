"""Small synthetic fixtures, separate from the research dataset."""
import json
from pathlib import Path
from data_contract import load_dataset, make_split


def make_fixture(root, families=6):
    root = Path(root)
    payload = {"hyperparameters": {"motif": "AAAAA", "sequence_length": 20}}
    for label, prefix in [(1, 'positive'), (0, 'negative')]:
        sequences = []
        for family in range(families):
            tag = ''.join('CGT'[(family // (3 ** j)) % 3] for j in range(4))
            sequences.append(('AAAAA' + 'C' * 11 + tag) if label else ('C' * 5 + 'G' * 11 + tag))
        payload[prefix + '_sequences'] = sequences
        payload[prefix + '_family_ids'] = list(range(families))
        payload[prefix + '_sample_ids'] = [f'f{f}/{prefix}' for f in range(families)]
    dataset_path = root / 'dataset.json'
    dataset_path.write_text(json.dumps(payload))
    dataset = load_dataset(dataset_path)
    split = make_split(dataset)
    split_path = root / 'split.json'
    split_path.write_text(json.dumps(split.payload))
    return dataset_path, split_path, dataset, split
