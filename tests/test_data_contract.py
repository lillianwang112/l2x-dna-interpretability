"""Dataset/split identity and teacher-export checks with known small fixtures."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from data_contract import load_dataset, make_split, load_split, write_artifact, load_artifact, _digest
from dna_l2x import load_dataset_json, load_teacher, main as l2x_main
from run_advisor_training import predict_rows, export_predictions, run_teacher, RowDataset
from run_diagnostics import run_diagnostics
from train import ConvClassifier, one_hot_encode
from fixtures import make_fixture
import simulation


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_context = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_context.name)
        self.dataset_path, self.split_path, self.dataset, self.split = make_fixture(self.root)
        self.probabilities = np.eye(2, dtype=np.float32)[self.dataset.labels]
        self.teacher_path = self.root / 'teacher.npz'
        self.teacher_fp = write_artifact(self.teacher_path, 'teacher_predictions', self.probabilities,
                                         self.dataset, self.split, seed=0)

    def tearDown(self):
        self.temp_context.cleanup()

    def tamper_artifact(self, **changes):
        with np.load(self.teacher_path, allow_pickle=False) as artifact:
            fields = {name: artifact[name] for name in artifact.files}
        fields.update(changes)
        output = self.root / 'tampered.npz'
        np.savez_compressed(output, **fields)
        return output

    def test_families_rows_and_exact_sequences_are_disjoint(self):
        self.assertEqual(sum(len(v) for v in self.split.indices.values()), len(self.dataset))
        groups = [set(self.dataset.family_ids[idx]) for idx in self.split.indices.values()]
        for i in range(3):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])
        self.assertEqual(make_split(self.dataset, seed=42).fingerprint, self.split.fingerprint)
        self.assertNotEqual(make_split(self.dataset, seed=43).fingerprint, self.split.fingerprint)
        self.assertEqual(load_split(self.split_path, self.dataset).fingerprint, self.split.fingerprint)

    def test_dataset_row_reordering_invalidates_split(self):
        payload = json.loads(self.dataset_path.read_text())
        for key in ['positive_sequences', 'positive_family_ids', 'positive_sample_ids']:
            payload[key][0], payload[key][1] = payload[key][1], payload[key][0]
        self.dataset_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'ordered dataset'):
            load_split(self.split_path, load_dataset(self.dataset_path))

    def test_changed_sequence_invalidates_split(self):
        payload = json.loads(self.dataset_path.read_text())
        sequence = payload['positive_sequences'][0]
        payload['positive_sequences'][0] = sequence[:8] + 'G' + sequence[9:]
        self.dataset_path.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            load_split(self.split_path, load_dataset(self.dataset_path))

    def test_missing_or_duplicate_identity_is_rejected(self):
        payload = json.loads(self.dataset_path.read_text())
        del payload['positive_family_ids']
        self.dataset_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'family and sample IDs'):
            load_dataset(self.dataset_path)
        self.dataset_path, _, _, _ = make_fixture(self.root)
        payload = json.loads(self.dataset_path.read_text())
        payload['positive_sample_ids'][1] = payload['positive_sample_ids'][0]
        self.dataset_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'unique'):
            load_dataset(self.dataset_path)

    def test_family_leak_is_rejected_even_with_a_recomputed_checksum(self):
        payload = json.loads(self.split_path.read_text())
        a = payload['partitions']['train'][0]
        b = payload['partitions']['validation'][0]
        payload['partitions']['train'][0] = b
        payload['partitions']['validation'][0] = a
        payload['fingerprint'] = _digest({k: v for k, v in payload.items() if k != 'fingerprint'})
        self.split_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'family'):
            load_split(self.split_path, self.dataset)

    def test_duplicate_and_missing_split_rows_are_rejected(self):
        for change in ['duplicate', 'missing']:
            payload = json.loads(json.dumps(self.split.payload))
            if change == 'duplicate':
                payload['partitions']['train'].append(payload['partitions']['train'][0])
            else:
                payload['partitions']['train'].pop()
            payload['fingerprint'] = _digest({k: v for k, v in payload.items() if k != 'fingerprint'})
            self.split_path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_split(self.split_path, self.dataset)

    def test_teacher_roundtrip_preserves_values_and_identity(self):
        values, metadata = load_artifact(self.teacher_path, 'teacher_predictions', self.dataset, self.split)
        np.testing.assert_array_equal(values, self.probabilities)
        self.assertEqual(metadata['fingerprint'], self.teacher_fp)
        self.assertEqual(metadata['train_indices'], self.split['train'].tolist())
        np.testing.assert_array_equal(load_teacher(self.teacher_path, len(self.dataset),
                                                 dataset=self.dataset, split=self.split), self.probabilities)

    def test_reordered_ids_and_values_are_rejected(self):
        path = self.tamper_artifact(values=self.probabilities[::-1], sample_ids=self.dataset.sample_ids[::-1])
        with self.assertRaisesRegex(ValueError, 'sample IDs'):
            load_artifact(path, 'teacher_predictions', self.dataset, self.split)

    def test_reordered_values_alone_are_detected_by_checksum(self):
        path = self.tamper_artifact(values=self.probabilities[::-1])
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_artifact(path, 'teacher_predictions', self.dataset, self.split)

    def test_artifact_from_different_split_is_rejected(self):
        other = make_split(self.dataset, seed=43)
        with self.assertRaisesRegex(ValueError, 'split fingerprint'):
            load_artifact(self.teacher_path, 'teacher_predictions', self.dataset, other)

    def test_wrong_training_rows_are_rejected(self):
        with np.load(self.teacher_path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata'].item()))
        metadata['train_indices'] = self.split['validation'].tolist()
        path = self.tamper_artifact(metadata=np.array(json.dumps(metadata)))
        with self.assertRaisesRegex(ValueError, 'training rows'):
            load_artifact(path, 'teacher_predictions', self.dataset, self.split)

    def test_legacy_unbound_arrays_and_invalid_probabilities_are_rejected(self):
        legacy = self.root / 'legacy.npy'
        np.save(legacy, self.probabilities)
        with self.assertRaisesRegex(ValueError, 'Unverified array'):
            load_artifact(legacy, 'teacher_predictions', self.dataset, self.split)
        with self.assertRaisesRegex(ValueError, 'verified dataset'):
            load_teacher(self.teacher_path, len(self.dataset))
        for probabilities in [np.full((len(self.dataset), 2), np.nan), np.ones((len(self.dataset), 2))]:
            with self.assertRaises(ValueError):
                write_artifact(self.root / 'invalid.npz', 'teacher_predictions', probabilities,
                               self.dataset, self.split, seed=0)

    def test_selection_scores_keep_teacher_provenance(self):
        scores = np.ones((len(self.dataset), 20))
        path = self.root / 'selection.npz'
        write_artifact(path, 'selection_scores', scores, self.dataset, self.split,
                       seed=1, teacher_fingerprint=self.teacher_fp)
        values, metadata = load_artifact(path, 'selection_scores', self.dataset, self.split)
        self.assertEqual(metadata['teacher_fingerprint'], self.teacher_fp)
        np.testing.assert_array_equal(values, scores)
        with self.assertRaises(ValueError):
            write_artifact(path, 'selection_scores', scores, self.dataset, self.split, seed=0)

    def test_encoding_preserves_canonical_row_order(self):
        sequences, X, y = load_dataset_json(self.dataset_path, seq_len=20)
        self.assertEqual(sequences, self.dataset.sequences)
        np.testing.assert_array_equal(y.argmax(axis=1), self.dataset.labels)
        np.testing.assert_array_equal(X[0].T, one_hot_encode(sequences[0]).numpy())
        with self.assertRaisesRegex(ValueError, 'no silent padding'):
            load_dataset_json(self.dataset_path, seq_len=100)

    def test_teacher_prediction_order_has_known_independent_answers(self):
        torch.manual_seed(0)
        model = ConvClassifier(seq_len=20)
        canonical = predict_rows(model, self.dataset, np.arange(len(self.dataset)), batch_size=4)
        order = np.arange(len(self.dataset))[::-1]
        reversed_predictions = predict_rows(model, self.dataset, order, batch_size=3)
        np.testing.assert_allclose(reversed_predictions, canonical[order], rtol=1e-5, atol=1e-6)
        expected = []
        model.eval()
        with torch.no_grad():
            for sequence in self.dataset.sequences:
                expected.append(torch.sigmoid(model(one_hot_encode(sequence).unsqueeze(0))).item())
        np.testing.assert_allclose(canonical[:, 1], expected, rtol=1e-5, atol=1e-6)
        rows = RowDataset(self.dataset, self.split['train'])
        ids = [str(rows[i][2]) for i in range(len(rows))]
        self.assertEqual(ids, self.dataset.sample_ids[self.split['train']].tolist())
        self.assertFalse(set(ids) & set(self.dataset.sample_ids[self.split['test']]))

    def test_export_checks_all_rows_in_an_independent_inference_order(self):
        torch.manual_seed(0)
        model = ConvClassifier(seq_len=20)
        path = self.root / 'checked_teacher.npz'
        values, report = export_predictions(model, self.dataset, self.split, path, batch_size=4)
        self.assertEqual(report['alignment_check']['rows_verified'], len(self.dataset))
        loaded, _ = load_artifact(path, 'teacher_predictions', self.dataset, self.split)
        np.testing.assert_array_equal(values, loaded)

    def test_export_refuses_a_failed_independent_alignment_check(self):
        model = ConvClassifier(seq_len=20)
        with patch('run_advisor_training.predict_rows', return_value=self.probabilities):
            with self.assertRaisesRegex(ValueError, 'reverse-order inference'):
                export_predictions(model, self.dataset, self.split, self.root / 'must_not_exist.npz')
        self.assertFalse((self.root / 'must_not_exist.npz').exists())


class GeneratorRegressionTests(unittest.TestCase):
    def test_saved_research_rows_are_exactly_reproduced_with_family_ids(self):
        payload = json.loads((Path(__file__).resolve().parents[1] / 'data/dataset.json').read_text())
        replayed, _ = simulation.simulate()
        for name in ['positive_sequences', 'negative_sequences', 'positive_family_ids',
                     'negative_family_ids', 'positive_sample_ids', 'negative_sample_ids']:
            self.assertEqual(payload[name], replayed[name])


class SmallPipelineIntegrationTests(unittest.TestCase):
    def test_teacher_l2x_and_diagnostics_share_partition_without_evaluating_test_metrics(self):
        # Twelve synthetic fixture rows, one epoch per model; not the research run.
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            dataset_path, split_path, dataset, split = make_fixture(root)
            teacher_report = run_teacher(dataset_path, split_path, root / 'teacher', epochs=1, batch_size=4)
            teacher_path = root / 'teacher' / 'teacher_predictions.npz'
            self.assertFalse(teacher_report['test_metrics_evaluated'])
            l2x_main(['--dataset_json', str(dataset_path), '--split_json', str(split_path),
                      '--teacher_npz', str(teacher_path), '--out_dir', str(root / 'l2x'),
                      '--epochs', '1', '--batch_size', '4', '--k', '1'])
            summary = json.loads((root / 'l2x' / 'summary.json').read_text())
            self.assertEqual(summary['evaluation_population'], 'validation ancestor families')
            self.assertEqual(summary['evaluation_indices'], split['validation'].tolist())
            self.assertEqual(summary['split_fingerprint'], teacher_report['split_fingerprint'])
            report = run_diagnostics(dataset_path, teacher_path, root / 'diagnostics',
                                     max_k=1, n_seeds=1, epochs=1, batch_size=4, split_path=split_path)
            self.assertEqual(report['evaluation_indices'], split['validation'].tolist())
            self.assertFalse(report['test_metrics_evaluated'])
            self.assertEqual(report['split_fingerprint'], split.fingerprint)
            self.assertEqual(report['runs'][0]['selection_metrics']['n_sequences'], len(split['validation']))
            artifact = root / 'diagnostics' / report['runs'][0]['selection_scores_file']
            values, metadata = load_artifact(artifact, 'selection_scores', dataset, split)
            self.assertEqual(metadata['teacher_fingerprint'], teacher_report['prediction_fingerprint'])
            self.assertEqual(values.shape, (len(dataset), 20))


if __name__ == '__main__':
    unittest.main()
