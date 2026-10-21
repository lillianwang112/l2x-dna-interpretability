"""Small known-answer checks for reporting, masking, and failure handling."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from keras import Input, Model
from keras.layers import Dense, Flatten, GlobalAveragePooling1D

import validation_checks as selection
import run_diagnostics as diagnostics
from dna_l2x import build_l2x_model
from data_contract import write_artifact
from fixtures import make_fixture


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.sequences = ["AAAAACCCCC", "CCCCCCCCCC"]
        self.labels = np.array([1, 0])
        self.scores = np.tile(np.arange(10, 0, -1), (2, 1))

    def evaluate(self, **kwargs):
        return selection.evaluate_selection(self.sequences, self.labels, self.scores,
                                             trials=10, seed=7, **kwargs)

    def test_hit_is_measured_and_distinguished_from_complete_recovery(self):
        result = self.evaluate(k=1)
        self.assertEqual(result['positive_any_motif_overlap_rate'], 1.0)
        self.assertEqual(result['positive_full_motif_window_recovery_rate'], 0.0)
        self.assertEqual(result['positive_motif_position_recall'], 0.2)
        self.assertEqual(result['random_baseline']['expected_any_overlap_rate'], 0.5)
        reversed_scores = self.scores[:, ::-1].copy()
        missed = selection.evaluate_selection(self.sequences, self.labels, reversed_scores,
                                              k=1, trials=10)
        self.assertEqual(missed['positive_any_motif_overlap_rate'], 0.0)
        self.assertEqual(missed['random_baseline']['model_minus_expected_random'], -0.5)

    def test_complete_window_recovery_and_exact_random_expectation(self):
        result = self.evaluate(k=5)
        self.assertEqual(result['positive_full_motif_window_recovery_rate'], 1.0)
        self.assertEqual(result['positive_motif_position_recall'], 1.0)
        self.assertAlmostEqual(self.evaluate(k=2)['random_baseline']['expected_any_overlap_rate'], 7 / 9)

    def test_overlapping_motifs_are_counted_as_a_union(self):
        result = selection.evaluate_selection(["AAAAAACCCC"], [1], self.scores[:1], k=5, trials=5)
        self.assertAlmostEqual(result['positive_motif_position_recall'], 5 / 6)
        self.assertEqual(result['positive_full_motif_window_recovery_rate'], 1.0)
        self.assertEqual(selection.motif_windows("AAAAAA"), [(0, 4), (1, 5)])

    def test_random_comparison_is_reproducible(self):
        self.assertEqual(self.evaluate(k=2), self.evaluate(k=2))
        self.assertEqual(self.evaluate(k=10)['random_baseline']['monte_carlo_mean'], 1.0)

    def test_absent_classes_produce_null_metrics_not_division_errors(self):
        negative = selection.evaluate_selection([self.sequences[1]], [0], self.scores[:1], k=1, trials=2)
        self.assertIsNone(negative['positive_any_motif_overlap_rate'])
        self.assertIsNone(negative['random_baseline']['expected_any_overlap_rate'])
        positive = selection.evaluate_selection([self.sequences[0]], [1], self.scores[:1], k=1, trials=2)
        self.assertIsNone(positive['selection_content']['negative_selected_a_frequency'])
        json.dumps(negative, allow_nan=False)
        json.dumps(positive, allow_nan=False)

    def test_ties_select_exactly_k_with_deterministic_positions(self):
        np.testing.assert_array_equal(selection.select_topk(np.ones((2, 4)), 2), [[0, 1], [0, 1]])

    def test_invalid_k_and_scores_fail_explicitly(self):
        for k in [0, -1, 11, 1.5, True]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                selection.select_topk(self.scores, k)
        for scores in [np.full((2, 10), np.nan), np.full((2, 10), np.inf), np.zeros(10), np.empty((0, 10))]:
            with self.assertRaises(ValueError):
                selection.select_topk(scores, 1)

    def test_alignment_and_invalid_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            selection.evaluate_selection(self.sequences, self.labels, self.scores[:1], k=1)
        with self.assertRaises(ValueError):
            selection.evaluate_selection(self.sequences, [0, 1], self.scores, k=1)
        with self.assertRaises(ValueError):
            self.evaluate(k=1, motif="")
        with self.assertRaises(ValueError):
            selection.evaluate_selection(self.sequences, self.labels, self.scores, trials=0)

    def test_cli_missing_input_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                selection.main(['--dataset', str(Path(temp) / 'absent.json')])
        self.assertEqual(raised.exception.code, 2)

    def test_cli_writes_computed_metrics_and_population(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            dataset_path, split_path, dataset, split = make_fixture(temp)
            scores = np.tile(np.arange(20, 0, -1), (len(dataset), 1))
            teacher_fp = write_artifact(temp / 'teacher.npz', 'teacher_predictions',
                                       np.eye(2)[dataset.labels], dataset, split, seed=0)
            write_artifact(temp / 'scores.npz', 'selection_scores', scores,
                           dataset, split, seed=0, teacher_fingerprint=teacher_fp)
            output = temp / 'nested' / 'metrics.json'
            with contextlib.redirect_stdout(io.StringIO()):
                selection.main(['--dataset', str(dataset_path), '--split_json', str(split_path),
                                '--scores', str(temp / 'scores.npz'), '--output', str(output),
                                '--k', '1', '--trials', '3'])
            report = json.loads(output.read_text())
            self.assertEqual(report['positive_any_motif_overlap_rate'], 1.0)
            self.assertEqual(report['population'], 'validation ancestor families')
            self.assertEqual(report['n_sequences'], len(split['validation']))



class FixedSelector:
    def predict(self, X, **kwargs):
        # All scores tie, so the shared rule must select position zero for k=1.
        return np.zeros((len(X), X.shape[1], 1), dtype=np.float32)


class FidelityTests(unittest.TestCase):
    def setUp(self):
        inputs = Input(shape=(3, 4))
        dense = Dense(2, activation='softmax')
        self.model = Model(inputs, dense(Flatten()(inputs)))
        weights = np.zeros((12, 2), dtype=np.float32)
        weights[0, 1] = 10  # A at position zero -> class 1.
        weights[1, 0] = 10  # C at position zero -> class 0.
        dense.set_weights([weights, np.zeros(2, dtype=np.float32)])
        self.X = np.zeros((2, 3, 4), dtype=np.float32)
        self.X[0, 0, 0] = 1
        self.X[1, 0, 1] = 1
        self.teacher = np.array([[0, 1], [1, 0]], dtype=np.float32)

    def test_keep_and_complement_have_known_answers(self):
        result = diagnostics.evaluate_masked_fidelity(self.model, FixedSelector(), self.X, self.teacher, k=1)
        self.assertEqual(result['keep_selected_teacher_agreement'], 1.0)
        self.assertEqual(result['remove_selected_teacher_agreement'], 0.5)

    def test_approximator_reuses_weights_and_supports_the_actual_architecture(self):
        model, selector = build_l2x_model(seq_len=100, k=5, tau=0.5, lr=1e-3)
        approximator = diagnostics.get_approximator(model)
        self.assertIs(approximator.layers[-1].kernel, model.layers[-1].kernel)
        X = np.zeros((2, 100, 4), dtype=np.float32)
        result = diagnostics.evaluate_masked_fidelity(model, selector, X, self.teacher, k=5)
        self.assertTrue(all(0 <= v <= 1 for v in result.values()))

    def test_incompatible_architecture_raises_instead_of_inventing_a_score(self):
        inputs = Input(shape=(3, 4))
        incompatible = Model(inputs, Dense(2, activation='softmax')(GlobalAveragePooling1D()(inputs)))
        with self.assertRaisesRegex(ValueError, 'Flatten'):
            diagnostics.evaluate_masked_fidelity(incompatible, FixedSelector(), self.X, self.teacher, k=1)

    def test_invalid_probabilities_and_selector_shapes_fail(self):
        for teacher in [np.full((2, 2), np.nan), np.ones((2, 2)), np.array([[2, -1], [0, 1]])]:
            with self.assertRaises(ValueError):
                diagnostics.evaluate_masked_fidelity(self.model, FixedSelector(), self.X, teacher, k=1)
        bad_selector = FixedSelector()
        bad_selector.predict = lambda *args, **kwargs: np.zeros((2, 3))
        with self.assertRaises(ValueError):
            diagnostics.evaluate_masked_fidelity(self.model, bad_selector, self.X, self.teacher, k=1)

    def test_label_accuracy_is_not_used_as_teacher_agreement_target(self):
        truth = np.array([[1, 0], [1, 0]])
        wrong_teacher = np.array([[0, 1], [0, 1]])
        result = diagnostics.baseline_metrics(truth, wrong_teacher)
        self.assertEqual(result['teacher_label_accuracy'], 0.0)
        self.assertEqual(result['majority_teacher_class_agreement'], 1.0)
        self.assertEqual(result['perfect_teacher_agreement'], 1.0)

    def test_per_seed_mean_and_variability_are_saved_separately(self):
        runs = [{'k': 1, 'keep_selected_teacher_agreement': x,
                 'remove_selected_teacher_agreement': 0.5} for x in [0.5, 1.0]]
        summary = diagnostics.summarize_runs(runs)[0]
        self.assertEqual(summary['keep_selected_teacher_agreement_mean'], 0.75)
        self.assertAlmostEqual(summary['keep_selected_teacher_agreement_std'], np.std([0.5, 1.0], ddof=1))
        self.assertIsNone(diagnostics.summarize_runs(runs[:1])[0]['keep_selected_teacher_agreement_std'])
        with self.assertRaises(ValueError):
            diagnostics.summarize_runs([])

    def test_training_failure_stops_run_without_fallback_metrics(self):
        class FailingModel:
            def fit(self, *args, **kwargs):
                raise ValueError('controlled failure')
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            dataset_path, split_path, dataset, split = make_fixture(temp)
            teacher_path = temp / 'teacher.npz'
            write_artifact(teacher_path, 'teacher_predictions', np.eye(2)[dataset.labels], dataset, split, seed=0)
            output = temp / 'failed_run'
            with patch.object(diagnostics, 'build_l2x_model', return_value=(FailingModel(), FixedSelector())):
                with self.assertRaisesRegex(RuntimeError, 'k=1, seed=0'):
                    diagnostics.run_diagnostics(dataset_path, teacher_path, output,
                                                max_k=1, n_seeds=1, split_path=split_path)
            self.assertFalse((output / 'fidelity_metrics.json').exists())



if __name__ == '__main__':
    unittest.main()
