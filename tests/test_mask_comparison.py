"""Matched training controls, frozen selectors, and probability metric checks."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from compare_mask_training import array_hash, compare, infer, prediction_metrics
from data_contract import write_artifact
from dna_l2x import build_l2x_model, load_dataset_json
from fixtures import make_fixture


class MaskComparisonTests(unittest.TestCase):
    def test_probability_and_class_metrics_have_independent_known_answers(self):
        target = np.array([[.8, .2], [.3, .7]])
        labels = np.array([0, 1])
        exact = prediction_metrics(target, target, labels)
        self.assertEqual(exact['teacher_kl'], 0)
        self.assertEqual(exact['balanced_teacher_agreement'], 1)
        constant = prediction_metrics(target, np.array([[.8, .2], [.8, .2]]), labels)
        self.assertEqual(constant['teacher_agreement'], .5)
        self.assertEqual(constant['teacher_negative_agreement'], 1)
        self.assertEqual(constant['teacher_positive_agreement'], 0)
        self.assertEqual(constant['teacher_vs_approximator_confusion'], [[1, 0], [1, 0]])
        self.assertAlmostEqual(constant['positive_probability_mae'], .25)
        expected_kl = (.3*np.log(.3/.8) + .7*np.log(.7/.2)) / 2
        self.assertAlmostEqual(constant['teacher_kl'], expected_kl)

    def test_missing_teacher_class_is_not_reported_as_zero_recall(self):
        p = np.array([[.8, .2], [.7, .3]])
        metrics = prediction_metrics(p, p, np.array([0, 1]))
        self.assertIsNone(metrics['teacher_positive_agreement'])
        self.assertIsNone(metrics['balanced_teacher_agreement'])

    def test_paired_branches_update_q_keep_selector_fixed_and_preserve_sources(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            dataset_path, split_path, dataset, split = make_fixture(root)
            _, x, _ = load_dataset_json(dataset_path, 20)
            target = np.eye(2, dtype=np.float32)[dataset.labels] * .8 + .1
            teacher_path = root / 'teacher.npz'
            fingerprint = write_artifact(teacher_path, 'teacher_predictions', target, dataset, split, seed=0)
            tf.keras.utils.set_random_seed(0)
            model, selector = build_l2x_model(20, 1, .5, .001)
            source_weights = array_hash(model.get_weights())
            source_selector = array_hash(selector.get_weights())
            folder = root / 'source/k1_seed0'
            folder.mkdir(parents=True)
            model.optimizer.build(model.trainable_variables)
            model.save_weights(folder / 'l2x.weights.h5')
            write_artifact(folder / 'selection_scores.npz', 'selection_scores', infer(selector, x, 4).squeeze(-1),
                           dataset, split, seed=0, teacher_fingerprint=fingerprint)
            output = root / 'comparison'
            result = compare(dataset_path, split_path, teacher_path, root / 'source', output,
                             k=1, seeds=(0,), epochs=1, batch_size=4, soft_draws=2)
            soft, hard = result['runs']
            self.assertEqual([soft['mode'], hard['mode']], ['soft', 'hard'])
            self.assertEqual(soft['initial_approximator_hash'], hard['initial_approximator_hash'])
            self.assertEqual(soft['batch_order_hashes'], hard['batch_order_hashes'])
            self.assertEqual(soft['optimizer_steps'], hard['optimizer_steps'])
            for run in [soft, hard]:
                self.assertTrue(run['selector_unchanged'])
                self.assertEqual(run['frozen_selector_hash'], source_selector)
                self.assertNotEqual(run['initial_approximator_hash'], run['final_approximator_hash'])
                self.assertEqual(len(run['history']), 1)
                self.assertEqual(len(run['final_soft_draws']), 2)
                self.assertEqual(np.sum(run['final_exact_k']['teacher_vs_approximator_confusion']), len(split['validation']))
            model.load_weights(folder / 'l2x.weights.h5')
            self.assertEqual(array_hash(model.get_weights()), source_weights)
            self.assertFalse(result['configuration']['test_metrics_evaluated'])
            self.assertEqual(result['configuration']['train_indices'], split['train'].tolist())
            self.assertEqual(result['configuration']['validation_indices'], split['validation'].tolist())
            with self.assertRaises(FileExistsError):
                compare(dataset_path, split_path, teacher_path, root / 'source', output, k=1, seeds=(0,))


if __name__ == '__main__':
    unittest.main()
