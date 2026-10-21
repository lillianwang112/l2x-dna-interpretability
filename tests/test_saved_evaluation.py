"""Final evaluation must preserve locked inputs and use the named held-out rows."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from compare_mask_training import infer
from data_contract import load_artifact, write_artifact
from dna_l2x import build_l2x_model, load_dataset_json
from evaluate_saved import evaluate, load_protocol, lock_protocol, summarize
from fixtures import make_fixture
from run_diagnostics import get_approximator
from validation_checks import select_topk


class SavedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        data = self.root / 'data'
        data.mkdir()
        self.dataset_path, self.split_path, self.dataset, self.split = make_fixture(data)
        target = np.eye(2, dtype=np.float32)[self.dataset.labels] * .8 + .1
        self.teacher = self.root / 'outputs/teacher/teacher_predictions.npz'
        fingerprint = write_artifact(self.teacher, 'teacher_predictions', target, self.dataset, self.split, seed=0)
        tf.keras.utils.set_random_seed(0)
        model, selector = build_l2x_model(20, 1, .5, .001)
        model.optimizer.build(model.trainable_variables)
        folder = self.root / 'outputs/fidelity/k1_seed0'
        folder.mkdir(parents=True)
        model.save_weights(folder / 'l2x.weights.h5')
        _, x, _ = load_dataset_json(self.dataset_path, 20)
        self.scores_path = folder / 'selection_scores.npz'
        write_artifact(self.scores_path, 'selection_scores', infer(selector, x, 4).squeeze(-1),
                       self.dataset, self.split, seed=0, teacher_fingerprint=fingerprint)
        for mode in ['soft', 'hard']:
            out = self.root / f'outputs/mask_comparison/seed0_{mode}'
            out.mkdir(parents=True)
            get_approximator(model).save_weights(out / 'approximator.weights.h5')
        self.protocol_path = self.root / 'results/protocol.json'
        self.protocol = lock_protocol(self.root, self.protocol_path,
                                      dataset_path=str(self.dataset_path.relative_to(self.root)),
                                      split_path=str(self.split_path.relative_to(self.root)),
                                      budgets=(1,), seeds=(0,), reference_k=1, random_trials=2)

    def tearDown(self):
        self.temp.cleanup()

    def test_locked_file_changes_are_rejected_before_evaluation(self):
        self.teacher.write_bytes(self.teacher.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            load_protocol(self.root, self.protocol_path)

    def test_evaluation_is_inference_only_and_matches_independent_test_row_answers(self):
        output = self.root / 'results/report.json'
        with contextlib.redirect_stdout(io.StringIO()), patch.object(tf.keras.Model, 'fit', side_effect=AssertionError('No fitting')):
            report = evaluate(self.root, self.protocol_path, output)
        self.assertEqual(report['status'], 'complete')
        self.assertTrue(report['weights_and_inputs_unchanged'])
        for partition in ['validation', 'test']:
            population = report['partitions'][partition]
            self.assertEqual(population['n_sequences'], len(self.split[partition]))
            self.assertEqual(population['teacher']['true_label_accuracy'], 1)
            self.assertEqual(population['teacher']['true_label_auroc'], 1)
            self.assertEqual(len(population['sweep_runs']), 1)
            self.assertEqual(population['mask_comparison_runs'][0]['metrics'], population['mask_comparison_runs'][1]['metrics'])
        scores, _ = load_artifact(self.scores_path, 'selection_scores', self.dataset, self.split)
        positive = [i for i in self.split['test'] if self.dataset.labels[i] == 1]
        hits = []
        for i in positive:
            selected = int(select_topk(scores[i:i+1], 1)[0, 0])
            seq = self.dataset.sequences[i]
            positions = {j for start in range(len(seq)-4) if seq[start:start+5] == 'AAAAA' for j in range(start, start+5)}
            hits.append(selected in positions)
        self.assertEqual(report['partitions']['test']['sweep_runs'][0]['selection']['positive_any_motif_overlap_rate'], np.mean(hits))
        self.assertEqual([e['sample_id'] for e in report['examples']], self.protocol['example_sample_ids'])
        self.assertTrue(all(e['sample_id'] in self.dataset.sample_ids[self.split['test']] for e in report['examples']))
        with self.assertRaises(FileExistsError):
            evaluate(self.root, self.protocol_path, output)

    def test_sample_sd_and_missing_class_summaries(self):
        summary = summarize([{'score': .2, 'missing': None}, {'score': .6, 'missing': None}])
        self.assertAlmostEqual(summary['score']['mean'], .4)
        self.assertAlmostEqual(summary['score']['sample_sd'], np.sqrt(.08))
        self.assertIsNone(summary['missing']['mean'])
        self.assertIsNone(summary['missing']['sample_sd'])


if __name__ == '__main__':
    unittest.main()
