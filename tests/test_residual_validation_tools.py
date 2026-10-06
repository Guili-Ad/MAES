"""Development admission must retain slow repetitions and fail closed."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from residual_report import performance_pair


def report(cost):
    return dict(frames=20, input_digest='pixels', candidate_digest='candidates',
        config_hash='config', effective_calibration_hash='calibration', inventory_hash='inventory',
        fixture_file_hashes={}, source_hash='source', source_files={'agent/test.py': 'hash'},
        measurement_hash='wall', tool_hash='tool',
        comparison={'comparison_valid': True, 'guards': {'same_inputs': True}},
        rows=[{'qualification_cpu_ms': cost, 'nonwaiting_planning_cpu_ms': 0.,
            'nonwaiting_dispatch_cpu_ms': 0., 'frame_processing_cpu_ms': cost} for _ in range(20)])


class ResidualValidationToolsTests(unittest.TestCase):
    def test_slow_repetition_is_not_discarded_or_averaged_away(self):
        baseline = [report(10.) for _ in range(3)]
        candidate = [report(10.), report(10.), report(20.)]
        result = performance_pair(baseline, candidate, candidate[0]['source_files'])
        self.assertFalse(result['passed'])
        self.assertEqual(result['p95_delta_ms']['qualification'], 10.)
        self.assertEqual(result['measured_samples'], 60)
        self.assertEqual(len(result['trial_p95_deltas_ms']), 3)

    def test_changed_input_source_or_guard_invalidates_pooled_result(self):
        for field in ('input_digest', 'tool_hash', 'source_hash', 'measurement_hash', 'frames'):
            with self.subTest(field=field):
                baseline, candidate = [report(10.), report(10.)], [report(10.), report(10.)]
                candidate[1][field] = 21 if field == 'frames' else 'changed'
                self.assertFalse(performance_pair(baseline, candidate, candidate[0]['source_files'])['passed'])
        baseline, candidate = [report(10.)], [report(10.)]
        candidate[0]['comparison']['guards']['same_inputs'] = False
        self.assertFalse(performance_pair(baseline, candidate, candidate[0]['source_files'])['passed'])

    def test_exact_two_ms_gate_and_current_source(self):
        base, new = report(10.), report(12.)
        self.assertTrue(performance_pair([base], [new], new['source_files'])['passed'])
        self.assertFalse(performance_pair([base], [new], {})['passed'])
        new = report(12.01)
        self.assertFalse(performance_pair([base], [new], new['source_files'])['passed'])

    def test_unmatched_repetitions_fail_explicitly(self):
        for bases, candidates in (([], []), ([report(10.)], [report(10.), report(10.)])):
            with self.assertRaises(ValueError):
                performance_pair(bases, candidates, {})


if __name__ == '__main__':
    unittest.main()
