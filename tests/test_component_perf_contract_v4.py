"""The development performance gate must reject noisy self-comparisons."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from round3_component_hotspot_v4 import optimization_gate


def summary(p50=1., p95=2., detector_increment=0.):
    return {
        'gold_cc': {'wall_ms': {'baseline': {'p50': 2., 'p95': 3.},
                              'candidate': {'p50': p50, 'p95': p95}}},
        'gold_detector': {'wall_ms': {'baseline': {'p95': 4.},
                                    'candidate': {'p95': 4.+detector_increment}}},
    }


class ComponentPerformanceGateTests(unittest.TestCase):
    def test_same_source_cannot_pass_even_when_timings_look_faster(self):
        gate = optimization_gate(summary(), 'same', 'same', True)
        self.assertFalse(gate['different_source'])
        self.assertFalse(all(gate.values()))

    def test_changed_source_or_detector_regression_cannot_pass(self):
        self.assertFalse(all(optimization_gate(summary(), 'old', 'new', False).values()))
        self.assertFalse(all(optimization_gate(summary(p95=3.1), 'old', 'new', True).values()))
        self.assertFalse(all(optimization_gate(summary(p50=2.1), 'old', 'new', True).values()))
        self.assertFalse(all(optimization_gate(summary(detector_increment=2.1), 'old', 'new', True).values()))

    def test_real_source_and_both_cc_quantiles_improve(self):
        self.assertTrue(all(optimization_gate(summary(), 'old', 'new', True).values()))


if __name__ == '__main__':
    unittest.main()
