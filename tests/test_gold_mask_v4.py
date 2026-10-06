"""Exact equivalence of fixed pale-gold HSV mask, not a new colour policy."""
import os
os.environ.setdefault('MAES_AGENT_TEST_MODE', '1')
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.music.vision import build_color_mask


def legacy(image):
    return build_color_mask(image, [[7, 5, 145]], [[45, 200, 255]])


def optimized(image):
    from agent.music.gold_mask import build_gold_mask
    return build_gold_mask(image)


class GoldMaskTests(unittest.TestCase):
    def test_all_16777216_uint8_bgr_values_exact(self):
        # Independent 256x256 planes bound memory and cover every BGR triple.
        b, g = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8))
        image = np.empty((256, 256, 3), dtype=np.uint8)
        image[..., 0], image[..., 1] = b, g
        for red in range(256):
            image[..., 2] = red
            expected, actual = legacy(image), optimized(image)
            self.assertTrue(np.array_equal(actual, expected), f'red={red}')
            self.assertEqual(actual.dtype, np.dtype(bool))

    def test_random_strided_reversed_and_extra_channels(self):
        image = np.random.default_rng(43).integers(0, 256, (99, 141, 5), dtype=np.uint8)
        for array in (image, image[::3, ::2], image[::-1, ::-1], image.transpose(1, 0, 2)):
            self.assertTrue(np.array_equal(optimized(array), legacy(array)))

    def test_value_saturation_hue_ties_and_integer_boundaries(self):
        triples = [(0, 0, 0), (145, 145, 145), (255, 255, 255),
                   (0, 255, 255), (255, 255, 0), (255, 0, 255)]
        for maximum in (144, 145, 146, 254, 255):
            for delta in range(256):
                minimum = max(0, maximum - delta)
                for middle in (minimum, min(maximum, minimum + 1), (minimum + maximum) // 2, maximum):
                    triples += [(minimum, middle, maximum), (minimum, maximum, middle),
                                (middle, minimum, maximum), (maximum, minimum, middle)]
        array = np.array(triples, np.uint8).reshape(-1, 1, 3)
        self.assertTrue(np.array_equal(optimized(array), legacy(array)))

    def test_empty_uint8_arrays_keep_shape(self):
        for shape in ((0, 0, 3), (0, 10, 3), (10, 0, 4)):
            image = np.empty(shape, np.uint8)
            result = optimized(image)
            self.assertEqual(result.shape, shape[:2])
            self.assertTrue(np.array_equal(result, legacy(image)))

    def test_non_uint8_contract_falls_back_to_existing_converter(self):
        image = np.random.default_rng(42).integers(0, 256, (13, 15, 3), dtype=np.uint8)
        for dtype in (np.int8, np.int16, np.uint16, np.float32, np.float64, np.bool_):
            array = image.astype(dtype)
            with patch('agent.music.vision.build_color_mask', wraps=build_color_mask) as fallback:
                self.assertTrue(np.array_equal(optimized(array), legacy(array)))
                self.assertEqual(fallback.call_count, 1)
                self.assertEqual(fallback.call_args.args[1:], ([[7, 5, 145]], [[45, 200, 255]]))
        values = image.tolist()
        self.assertTrue(np.array_equal(optimized(values), legacy(values)))

    def test_invalid_shapes_preserve_legacy_error(self):
        for image in (np.zeros((3, 5), np.uint8), np.zeros((3, 5, 2), np.uint8), None):
            with self.assertRaises(ValueError) as expected:
                legacy(image)
            with self.assertRaises(type(expected.exception)) as actual:
                optimized(image)
            self.assertEqual(str(actual.exception), str(expected.exception))


if __name__ == '__main__':
    unittest.main()
