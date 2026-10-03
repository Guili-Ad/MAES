"""Exact contracts for the row-overlap sweep, including diagonal endpoints."""
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.music.components import connected_components


def reference_components(mask, min_pixels):
    """Pre-optimization row scan, retained only as a development oracle."""
    boolean = np.asarray(mask, dtype=bool)
    edges = np.diff(np.pad(boolean.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    rows, starts = np.nonzero(edges == 1)
    _, ends = np.nonzero(edges == -1)
    parent, runs, previous, current = [], [], [], []
    last_row = -2

    def find(label):
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    for row, start, end in zip(rows.tolist(), starts.tolist(), ends.tolist()):
        if row != last_row:
            previous = current if row == last_row + 1 else []
            current = []
            last_row = row
        label = len(parent)
        parent.append(label)
        for a, b, prior in previous:
            if b < start or a > end:
                continue
            left, right = find(label), find(prior)
            if left != right:
                parent[right] = left
        runs.append((row, start, end, label))
        current.append((start, end, label))
    aggregates = {}
    for row, start, end, label in runs:
        root = find(label)
        if root not in aggregates:
            aggregates[root] = [start, row, end, row + 1, end - start]
        else:
            item = aggregates[root]
            item[0] = min(item[0], start)
            item[1] = min(item[1], row)
            item[2] = max(item[2], end)
            item[3] = max(item[3], row + 1)
            item[4] += end - start
    return [((x, y, right - x, bottom - y), count)
            for x, y, right, bottom, count in aggregates.values()
            if count >= min_pixels]


def pixel_reference(mask, min_pixels):
    """Independent 8-neighbour flood fill ordered by first row-major pixel."""
    mask = np.asarray(mask, dtype=bool)
    seen = set()
    answer = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or (x, y) in seen:
                continue
            seen.add((x, y))
            pending, component = [(x, y)], []
            while pending:
                px, py = pending.pop()
                component.append((px, py))
                for ny in range(max(0, py - 1), min(height, py + 2)):
                    for nx in range(max(0, px - 1), min(width, px + 2)):
                        if mask[ny, nx] and (nx, ny) not in seen:
                            seen.add((nx, ny))
                            pending.append((nx, ny))
            if len(component) < min_pixels:
                continue
            xs, ys = zip(*component)
            answer.append(((min(xs), min(ys), max(xs) - min(xs) + 1,
                            max(ys) - min(ys) + 1), len(component)))
    return answer


class ComponentsSweepTests(unittest.TestCase):
    def assert_equivalent(self, mask, min_pixels=1, *, independent=False):
        actual = connected_components(mask, min_pixels)
        self.assertEqual(actual, reference_components(mask, min_pixels))
        if independent:
            self.assertEqual(actual, pixel_reference(mask, min_pixels))

    def test_exhaustive_three_by_four(self):
        powers = np.arange(12, dtype=np.uint16)
        for value in range(1 << 12):
            mask = ((value >> powers) & 1).reshape(3, 4)
            self.assert_equivalent(mask, independent=True)
            self.assert_equivalent(mask.T, independent=True)

    def test_random_masks_and_filters(self):
        random = np.random.default_rng(20261003)
        for shape in ((1, 71), (37, 1), (8, 19), (24, 61), (64, 127)):
            for probability in (.01, .08, .3, .5, .9, 1.):
                for _ in range(4):
                    mask = random.random(shape) < probability
                    for minimum in (0, 1, 2, 7, 41, mask.size + 1):
                        self.assert_equivalent(mask, minimum)
                    self.assert_equivalent(mask, independent=True)

    def test_diagonal_endpoint_equality_is_connected(self):
        # Runs are end-exclusive. b == start and a == end still connect
        # diagonally; changing these comparisons to <= or >= loses pixels.
        for mask in (np.eye(8, dtype=bool), np.fliplr(np.eye(8, dtype=bool))):
            self.assertEqual(connected_components(mask, 1), [((0, 0, 8, 8), 8)])
            self.assert_equivalent(mask, independent=True)

    def test_large_bridge_retains_component_output_order(self):
        mask = np.zeros((7, 31), dtype=bool)
        mask[0, (1, 5, 12, 20, 28)] = True
        mask[1, 0:30] = True
        mask[3:7, 3:8] = True
        mask[4:6, 15:17] = True
        self.assert_equivalent(mask, independent=True)

    def test_blank_row_does_not_join_components(self):
        mask = np.ones((5, 9), dtype=bool)
        mask[2] = False
        self.assertEqual(connected_components(mask, 1),
                         [((0, 0, 9, 2), 18), ((0, 3, 9, 2), 18)])

    def test_checkerboard_many_short_runs(self):
        yy, xx = np.indices((64, 240))
        self.assert_equivalent((xx + yy) % 2 == 0, independent=True)
        self.assert_equivalent(xx % 3 == 0, independent=True)

    def test_non_boolean_and_non_contiguous_input(self):
        random = np.random.default_rng(123)
        array = random.integers(0, 4, (33, 79), dtype=np.int16)
        self.assert_equivalent(array[:, ::3], independent=True)
        self.assert_equivalent(array[::-1, ::-2], independent=True)

    def test_empty_dimensions(self):
        for shape in ((0, 0), (0, 12), (9, 0)):
            self.assert_equivalent(np.zeros(shape, bool), independent=True)


if __name__ == '__main__':
    unittest.main()
