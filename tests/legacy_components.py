"""Frozen component reference for equivalence tests, not runtime detection."""
from typing import Any
import numpy as np

def connected_components(mask: Any, min_pixels: int) -> list[tuple[tuple[int, int, int, int], int]]:
    """RLE connected components: Python iterates runs, never individual pixels."""
    if np is None:
        raise RuntimeError("NumPy is required by the music vision engine")
    boolean = np.asarray(mask, dtype=bool)
    parent: list[int] = []
    runs: list[tuple[int, int, int, int]] = []  # row, start, end-exclusive, label
    previous: list[tuple[int, int, int]] = []

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in range(boolean.shape[0]):
        columns = np.flatnonzero(boolean[row])
        if not columns.size:
            previous = []
            continue
        split_at = np.flatnonzero(np.diff(columns) > 1) + 1
        groups = np.split(columns, split_at)
        current: list[tuple[int, int, int]] = []
        for group in groups:
            start, end = int(group[0]), int(group[-1]) + 1
            label = len(parent)
            parent.append(label)
            for previous_start, previous_end, previous_label in previous:
                if previous_end < start or previous_start > end:
                    continue
                union(label, previous_label)
            runs.append((row, start, end, label))
            current.append((start, end, label))
        previous = current

    aggregates: dict[int, list[int]] = {}
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
    result: list[tuple[tuple[int, int, int, int], int]] = []
    for x0, y0, x1, y1, count in aggregates.values():
        if count >= min_pixels:
            result.append(((x0, y0, x1 - x0, y1 - y0), count))
    return result
