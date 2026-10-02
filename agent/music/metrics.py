"""Bounded exact recent samples plus 1 ms whole-run histogram (not game timing)."""
import math
from collections import deque


class MetricSeries:
    def __init__(self):
        self.recent = deque(maxlen=60)
        self.histogram = [0] * 5002
        self.count = self.invalid = 0
        self.maximum = 0.0

    def append(self, value):
        if not math.isfinite(value) or value < 0:
            self.invalid += 1
            return
        self.recent.append(value)
        self.count += 1
        self.maximum = max(self.maximum, value)
        self.histogram[min(5001, math.ceil(value))] += 1

    def __len__(self):
        return len(self.recent)

    def extend(self, values):
        for value in values:
            self.append(value)

    def __getitem__(self, index):
        return list(self.recent)[index]

    def __iter__(self):
        return iter(self.recent)

    def summary(self):
        def quantile(fraction):
            wanted = math.ceil(self.count * fraction)
            if not wanted:
                return 0
            total = 0
            for bucket, frequency in enumerate(self.histogram):
                total += frequency
                if total >= wanted:
                    return bucket if bucket < 5001 else None
        return {'count': self.count, 'p50_upper_ms': quantile(.5), 'p95_upper_ms': quantile(.95),
                'max': round(self.maximum, 3), 'bucket_ms': 1, 'overflow_count': self.histogram[-1],
                'invalid_count': self.invalid}
