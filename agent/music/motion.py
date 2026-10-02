"""Numerically stable shared motion fit; no tap/hold scheduling policy."""
import math


def weighted_slope(observations, *, recency=True):
    values = list(observations)
    if len(values) < 2 or any(not math.isfinite(o.timestamp) or not math.isfinite(o.progress) for o in values):
        return 0.0
    origin = values[0].timestamp
    w = wx = wy = wxx = wxy = 0.0
    for index, observation in enumerate(values):
        weight = float(index + 1) if recency else 1.0
        x, y = observation.timestamp - origin, observation.progress
        w += weight
        wx += weight * x
        wy += weight * y
        wxx += weight * x * x
        wxy += weight * x * y
    denominator = w * wxx - wx * wx
    if denominator <= 1e-9:
        return 0.0
    result = (w * wxy - wx * wy) / denominator
    return result if math.isfinite(result) else 0.0
