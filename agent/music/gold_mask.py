"""Exact fast mask for the existing fixed pale-gold HSV interval.

This is the UInt8 BGR algebraic equivalent of build_color_mask with
HSV [7,5,145]..[45,200,255], not a detector or a changed colour threshold.
Other input dtypes retain the existing converter's truncation/wrap contract.
"""
from __future__ import annotations

import numpy as np


def build_gold_mask(image):
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] < 3:
        from .vision import build_color_mask
        return build_color_mask(image, [[7, 5, 145]], [[45, 200, 255]])

    b, g, r = (array[..., channel].astype(np.int16) for channel in range(3))
    maximum = np.maximum(np.maximum(r, g), b)
    delta = maximum - np.minimum(np.minimum(r, g), b)
    # Saturation truncates to floor(255*delta/maximum). Its inclusive [5,200]
    # interval is exactly 5*maximum <= 255*delta < 201*maximum. Values must be
    # >=145, so there is no zero-divisor case. Use Int32 only for these products.
    scaled_delta = np.multiply(delta, 255, dtype=np.int32)
    saturation = (scaled_delta >= 5*maximum) & (scaled_delta < np.multiply(maximum, 201, dtype=np.int32))

    # Existing conversion chooses R before G before B on maximum ties. Hue
    # [7,45] means degree mod 360 in [14,91]. In the R branch only the lower
    # bound can bind; in the non-R-tie G branch only the upper bound can bind.
    # floor(60*(b-r)/delta)+120 <=91 is 60*(r-b)>28*delta, strictly >.
    # B-only maxima cannot reach this interval. All hue products fit Int16.
    red = (r >= g) & (r >= b) & (60*(g-b) >= 14*delta)
    green = (g > r) & (g >= b) & (60*(r-b) > 28*delta)
    return (maximum >= 145) & saturation & (red | green)
