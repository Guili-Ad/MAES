"""Cheap negative prefilter for the supported pause overlay, never a UI verdict.

A possible white dialog body OR green title band must still pass the original
OCR. Unknown image formats/resolutions and dark frames also fall through to
OCR. This gate neither recognizes resume nor changes terminal recognition.
"""
import numpy as np


def pause_overlay_possible(image) -> bool:
    if (not isinstance(image, np.ndarray) or image.shape != (720, 1280, 3)
            or image.dtype != np.uint8):
        return True
    # The right-hand blank panel avoids cover art, notes and judgement text;
    # the title band provides an independent, more sensitive positive route.
    body = image[240:425:4, 770:930:4, :3].astype(np.int16)
    title = image[165:210:2, 590:690:2, :3].astype(np.int16)
    if body.max() < 80:
        return True
    white = (body.min(axis=2) >= 195) & (body.max(axis=2)-body.min(axis=2) < 40)
    blue, green, red = (title[..., i] for i in range(3))
    header = (green >= 85) & (green > red+25) & (green > blue+15)
    return bool(white.mean() >= .12 or header.mean() >= .10)
