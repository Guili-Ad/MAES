"""Paired fixed-colour mask benchmark on actual decoded, calibrated video ROIs.

Development-only: no controller, no action table, no runtime-resource writes.
Both functions receive identical non-contiguous sample=3 BGR views. This
measures the mask only, not full tracker admission or game judgement timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workspace_paths import ffmpeg_binary


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--start', type=float, required=True)
    parser.add_argument('--duration', type=float, default=1.)
    parser.add_argument('--frames', type=int, default=22)
    parser.add_argument('--rounds', type=int, default=25)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, default=ffmpeg_binary('ffmpeg.exe'))
    parser.add_argument('--ffprobe', type=Path, default=ffmpeg_binary('ffprobe.exe'))
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / 'temp'):
        parser.error('Benchmark output must stay in app/temp.')
    if min(args.frames, args.rounds) <= 0 or args.duration <= 0:
        parser.error('frames, rounds and duration must be positive.')
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from agent.music.gold_mask import build_gold_mask
    from agent.music.vision import build_color_mask
    from tap_replay import video_frames

    raw = json.loads(args.calibration.read_text(encoding='utf-8'))
    cal = raw['profiles']['7@1280x720'] if 'profiles' in raw else raw
    x, y, width, height = cal['candidate_roi']
    rois = []
    for pts, image, _ in video_frames(args):
        if len(rois) < args.frames:
            rois.append((pts, image[max(0, y):min(720, y + height):3,
                                     max(0, x):min(1280, x + width):3]))
    if not rois:
        raise ValueError('No actual decoded ROIs')

    def old(image):
        return build_color_mask(image, [[7, 5, 145]], [[45, 200, 255]])

    def source_hash():
        return {name: hashlib.sha256((ROOT / 'agent/music' / name).read_bytes()).hexdigest()
                for name in ('gold_mask.py', 'vision.py')}

    source_before = source_hash()
    for _, image in rois:
        if not np.array_equal(old(image), build_gold_mask(image)):
            raise AssertionError('Actual decoded ROI differs from fixed legacy mask')
    for _ in range(2):
        for _, image in rois:
            old(image)
            build_gold_mask(image)
    samples = []
    for repeat in range(args.rounds):
        for index, (pts, image) in enumerate(rois):
            row = {'round': repeat, 'frame': index, 'pts': pts}
            functions = [('legacy', old), ('candidate', build_gold_mask)]
            if (repeat + index) % 2:
                functions.reverse()
            for name, function in functions:
                begin = time.perf_counter()
                function(image)
                row[name + '_ms'] = (time.perf_counter() - begin) * 1000.
            row['saved_ms'] = row['legacy_ms'] - row['candidate_ms']
            samples.append(row)
    source_after = source_hash()

    def summary(field):
        values = [row[field] for row in samples]
        return {f'p{p}_ms': float(np.percentile(values, p)) for p in (50, 95, 99)}

    report = {'schema': 1, 'scope': 'fixed gold mask only', 'video': str(args.video.resolve()),
              'calibration': str(args.calibration.resolve()), 'candidate_roi': [x, y, width, height],
              'sample': 3, 'actual_rois': [{'pts': pts, 'shape': list(image.shape),
                  'strides': list(image.strides), 'sha256': hashlib.sha256(image.tobytes()).hexdigest()}
                  for pts, image in rois], 'pixel_equivalence': True,
              'source_before': source_before, 'source_after': source_after,
              'source_changed': source_before != source_after, 'pairs': len(samples),
              'legacy': summary('legacy_ms'), 'candidate': summary('candidate_ms'),
              'paired_savings': summary('saved_ms'), 'samples': samples,
              'notes': 'Alternating A/B order; decode, image copying and equivalence checks outside timing. Excludes full tracker, controller, OCR, game judgement and new-module import.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('source_changed', 'pixel_equivalence', 'pairs',
                                                 'legacy', 'candidate', 'paired_savings')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
