"""Audit current-pixel Tap contour work inside an existing production-loop replay.

Uses an existing replay command, mock input only. ROI copies happen after the
measured function returns. This is a hotspot probe, not a full-loop admission
benchmark or a game judgement analysis. No production source is modified.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-command', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rois', type=int, default=768)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'temp'):
        parser.error('Analysis output must stay in app/temp.')
    if args.max_rois < 0:
        parser.error('max-rois must not be negative.')
    command = json.loads(args.replay_command.read_text(encoding='utf-8'))
    replay_args = command[command.index('--branch-root'):]
    branch = Path(replay_args[replay_args.index('--branch-root') + 1]).resolve()
    replay_args[replay_args.index('--output') + 1] = str(output.with_suffix('.loop.json'))
    sys.path[:0] = [str(ROOT / 'tools'), str(branch)]
    import numpy as np
    import tap_replay
    from agent.music import holds, tap_physical_identity as physical

    def hashes():
        return {name: hashlib.sha256((branch / 'agent/music' / name).read_bytes()).hexdigest()
                for name in ('tap_physical_identity.py', 'holds.py', 'tracking.py', 'tap_hold_chain.py')}

    source_before = hashes()
    inspect_rows, detector_rows, rois = [], [], {}
    original_inspect, original_detector = physical._inspect_contour, holds.detect_hold_tails

    def inspect(candidate, frame):
        begin = time.perf_counter()
        result = original_inspect(candidate, frame)
        elapsed = (time.perf_counter() - begin) * 1000.
        row = {'sequence': frame.sequence, 'time': frame.midpoint,
               'candidate': asdict(candidate), 'verdict': result.verdict,
               'reason': result.reason, 'cpu_ms': elapsed,
               'normalized': asdict(result.candidate) if result.candidate is not None else None}
        x, y, width, height = candidate.box
        image = frame.image
        if (len(rois) < args.max_rois and isinstance(image, np.ndarray) and image.ndim == 3
                and x >= 0 and y >= 0 and x + width <= image.shape[1] and y + height <= image.shape[0]):
            key = f'roi_{len(rois):04d}'
            rois[key] = image[y:y + height, x:x + width, :3].copy()
            row['roi_key'] = key
        inspect_rows.append(row)
        return result

    def detector(*positional, **keywords):
        begin = time.perf_counter()
        result = original_detector(*positional, **keywords)
        elapsed = (time.perf_counter() - begin) * 1000.
        detector_rows.append({'cpu_ms': elapsed, 'detections': len(result),
                              'physical_only': keywords.get('physical_only', False)})
        return result

    previous_argv = sys.argv
    try:
        sys.argv = ['tap_replay.py'] + replay_args
        with patch.object(physical, '_inspect_contour', new=inspect), \
             patch.object(holds, 'detect_hold_tails', new=detector):
            tap_replay.main()
    finally:
        sys.argv = previous_argv
    source_after = hashes()

    def summary(values):
        return {**{f'p{p}_ms': float(np.percentile(values, p)) if values else 0.
                   for p in (50, 95, 99)}, 'count': len(values), 'sum_ms': float(sum(values))}

    per_frame = {}
    for row in inspect_rows:
        per_frame[row['sequence']] = per_frame.get(row['sequence'], 0.) + row['cpu_ms']
    report = {'schema': 1, 'source_before': source_before, 'source_after': source_after,
              'source_changed': source_before != source_after, 'replay_command': command,
              'notes': 'Original function time only; ROI copies are after timing. Mock production loop; not game judgement. Per-frame contours omit zero-call frames.',
              'inspect_call_timing': summary([row['cpu_ms'] for row in inspect_rows]),
              'inspect_active_frame_timing': summary(list(per_frame.values())),
              'gold_detector_call_timing': summary([row['cpu_ms'] for row in detector_rows]),
              'inspect_rows': inspect_rows, 'gold_detector_rows': detector_rows,
              'roi_count': len(rois), 'roi_limit': args.max_rois}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix('.rois.npz'), **rois)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('source_changed', 'inspect_call_timing',
                     'inspect_active_frame_timing', 'gold_detector_call_timing', 'roi_count')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
