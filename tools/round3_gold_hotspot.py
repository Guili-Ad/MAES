"""Paired fixed-video-frame CC sweep benchmark; no simulator or game scoring."""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import asdict
import hashlib
import importlib.util
import json
import pstats
from pathlib import Path
import subprocess
import sys
import time

APP_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = APP_ROOT.parent / '.work/round3-implementation-20261003'


def load_component(path):
    spec = importlib.util.spec_from_file_location('component_oracle', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.connected_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, default=WORK_ROOT/'gold-fixtures.json')
    parser.add_argument('--baseline-component', type=Path,
                        default=WORK_ROOT/'baseline/source/agent/music/components.py')
    parser.add_argument('--repeats', type=int, default=25)
    parser.add_argument('--profile-repeats', type=int, default=0,
                        help='Optional detector plus seven judgement-ribbon sweeps per fixed frame')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if not args.output.resolve().is_relative_to((APP_ROOT/'temp').resolve()):
        parser.error('Report must stay inside app/temp')
    sys.path.insert(0, str(APP_ROOT))
    import numpy as np
    from agent.music.components import connected_components as candidate
    from agent.music.models import MusicCalibrationData, MusicConfig
    from agent.music import holds
    from agent.music.vision import VisualMask, build_color_mask
    from agent.music.hold_topology import ribbon_at_judgement
    baseline = load_component(args.baseline_component)
    package = WORK_ROOT/'baseline/candidate-state'
    calibration = MusicCalibrationData(**json.loads(
        (package/'user-data/calibration/music.json').read_text())['profiles']['7@1280x720'])
    trace = package/'logs/tap-traces/20261003T114448Z-63e80544.jsonl'
    with trace.open() as stream:
        config = MusicConfig(**json.loads(next(stream))['config'])
    ffmpeg = APP_ROOT.parent/'.work/ffmpeg-7.1.1-extract/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe'
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    rows, timings = [], {}
    profiler = cProfile.Profile() if args.profile_repeats else None
    old_hold_cc = holds.connected_components

    def summary(samples):
        return {'count': len(samples), 'p50': float(np.percentile(samples, 50)),
                'p95': float(np.percentile(samples, 95)), 'max': float(max(samples))}

    def measure_pair(name, functions, expected, item):
        local = {'baseline': [], 'candidate': [], 'delta': []}
        # Alternate execution order to avoid assigning systematic warmup/CPU
        # drift to one implementation. FFT/video decode is outside timing.
        for repeat in range(args.repeats):
            order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
            pair = {}
            for variant in order:
                start = time.perf_counter_ns()
                actual = functions[variant]()
                elapsed = (time.perf_counter_ns() - start)/1e6
                if actual != expected:
                    raise AssertionError(f'{name} changed outputs at {item}, {variant}')
                pair[variant] = elapsed
                local[variant].append(elapsed)
            local['delta'].append(pair['candidate']-pair['baseline'])
        for variant, samples in local.items():
            timings.setdefault(name, {}).setdefault(variant, []).extend(samples)
        return {variant: summary(samples) for variant, samples in local.items()}

    try:
        for item in json.loads(args.fixtures.read_text()):
            video = APP_ROOT.parent/f'test-materials/music/20261003test3-{item["video_index"]}.mp4'
            raw = subprocess.check_output([
                str(ffmpeg), '-v', 'error', '-ss', str(item['time']), '-i', str(video),
                '-frames:v', '1', '-vf', 'scale=1280:720:flags=area',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1'], creationflags=flags)
            if len(raw) != 1280*720*3:
                raise RuntimeError(f'Missing complete video frame: {item}')
            frame = np.frombuffer(raw, np.uint8).reshape(720, 1280, 3)
            x, y, width, height = calibration.candidate_roi
            crop = frame[y:y+height:3, x:x+width:3]
            gold_mask = build_color_mask(crop, [[7, 5, 145]], [[45, 200, 255]])
            ordinary_mask = VisualMask.from_image(frame, calibration).mask
            record = {**item, 'frame_sha256': hashlib.sha256(raw).hexdigest()}
            for name, mask, minimum in (
                    ('gold_cc', gold_mask, max(3, config.hold_tail_min_pixels//9)),
                    ('ordinary_cc', ordinary_mask, calibration.candidate_min_pixels)):
                expected = baseline(mask, minimum)
                if candidate(mask, minimum) != expected:
                    raise AssertionError(f'Component mismatch at {item}: {name}')
                record[name] = {
                    'shape': list(mask.shape), 'pixels': int(mask.sum()),
                    'components': len(expected),
                    'timing_ms': measure_pair(name, {
                        'baseline': lambda m=mask, n=minimum: baseline(m, n),
                        'candidate': lambda m=mask, n=minimum: candidate(m, n)}, expected, item)}

            def detect(component):
                # Both variants run this exact, currently loaded detector.
                # Only the row-overlap primitive differs; thresholds/topology
                # and all root-owned policy changes are held identical.
                holds.connected_components = component
                return [asdict(d) for d in holds.detect_hold_tails(frame, calibration, config)]

            expected = detect(baseline)
            if detect(candidate) != expected:
                raise AssertionError(f'Gold detector mismatch at {item}')
            record['gold_detector'] = {
                'detections': len(expected),
                'timing_ms': measure_pair('gold_detector', {
                    'baseline': lambda: detect(baseline),
                    'candidate': lambda: detect(candidate)}, expected, item)}
            if profiler is not None:
                holds.connected_components = candidate
                profiler.enable()
                for _ in range(args.profile_repeats):
                    holds.detect_hold_tails(frame, calibration, config)
                    for lane in range(calibration.lane_count):
                        ribbon_at_judgement(frame, calibration, lane)
                profiler.disable()
            rows.append(record)
            print(json.dumps({'video': item['video_index'], 'time': item['time'],
                              'gold_cc': record['gold_cc']['timing_ms'],
                              'ordinary_cc': record['ordinary_cc']['timing_ms'],
                              'gold_detector': record['gold_detector']['timing_ms']},
                             ensure_ascii=False), flush=True)
    finally:
        holds.connected_components = old_hold_cc
    report = {
        'baseline_component': str(args.baseline_component.resolve()),
        'baseline_component_sha256': hashlib.sha256(args.baseline_component.read_bytes()).hexdigest(),
        'candidate_component_sha256': hashlib.sha256((APP_ROOT/'agent/music/components.py').read_bytes()).hexdigest(),
        'shared_detector_sha256': hashlib.sha256((APP_ROOT/'agent/music/holds.py').read_bytes()).hexdigest(),
        'fixtures': str(args.fixtures.resolve()), 'repeats_per_frame': args.repeats,
        'frames': rows,
        'aggregate_timing_ms': {name: {variant: summary(samples) for variant, samples in variants.items()}
                                for name, variants in timings.items()},
        'output_equivalence': 'Every timed output is identical, including component order, bbox, pixels, and full gold detection dataclasses.',
        'limitations': '22 compressed area-scaled fixed video frames, NumPy-only. No real Maa input or game Bad/Miss inference. Only CC primitive differs between paired measurements.'}
    if profiler is not None:
        stats = pstats.Stats(profiler)
        report['profile'] = {
            'repeats_per_frame': args.profile_repeats,
            'functions_by_cumulative_seconds': [
                {'file': key[0], 'line': key[1], 'function': key[2],
                 'primitive_calls': value[0], 'total_calls': value[1],
                 'self_seconds': value[2], 'cumulative_seconds': value[3]}
                for key, value in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:50]],
            'limitations': 'Instrumented profile includes call-hook overhead; not paired latency admission data.'}
        profiler.dump_stats(str(args.output.with_suffix('.pstats')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report['aggregate_timing_ms'], ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
