"""Paired real gold-mask CC/physical-detector benchmark, without a controller.

Decode all inputs before measuring so FFmpeg cannot compete with either side.
The same current detector/thresholds are shared; only the component primitive
changes. Reports are hotspot evidence, not full-loop or game-grade evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

APP_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = APP_ROOT.parent/'.work/round3-implementation-20261003'


def optimization_gate(summary, baseline_hash, candidate_hash, unchanged):
    """Never accept a noisy self-comparison as an optimization."""
    cc = summary['gold_cc']['wall_ms']
    return {
        'different_source': baseline_hash != candidate_hash,
        'source_verified_unchanged': unchanged,
        'gold_cc_p50_lower': cc['candidate']['p50'] < cc['baseline']['p50'],
        'gold_cc_p95_lower': cc['candidate']['p95'] < cc['baseline']['p95'],
        'detector_p95_increment_within_2ms': (
            summary['gold_detector']['wall_ms']['candidate']['p95']
            - summary['gold_detector']['wall_ms']['baseline']['p95'] <= 2.),
    }


def load_component(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.connected_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-component', type=Path, default=
        WORK_ROOT/'performance-reference-fast-ba8da1c2/agent/music/components.py')
    parser.add_argument('--candidate-component', type=Path, default=
        APP_ROOT/'agent/music/components.py')
    parser.add_argument('--fixtures', type=Path, default=WORK_ROOT/'gold-fixtures.json')
    parser.add_argument('--clip', default='breakthrough-dual-flick',
                        help='Additional full fixed-PTS gold-mask interval; empty disables it')
    parser.add_argument('--repeats', type=int, default=25)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    if not args.output.resolve().is_relative_to((APP_ROOT/'temp').resolve()):
        parser.error('Output must remain in app/temp')
    sys.path.insert(0, str(APP_ROOT))
    sys.path.insert(0, str(APP_ROOT/'tools'))
    import numpy as np
    from tap_replay import video_frames, load_replay_config, replay_identity
    from workspace_paths import ffmpeg_binary
    from agent.music import holds
    from agent.music.gold_mask import build_gold_mask
    from agent.music.models import MusicCalibrationData, MusicConfig
    from agent.music.storage import metric_summary

    source_paths = {
        'baseline_components': args.baseline_component,
        'candidate_components': args.candidate_component,
        'shared_holds': APP_ROOT/'agent/music/holds.py',
        'shared_topology': APP_ROOT/'agent/music/hold_topology.py',
        'shared_gold_mask': APP_ROOT/'agent/music/gold_mask.py',
    }
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in source_paths.items()}
    baseline = load_component(args.baseline_component, '_cc_baseline_reference')
    candidate = load_component(args.candidate_component, '_cc_candidate')
    package = WORK_ROOT/'baseline/candidate-state'
    calibration_path = package/'user-data/calibration/music.json'
    calibration = MusicCalibrationData(**json.loads(calibration_path.read_text())
                                       ['profiles']['7@1280x720'])
    trace_path = package/'logs/tap-traces/20261003T114448Z-63e80544.jsonl'
    config = MusicConfig(**load_replay_config(trace_path))
    minimum = max(3, config.hold_tail_min_pixels//9)
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    ffmpeg, ffprobe = ffmpeg_binary('ffmpeg.exe'), ffmpeg_binary('ffprobe.exe')
    frames, masks, rows, samples = [], [], [], {}

    def gold_mask(image):
        height, width = image.shape[:2]
        x, y, w, h = calibration.candidate_roi
        mask = build_gold_mask(image[max(0, y):min(height, y+h):3,
                                     max(0, x):min(width, x+w):3])
        mask.flags.writeable = False
        return mask

    # Complete every decoder before any timed pair. Retain only the small
    # sampled masks for the additional 240-frame hotspot interval.
    for item in json.loads(args.fixtures.read_text()):
        video = APP_ROOT.parent/f'test-materials/music/20261003test3-{item["video_index"]}.mp4'
        raw = subprocess.check_output([
            str(ffmpeg), '-v', 'error', '-ss', str(item['time']), '-i', str(video),
            '-frames:v', '1', '-vf', 'scale=1280:720:flags=area',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1'], creationflags=flags)
        if len(raw) != 1280*720*3:
            raise RuntimeError(f'Incomplete video frame: {item}')
        image = np.frombuffer(raw, np.uint8).reshape(720, 1280, 3)
        mask = gold_mask(image)
        record = {**item, 'frame_sha256': hashlib.sha256(raw).hexdigest(),
                  'mask_sha256': hashlib.sha256(mask).hexdigest(),
                  'mask_shape': list(mask.shape), 'pixels': int(mask.sum()),
                  'source': 'fixed-single-frame'}
        frames.append((record, image))
        masks.append((record, mask))
    if args.clip:
        inventory = json.loads((WORK_ROOT/'clip-fixtures.json').read_text())
        selected = [item for item in inventory if item['name'] == args.clip]
        if len(selected) != 1:
            parser.error('Unknown or duplicate clip')
        item = selected[0]
        options = SimpleNamespace(
            video=APP_ROOT.parent/f'test-materials/music/20261003test3-{item["video_index"]}.mp4',
            start=item['start'], duration=item['end']-item['start'], ffmpeg=ffmpeg, ffprobe=ffprobe)
        for sequence, (pts, image, _) in enumerate(video_frames(options)):
            mask = gold_mask(image)
            masks.append(({'clip': args.clip, 'sequence': sequence, 'pts': pts,
                'frame_sha256': hashlib.sha256(image).hexdigest(),
                'mask_sha256': hashlib.sha256(mask).hexdigest(),
                'mask_shape': list(mask.shape), 'pixels': int(mask.sum()),
                'source': 'fixed-source-PTS'}, mask))

    def measure_pair(name, functions, expected, record):
        local = {clock: {side: [] for side in ('baseline', 'candidate', 'paired_delta')}
                 for clock in ('wall_ms', 'thread_cpu_ms')}
        for side in ('baseline', 'candidate'):
            for _ in range(3):
                if functions[side]() != expected:
                    raise AssertionError(f'{name} warmup output mismatch: {record}')
        for repeat in range(args.repeats):
            order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
            pair = {clock: {} for clock in local}
            for side in order:
                wall_start, cpu_start = time.perf_counter_ns(), time.thread_time_ns()
                actual = functions[side]()
                cpu_end, wall_end = time.thread_time_ns(), time.perf_counter_ns()
                if actual != expected:
                    raise AssertionError(f'{name} output mismatch: {record}, {side}')
                pair['wall_ms'][side] = (wall_end-wall_start)/1e6
                pair['thread_cpu_ms'][side] = (cpu_end-cpu_start)/1e6
            for clock in local:
                for side in ('baseline', 'candidate'):
                    local[clock][side].append(pair[clock][side])
                local[clock]['paired_delta'].append(pair[clock]['candidate']-pair[clock]['baseline'])
        for clock, variants in local.items():
            for side, values in variants.items():
                samples.setdefault(name, {}).setdefault(clock, {}).setdefault(side, []).extend(values)
        return {clock: {side: metric_summary(values) for side, values in variants.items()}
                for clock, variants in local.items()}

    for record, mask in masks:
        expected = baseline(mask, minimum)
        local = measure_pair('gold_cc', {'baseline': lambda: baseline(mask, minimum),
            'candidate': lambda: candidate(mask, minimum)}, expected, record)
        rows.append({**record, 'components': len(expected), 'gold_cc': local})
    saved_component = holds.connected_components
    try:
        for record, image in frames:
            def detect(component):
                holds.connected_components = component
                return holds.detect_hold_tails(image, calibration, config, physical_only=True)
            expected = detect(baseline)
            local = measure_pair('gold_detector', {'baseline': lambda: detect(baseline),
                'candidate': lambda: detect(candidate)}, expected, record)
            # Store full attributes once, outside timing; no input controller.
            rows.append({**record, 'physical_detections': [asdict(d) for d in expected],
                         'gold_detector': local})
    finally:
        holds.connected_components = saved_component
    unchanged = all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name]
                    for name, path in source_paths.items())
    aggregate = {name: {clock: {side: metric_summary(values) for side, values in variants.items()}
                           for clock, variants in clocks.items()}
                 for name, clocks in samples.items()}
    gate = optimization_gate(aggregate, hashes['baseline_components'],
                             hashes['candidate_components'], unchanged)
    report = {'schema': 1, 'mode': 'paired-current-gold-mask-component-hotspot',
        **replay_identity(config, calibration, str(trace_path)),
        'source_paths': {name: str(path.resolve()) for name, path in source_paths.items()},
        'source_hashes': hashes, 'repeats_per_frame': args.repeats,
        'fixed_detector_frames': len(frames), 'gold_mask_frames': len(masks),
        'aggregate_timing': aggregate, 'optimization_gate': gate,
        'admission_passed': all(gate.values()), 'rows': rows,
        'output_equivalence': 'Every timed ordered bbox/pixel component and all physical detector attributes exact.',
        'limitations': 'Compressed area-scaled video, NumPy masks only. All decode finished before measurement. Wall time includes scheduling; thread CPU excludes preemption. Warmed alternating same-mask/function pairs, same current detector/geometry/thresholds, only CC differs. Full-loop scheduling, Maa transport, game Bad/Miss and FC are not validated.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'aggregate_timing': aggregate, 'optimization_gate': gate,
                      'admission_passed': report['admission_passed']}, ensure_ascii=False), flush=True)
    return 0 if report['admission_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
