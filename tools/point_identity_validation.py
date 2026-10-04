"""Explicit-path October 4 production-loop and identical-PTS CPU validation.

No controller is opened. Results are development evidence, not game grades.
The production loop uses the recorded run's whole-run stage histogram medians.
The separate fixed-PTS pass services every decoded frame in order, so CPU
comparisons must not conflate runtime screenshot phase changes with speedups.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

APP = Path(__file__).resolve().parents[1]
WORK = APP.parent / '.work/point-identity-20261004'
BASE_STATE = WORK / 'baseline/candidate-state'
CONFIG_HASH = '6200cae263aa15a7'
CALIBRATION_HASH = '20bfdc72be93e1b61d5d2d8803ca39562ac72d77b55d9c6b0be31a6dd3371714'
BIN = APP.parent / '.work/ffmpeg-7.1.1-extract/ffmpeg-7.1.1-essentials_build/bin'


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def header(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return json.loads(next(stream))


def fingerprint(root):
    values = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((root / 'agent').rglob('*.py'))}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def input_summary(report):
    inputs = [r for r in report.get('trace', []) if r.get('kind') == 'input' and isinstance(r.get('receipt'), dict)]
    keys = [(r.get('segment_id'), r['receipt'].get('event_id')) for r in inputs]
    point_inputs = [r for r in report.get('trace', []) if r.get('kind') == 'point_input' and r.get('started') is not None]
    return {'frames': report['frames'], 'captures': report.get('captures'),
            'head_count': len(report['heads']), 'action_count': len(report['actions']),
            'input_count': len(inputs), 'origins': dict(Counter(r.get('origin') for r in inputs)),
            'receipt_observability': 'production-loop trace' if report.get('mode') == 'production-loop' else 'fixed-PTS helper does not export runtime input receipts; use transport actions only',
            'duplicate_event_receipts': len(keys) - len(set(keys)),
            'errors': [r for r in inputs if r['receipt'].get('error')],
            'point_families': dict(Counter(r.get('family') for r in point_inputs)),
            'qualification_reasons': dict(Counter(r.get('reason') for r in report.get('trace', []) if r.get('kind') == 'point_qualification')),
            'tracking_cpu_ms': report.get('full_tracking_timing_ms', report.get('timing_ms')),
            'stop': report.get('stop'),
            'physical_duplicates': 'Unknown without physical visual annotations; unique event IDs are insufficient.'}


def fixed_pts_digest(report):
    times = [r['time'] for r in report.get('observations', [])]
    if len(times) != report['frames']:
        raise ValueError('Fixed-PTS pass did not record every decoded frame')
    return hashlib.sha256(json.dumps(times, separators=(',', ':')).encode()).hexdigest()


def compile_loop_reports(args, fixtures):
    """Summarize preserved JSON directly; never hand-copy event counts."""
    report_path = WORK / 'reports' / f'{args.label}.json'
    markdown_path = report_path.with_suffix('.md')
    if report_path.exists() or markdown_path.exists():
        raise FileExistsError('Preserve existing final summary')
    source_summaries = {}
    for label in args.compile_labels:
        source_summaries[label] = {c['name']: c for c in json_read(WORK / 'reports' / label / 'summary.json')['clips']}
    clips = []
    for fixture in fixtures:
        name = fixture['name']
        selected = [(label, values[name]) for label, values in source_summaries.items()
                    if name in values and 'loop' in values[name]['modes']]
        if len(selected) != 1:
            raise ValueError(f'Expected one completed candidate loop report for {name}')
        label, summary = selected[0]
        path = WORK / 'reports' / label / f'{name}-loop.json'
        report = json_read(path)
        trace = report.get('trace', [])
        baseline_paths = [WORK / 'reports' / label / f'{name}-loop.json' for label in args.compare_label or []]
        existing = [p for p in baseline_paths if p.is_file()]
        if len(existing) != 1:
            raise ValueError(f'Expected one baseline loop report for {name}')
        baseline = json_read(existing[0])
        before, after = input_summary(baseline), input_summary(report)
        conflicts = [r for r in trace if r.get('kind') == 'point_identity_sent_conflict']
        unowned = [{'event': r['event'], 'physical_id': r.get('physical_id'),
                    'raw_hit': r.get('raw_hit'), 'down_call_started': r['receipt'].get('down_call_started'),
                    'down_call_finished': r['receipt'].get('down_call_finished'),
                    'up_call_finished': r['receipt'].get('up_call_finished'),
                    'owner': r.get('owner')}
                   for r in trace if r.get('kind') == 'input' and r.get('origin') == 'hold_note' and r.get('owner') is None]
        clips.append({'name': name, 'video': fixture['video'], 'start': fixture['start'], 'end': fixture['end'],
                      'candidate_report': str(path), 'baseline_report': str(existing[0]),
                      'agent_fingerprint': summary['modes']['loop']['source_fingerprint'],
                      'config_hash': report['config_hash'], 'calibration_hash': report['calibration_hash'],
                      'effective_calibration_hash': report['effective_calibration_hash'],
                      'baseline': before, 'candidate': after,
                      'sent_identity_conflict_count': len(conflicts),
                      'sent_identity_conflicts': conflicts, 'owner_independent_gold_inputs': unowned,
                      'point_refresh_before_wait': sum(r.get('kind') == 'point_refresh_before_wait' for r in trace),
                      'qualification_reasons': after['qualification_reasons']})
    fingerprints = sorted({r['agent_fingerprint'] for r in clips})
    configs = sorted({r['config_hash'] for r in clips})
    calibrations = sorted({r['effective_calibration_hash'] for r in clips})
    if len(fingerprints) != 1 or configs != [CONFIG_HASH] or calibrations != [CALIBRATION_HASH]:
        raise ValueError('Mixed source/config/calibration in final loop evidence')
    totals = {}
    for side in ('baseline', 'candidate'):
        origins = Counter()
        for row in clips:
            origins.update(row[side]['origins'])
        totals[side] = {'decoded_frames': sum(r[side]['frames'] for r in clips),
                        'input_receipts': sum(r[side]['input_count'] for r in clips),
                        'head_count': sum(r[side]['head_count'] for r in clips),
                        'transport_actions': sum(r[side]['action_count'] for r in clips),
                        'origins': dict(origins),
                        'duplicate_event_receipts': sum(r[side]['duplicate_event_receipts'] for r in clips),
                        'error_count': sum(len(r[side]['errors']) for r in clips)}
    payload = {'schema': 1, 'complete_production_loop_clips': len(clips), 'agent_fingerprints': fingerprints,
               'config_hashes': configs, 'effective_calibration_hashes': calibrations,
               'metadata_note': 'interface.json UI version changed 1.0.4 to 1.0.5 during the validation interval; agent source and action configuration remained frozen. No unified build_id is claimed. Final package manifests require independent verification.',
               'totals': totals, 'sent_identity_conflicts': sum(r['sent_identity_conflict_count'] for r in clips),
               'clips': clips,
               'performance_status': 'First fixed-PTS pass failed freshness-v8 with +3.692 ms P95; preserved separately. No performance pass is claimed by this functional loop summary.',
               'limitations': ['No live controller/game judgment; NumPy on compressed videos and fixed live-stage median injection.',
                               'Different serviced capture phases are not pure CPU performance evidence.',
                               'Unique event receipts and zero conflict logs do not prove zero duplicated physical notes.',
                               'Selected clips bypass startup_gate, so cold-start/tamper guards require separate tests.',
                               'Counts and owner-independent inputs are execution evidence, not Perfect/Bad/Miss or FC.']}
    write_json(report_path, payload)
    lines = ['# Point Identity production-loop final summary', '',
             f"Agent source fingerprint: `{fingerprints[0]}`",
             f"Configuration: `{configs[0]}`; effective calibration: `{calibrations[0]}`.", '',
             payload['metadata_note'], '',
             '| Clip | Inputs baseline → candidate | Track baseline → candidate | Gold baseline → candidate | Flick baseline → candidate | Duplicate receipts | Sent identity conflicts | Simulated errors |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in clips:
        b, c = row['baseline'], row['candidate']
        pairs = [f"{b['origins'].get(key,0)} → {c['origins'].get(key,0)}" for key in ('track','hold_note','flick')]
        lines.append(f"| {row['name']} | {b['input_count']} → {c['input_count']} | {' | '.join(pairs)} | {c['duplicate_event_receipts']} | {row['sent_identity_conflict_count']} | {len(c['errors'])} |")
    lines += ['', '## Owner-independent execution evidence', '']
    for row in clips:
        for event in row['owner_independent_gold_inputs']:
            lines.append(f"- {row['name']}: `{event['event']}`, Down started {event['down_call_started']:.6f} s, owner unknown; Up completed {event['up_call_finished']:.6f} s.")
    lines += ['', '## Limits', '', payload['performance_status'], '']
    lines += [f'- {item}' for item in payload['limitations']]
    markdown_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(report_path), 'clips': len(clips), 'totals': totals}, ensure_ascii=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-root', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--inventory', type=Path, default=WORK / 'fixtures/clips.json')
    parser.add_argument('--clips', nargs='*')
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--fixed-pts-cpu', action='store_true')
    parser.add_argument('--only-fixed-pts', action='store_true', help='CPU rerun without a second runtime-phase replay')
    parser.add_argument('--cpu-limit-ms', type=float, default=2., help='Stop a compared fixed-PTS pass if the P95 increment exceeds this limit')
    parser.add_argument('--compare-label', action='append', help='Repeat for baseline labels covering disjoint clip sets')
    parser.add_argument('--compile-labels', nargs='+', help='Lightweight final JSON/Markdown summary from completed loop labels; no decode')
    parser.add_argument('--ffmpeg', type=Path, default=BIN / 'ffmpeg.exe')
    parser.add_argument('--ffprobe', type=Path, default=BIN / 'ffprobe.exe')
    args = parser.parse_args()
    if not args.label.replace('-', '').replace('_', '').isalnum():
        parser.error('Label must contain only letters, digits, hyphens or underscores')
    branch = args.branch_root.resolve()
    for path in (branch / 'agent/music/tracking.py', args.inventory, args.ffmpeg, args.ffprobe):
        if not path.is_file():
            raise FileNotFoundError(path)
    fixtures = json_read(args.inventory)
    names = [f['name'] for f in fixtures]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate clip names')
    if args.clips and not set(args.clips).issubset(names):
        parser.error('Unknown clip selection')
    if args.compile_labels:
        return compile_loop_reports(args, fixtures)
    calibration = BASE_STATE / 'user-data/calibration/music.json'
    if not calibration.is_file():
        raise FileNotFoundError(calibration)
    inventory = []
    for fixture in fixtures:
        video = Path(fixture['video']).resolve()
        config = BASE_STATE / f"logs/tap-traces/{fixture['run']}.jsonl"
        if not video.is_file() or not config.is_file():
            raise FileNotFoundError(f'Missing video or matching live trace for {fixture["name"]}')
        if video.name != f'20261004test1-{int(video.stem.rsplit("-", 1)[1])}.mp4':
            raise ValueError(f'Unexpected date or video family: {video}')
        live = header(config)
        if live['config_hash'] != CONFIG_HASH or live['effective_calibration_hash'] != CALIBRATION_HASH:
            raise ValueError(f'Live config/calibration mismatch: {fixture["name"]}')
        if fixture['end'] <= fixture['start']:
            raise ValueError('Invalid clip duration')
        costs = live['summary']['metrics_ms']['whole_run']
        profile = {'default': {f'{field}_ms': costs[source]['p50_upper_ms'] for field, source in
                   [('capture', 'capture'), ('provider', 'provider'), ('mask', 'mask'),
                    ('tracking', 'tracking'), ('action', 'action'), ('ocr', 'ocr')]},
                   'provenance': f'{fixture["run"]}: whole-run 1ms histogram upper median',
                   'limitations': 'Fixed simulated medians, not original live outliers. CPU measured separately.'}
        inventory.append({**fixture, 'video': str(video), 'video_bytes': video.stat().st_size,
                          'config': str(config), 'calibration': str(calibration),
                          'config_hash': CONFIG_HASH, 'effective_calibration_hash': CALIBRATION_HASH,
                          'cost_profile': profile})
    output = APP / 'temp/point-identity-validation' / args.label
    persistent = WORK / 'reports' / args.label
    if output.exists() or persistent.exists():
        raise FileExistsError('Preserve previous reports; select a new label')
    output.mkdir(parents=True)
    persistent.mkdir(parents=True)
    write_json(output / 'inventory.json', inventory)
    shutil.copy2(output / 'inventory.json', persistent / 'inventory.json')
    if args.inventory_only:
        print(json.dumps({'inventory': str(persistent / 'inventory.json'), 'clips': len(inventory)}))
        return 0
    summaries = []
    for item in inventory:
        if args.clips and item['name'] not in args.clips:
            continue
        name = item['name']
        profile = output / f'{name}-costs.json'
        write_json(profile, item['cost_profile'])
        modes = ['fixed-pts'] if args.only_fixed_pts else (['loop', 'fixed-pts'] if args.fixed_pts_cpu else ['loop'])
        summary = {'name': name, 'source': str(branch), 'coverage': item['coverage'], 'modes': {}}
        for mode in modes:
            report_path = output / f'{name}-{mode}.json'
            command = [sys.executable, '-B', str(APP / 'tools/tap_replay.py'),
                       '--branch-root', str(branch), '--video', item['video'],
                       '--calibration', str(calibration), '--config', item['config'],
                       '--start', str(item['start']), '--duration', str(item['end'] - item['start']),
                       '--trace-observations', '--ffmpeg', str(args.ffmpeg), '--ffprobe', str(args.ffprobe),
                       '--output', str(report_path)]
            if mode == 'loop':
                command += ['--loop', '--cost-profile', str(profile)]
            before = fingerprint(branch)
            begin = time.perf_counter()
            result = subprocess.run(command, capture_output=True, text=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            after = fingerprint(branch)
            write_json(output / f'{name}-{mode}-command.json', command)
            (output / f'{name}-{mode}-stdout.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
            if result.returncode:
                raise RuntimeError(f'Replay failed ({name}/{mode}): {result.stdout[-3000:]} {result.stderr[-3000:]}')
            if before != after:
                raise RuntimeError(f'Source changed while replaying {name}; mixed-phase evidence rejected')
            report = json_read(report_path)
            if report['config_hash'] != CONFIG_HASH or report['effective_calibration_hash'] != CALIBRATION_HASH:
                raise ValueError('Replay silently changed config/calibration')
            summary['modes'][mode] = {'source_fingerprint': before,
                                      'wall_seconds': time.perf_counter() - begin, **input_summary(report)}
            if mode == 'fixed-pts':
                summary['modes'][mode]['pts_digest'] = fixed_pts_digest(report)
                summary['modes'][mode]['cpu_notes'] = 'Identical decoded PTS are required for comparison. Actual engine/update/refine CPU; different behavior states are not a behavior-preserving performance replacement.'
            if args.compare_label:
                old_paths = [WORK / 'reports' / label / f'{name}-{mode}.json' for label in args.compare_label]
                available = [path for path in old_paths if path.is_file()]
                if len(available) != 1:
                    raise ValueError(f'Need exactly one baseline report for {name}/{mode}: {available}')
                old = json_read(available[0])
                comparison = {'head_count_before': len(old['heads']), 'head_count_after': len(report['heads']),
                              'baseline_report': str(available[0]),
                              'action_count_before': len(old['actions']), 'action_count_after': len(report['actions']),
                              'same_actions': old['actions'] == report['actions'],
                              'same_scheduled': old['scheduled'] == report['scheduled']}
                if mode == 'fixed-pts':
                    comparison['same_decoded_pts'] = fixed_pts_digest(old) == fixed_pts_digest(report)
                    comparison['tracking_p95_increment_ms'] = report['timing_ms']['p95'] - old['timing_ms']['p95']
                    comparison['cpu_p95_pass'] = comparison['tracking_p95_increment_ms'] <= args.cpu_limit_ms
                    if not comparison['same_decoded_pts']:
                        raise ValueError('CPU comparison uses different decoded PTS')
                else:
                    comparison['same_serviced_snapshots'] = old['observations'] == report['observations']
                    comparison['cpu_comparison_valid'] = comparison['same_serviced_snapshots']
                summary['modes'][mode]['comparison'] = comparison
                if mode == 'fixed-pts' and not comparison['cpu_p95_pass']:
                    summary['validation_stop'] = 'CPU P95 increment exceeds the per-clip limit; no subsequent clips run.'
                    for path in output.glob(f'{name}*'):
                        shutil.copy2(path, persistent / path.name)
                    write_json(output / 'summary.json', {'schema': 1, 'clips': summaries + [summary],
                               'stopped_on_cpu_regression': name})
                    shutil.copy2(output / 'summary.json', persistent / 'summary.json')
                    raise RuntimeError(f'CPU P95 increment exceeds {args.cpu_limit_ms} ms for {name}: {comparison["tracking_p95_increment_ms"]:.3f} ms')
            print(json.dumps({'clip': name, 'mode': mode, 'frames': report['frames'], 'heads': len(report['heads']),
                              'wall_seconds': round(time.perf_counter() - begin, 2)}, ensure_ascii=False), flush=True)
        for path in output.glob(f'{name}*'):
            shutil.copy2(path, persistent / path.name)
        summaries.append(summary)
        write_json(output / 'summary.json', {'schema': 1, 'clips': summaries,
                   'limitations': 'No game/controller; compressed NumPy video, synthetic costs/OCR. Event counts are not Bad/Miss or FC. Fixed-PTS CPU comparisons separate phase from runtime behavior.'})
        shutil.copy2(output / 'summary.json', persistent / 'summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
