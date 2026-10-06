"""Local Test3 clip inventory and production-loop comparison; no controller.

Song/timestamp metadata stays in local validation reports, never recognition
resources. The fixed clips include leading gameplay so hold ownership is not
started from a tail-only screenshot. NumPy on compressed recordings plus mock
input/OCR does not measure game Bad/Miss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

APP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = APP_ROOT.parent
WORK_ROOT = WORKSPACE / '.work/round3-implementation-20261003'
EXPECTED_CONFIG = '6200cae263aa15a7'
EXPECTED_CALIBRATION = '20bfdc72be93e1b61d5d2d8803ca39562ac72d77b55d9c6b0be31a6dd3371714'

def read_header(path: Path) -> dict:
    with path.open(encoding='utf-8-sig') as stream:
        return json.loads(next(stream))


def source_fingerprint(branch: Path) -> dict:
    files = {str(path.relative_to(branch)):hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted((branch/'agent').rglob('*.py'))}
    return {'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
            'files':files}


def action_summary(report: dict) -> dict:
    receipts = []
    by_origin = {}
    for row in report.get('trace', []):
        receipt = row.get('receipt')
        if row.get('kind') == 'input' and isinstance(receipt, dict):
            receipts.append(receipt)
            origin = row.get('origin', 'tap')
            by_origin[origin] = by_origin.get(origin, 0) + 1
    keys = [(r.get('segment_id', 0), r.get('event_id')) for r in receipts]
    return {
        'frames': report['frames'], 'captures': report.get('captures'),
        'heads': len(report['heads']), 'actions': len(report['actions']),
        'input_receipts': len(receipts),
        'input_receipts_by_origin': by_origin,
        'input_down_attempted': sum(r.get('down_call_started') is not None for r in receipts),
        'input_down_completed': sum(r.get('down_call_finished') is not None for r in receipts),
        'input_up_completed': sum(r.get('up_call_finished') is not None for r in receipts),
        'duplicate_receipt_event_ids': len(keys) - len(set(keys)),
        'physical_note_duplicates': 'unknown: unique event IDs do not prove distinct physical notes',
        'input_errors': [r for r in receipts if r.get('error')],
        'stop': report.get('stop'), 'stage_costs': report.get('metrics_ms'),
        'tracking_cpu_ms': report.get('timing_ms'),
        'full_tracking_cpu_ms': report.get('full_tracking_timing_ms'),
        'identity_prepass_cpu_ms': report.get('identity_prepass_timing_ms'),
        'cpu_timing_notes': report.get('cpu_timing_notes', 'Old report: timing_ms omits external gold prepass CPU'),
        'critical_trace_drops': 'production trace header drop counters are not exported by the old loop helper',
        'game_bad_miss': 'unavailable: no game/controller input, compressed NumPy frames and simulated OCR',
    }


def compare_actions(baseline: dict, candidate: dict) -> dict:
    # Actions lack musical identity. Show raw transport deltas and head rows,
    # not a fabricated per-song grade or physical-note duplicate estimate.
    first, second = baseline['heads'], candidate['heads']
    common = min(len(first), len(second))
    changes = []
    for index in range(common):
        left, right = first[index], second[index]
        differences = {key: {'baseline': left.get(key), 'candidate': right.get(key)}
                       for key in ('lane', 'gesture', 'time') if left.get(key) != right.get(key)}
        if differences:
            changes.append({'ordinal_index': index, 'differences': differences})
    return {'baseline_heads': len(first), 'candidate_heads': len(second),
            'baseline_actions': len(baseline['actions']), 'candidate_actions': len(candidate['actions']),
            'ordinal_head_changes': changes,
            'warning': 'Ordinal alignment is diagnostic only after an inserted/lost action; not musical identity or game grade.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-root', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--clips', nargs='*', help='Names from the inventory; default all')
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--compare-label')
    parser.add_argument('--inventory', type=Path, default=WORK_ROOT / 'clip-fixtures.json',
                        help='Local video/run/start/end/coverage fixtures; never runtime resources')
    args = parser.parse_args()
    if not args.label.replace('-', '').replace('_', '').isalnum():
        parser.error('Label must use letters, digits, hyphens or underscores')
    branch = args.branch_root.resolve()
    if not (branch / 'agent/music/tracking.py').is_file():
        parser.error('Selected source root is missing production code')
    fixture_items = json.loads(args.inventory.read_text(encoding='utf-8'))
    clips = [(item['name'], item['video_index'], item['run'], item['start'], item['end'], item['coverage'])
             for item in fixture_items]
    chosen = [entry for entry in clips if not args.clips or entry[0] in args.clips]
    if args.clips and {entry[0] for entry in chosen} != set(args.clips):
        parser.error('Unknown clip selection')
    package = WORK_ROOT / 'baseline/candidate-state'
    calibration = package / 'user-data/calibration/music.json'
    output = APP_ROOT / 'temp/round3-implementation' / args.label
    persistent = WORK_ROOT / 'reports' / args.label
    output.mkdir(parents=True, exist_ok=True)
    persistent.mkdir(parents=True, exist_ok=True)
    inventory = []
    for name, video_index, run, start, end, coverage in clips:
        video = WORKSPACE / f'test-materials/music/20261003test3-{video_index}.mp4'
        config = package / f'logs/tap-traces/{run}.jsonl'
        if not video.is_file() or not config.is_file() or not calibration.is_file():
            raise FileNotFoundError(f'Missing fixture input for {name}')
        header = read_header(config)
        if header['config_hash'] != EXPECTED_CONFIG or header['effective_calibration_hash'] != EXPECTED_CALIBRATION:
            raise ValueError(f'Live configuration/calibration mismatch for {name}')
        # Use the same measured live stage median as a fixed injection profile
        # in both branches. This does not represent all real latency outliers.
        costs = header['summary']['metrics_ms']['whole_run']
        profile = {'default': {f'{field}_ms': costs[source]['p50_upper_ms']
                              for field, source in [('capture', 'capture'), ('provider', 'provider'),
                                                    ('mask', 'mask'), ('tracking', 'tracking'),
                                                    ('action', 'action'), ('ocr', 'ocr')]},
                   'provenance': f'{run}: whole-run 1ms histogram upper median',
                   'limitations': 'Fixed medians, not replayed live latency; real classifier CPU separately measured.'}
        item = {'name': name, 'video': str(video), 'run': run, 'config': str(config),
                'calibration': str(calibration), 'config_hash': EXPECTED_CONFIG,
                'effective_calibration_hash': EXPECTED_CALIBRATION,
                'start': start, 'end': end, 'duration': end-start,
                'coverage': coverage, 'cost_profile': profile,
                'judgement_labels': 'Prior read-only video review; no grade inferred from replay.'}
        inventory.append(item)
    (output / 'inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(output / 'inventory.json', persistent / 'inventory.json')
    if args.inventory_only:
        print(json.dumps({'inventory': str(persistent / 'inventory.json'), 'clips': len(inventory)}))
        return 0
    summary_path = output / 'summary.json'
    summaries = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else []
    for item in inventory:
        if item['name'] not in {entry[0] for entry in chosen}:
            continue
        name = item['name']
        profile = output / f'{name}-costs.json'
        profile.write_text(json.dumps(item['cost_profile'], indent=2), encoding='utf-8')
        report_path = output / f'{name}.json'
        if report_path.exists():
            raise FileExistsError(f'Report already exists; use another label: {report_path}')
        command = [sys.executable, '-B', str(APP_ROOT / 'tools/tap_replay.py'),
                   '--branch-root', str(branch), '--video', item['video'],
                   '--calibration', item['calibration'], '--config', item['config'],
                   '--start', str(item['start']), '--duration', str(item['duration']),
                   '--loop', '--trace-observations', '--cost-profile', str(profile),
                   '--output', str(report_path)]
        started = time.perf_counter()
        source_before = source_fingerprint(branch)
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        result = subprocess.run(command, text=True, capture_output=True, creationflags=flags)
        source_after = source_fingerprint(branch)
        (output / f'{name}-command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
        (output / f'{name}-stdout.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'Replay failed for {name}: {result.stdout[-2000:]} {result.stderr[-2000:]}')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report['config_hash'] != EXPECTED_CONFIG or report['effective_calibration_hash'] != EXPECTED_CALIBRATION:
            raise ValueError(f'Replay silently changed identity: {name}')
        summary = {'name': name, 'branch': str(branch), 'duration_wall_seconds': time.perf_counter()-started,
                   'source_before':source_before,'source_after':source_after,
                   'source_changed_during_replay':source_before['sha256']!=source_after['sha256'],
                   **action_summary(report)}
        if args.compare_label:
            old_path = APP_ROOT / 'temp/round3-implementation' / args.compare_label / f'{name}.json'
            summary['comparison'] = compare_actions(json.loads(old_path.read_text(encoding='utf-8')), report)
        summaries.append(summary)
        for file in (report_path, profile, output / f'{name}-command.json', output / f'{name}-stdout.txt'):
            shutil.copy2(file, persistent / file.name)
        (output / 'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
        shutil.copy2(output / 'summary.json', persistent / 'summary.json')
        print(json.dumps({'clip': name, 'frames': summary['frames'], 'heads': summary['heads'],
                          'duplicate_event_ids': summary['duplicate_receipt_event_ids'],
                          'input_errors': len(summary['input_errors']),
                          'wall_seconds': round(summary['duration_wall_seconds'], 2)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
