"""Explicit Oct5 fixtures: production-loop mock replay and fixed-PTS CPU.

Never connects a controller, modifies old packages, or infers game judgments.
All configuration, video and calibration paths are explicit in the inventory.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP/'tools'))
WORK = APP.parent/'.work/residual-fix-20261005'
RUNS = ['20261005T004545Z-74617701', '20261005T004849Z-9da9aa33',
        '20261005T005145Z-ccdf97ec', '20261005T005606Z-e2897f97',
        '20261005T012423Z-2e627a46', '20261005T012908Z-a53b1982',
        '20261005T013216Z-c9df8dd0', '20261005T013601Z-4ec1ecb4',
        '20261005T013902Z-4e162f5d', '20261005T014151Z-adcab443',
        '20261005T014449Z-fc276cb6', '20261005T014747Z-2df36c48']
CLIPS = [('center-01', 1, 33., 38.), ('center-04', 4, 95., 100.),
         ('yellow-05', 5, 111., 116.), ('false-flick-gold-10', 10, 55., 61.),
         ('stale-flick-yellow-12', 12, 98., 105.), ('dense-center-11', 11, 91., 97.),
         ('gold-flick-01', 1, 84., 91.), ('gold-flick-02', 2, 84., 91.),
         ('yellow-center-12', 12, 132., 137.), ('fc-protect-08', 8, 29., 36.)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def inventory():
    package = APP/'dist/MAES_PointIdentity_Candidate'
    rows = []
    for name, index, start, end in CLIPS:
        config = package/f'logs/tap-traces/{RUNS[index-1]}.jsonl'
        with config.open(encoding='utf-8-sig') as stream:
            header = json.loads(next(stream))
        whole = header['summary']['metrics_ms']['whole_run']
        rows.append(dict(name=name, video=str(APP.parent/f'test-materials/music/20261005test1-{index}.mp4'),
            config=str(config), calibration=str(WORK/'baseline/candidate-state/calibration/music.json'),
            start=start, end=end, config_hash=header['config_hash'],
            effective_calibration_hash=header['effective_calibration_hash'], run=RUNS[index-1],
            cost_profile={'default': {f'{field}_ms': whole[source]['p50_upper_ms'] for field, source in
                [('capture', 'capture'), ('provider', 'provider'), ('mask', 'mask'),
                 ('tracking', 'tracking'), ('action', 'action'), ('ocr', 'ocr')]},
                'provenance': 'Recorded whole-run histogram medians; not original per-frame jitter.'}))
    path = WORK/'fixtures/clips.json'
    if path.exists():
        raise FileExistsError('Preserve inventory; do not overwrite recorded inputs.')
    write(path, rows)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--branch-root', type=Path)
    parser.add_argument('--label')
    parser.add_argument('--clips', nargs='*')
    parser.add_argument('--full-prelude', action='store_true',
        help='Decode from video PTS zero through the target end; preserve all preceding heads/relationships')
    args = parser.parse_args()
    if args.prepare:
        print(inventory()); return 0
    if not args.branch_root or not (args.branch_root/'agent').is_dir() or not args.label:
        parser.error('Explicit branch root and unique label required.')
    if not args.label.replace('-', '').replace('_', '').isalnum():
        parser.error('Invalid label.')
    rows = read(WORK/'fixtures/clips.json')
    if args.clips and not set(args.clips) <= {r['name'] for r in rows}:
        parser.error('Unknown clip.')
    output = WORK/'loop-reports'/args.label
    if output.exists():
        raise FileExistsError('Preserve prior reports; select a new label.')
    output.mkdir(parents=True)
    transient = APP/'temp/residual-validation'/args.label
    if transient.exists():
        raise FileExistsError('Preserve transient replay evidence.')
    transient.mkdir(parents=True)
    fingerprints = {str(p.relative_to(args.branch_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((args.branch_root/'agent').rglob('*.py'))}
    summary = []
    from workspace_paths import ffmpeg_binary
    for row in rows:
        if args.clips and row['name'] not in args.clips:
            continue
        target = transient/f'{row["name"]}.json'
        cost = output/f'{row["name"]}-cost.json'
        write(cost, row['cost_profile'])
        command = [sys.executable, '-B', str(APP/'tools/tap_replay.py'), '--loop',
            '--branch-root', str(args.branch_root.resolve()), '--calibration', row['calibration'],
            '--config', row['config'], '--video', row['video'],
            '--start', str(0. if args.full_prelude else row['start']),
            '--duration', str(row['end'] if args.full_prelude else row['end']-row['start']), '--output', str(target),
            '--cost-profile', str(cost), '--ffmpeg', str(ffmpeg_binary('ffmpeg.exe')),
            '--ffprobe', str(ffmpeg_binary('ffprobe.exe'))]
        result = subprocess.run(command, capture_output=True, text=True)
        (output/f'{row["name"]}-console.txt').write_text(result.stdout+result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'Replay failed {row["name"]}: {result.stderr[-1000:]}')
        report = read(target)
        shutil.copy2(target, output/target.name)
        if report['config_hash'] != row['config_hash'] or report['effective_calibration_hash'] != row['effective_calibration_hash']:
            raise ValueError('Config/calibration changed.')
        inputs = [r for r in report['trace'] if r['kind'] == 'input' and isinstance(r.get('receipt'), dict)]
        keys = [(r.get('segment_id'), r['receipt'].get('event_id')) for r in inputs]
        window = [r for r in inputs if row['start'] <= (r['receipt'].get('down_call_started') or -1) < row['end']]
        stats = dict(name=row['name'], frames=report['frames'], captures=report.get('captures'),
            full_prelude=args.full_prelude, target_start=row['start'], target_end=row['end'],
            inputs=len(inputs), origins=dict(Counter(r.get('origin') for r in inputs)),
            target_inputs=len(window), target_origins=dict(Counter(r.get('origin') for r in window)),
            duplicate_receipts=len(keys)-len(set(keys)),
            input_errors=sum(bool(r['receipt'].get('error')) for r in inputs),
            recoveries=sum(r['kind']=='tap_mask_recovered' for r in report['trace']),
            target_recoveries=sum(r['kind']=='tap_mask_recovered' and row['start'] <= r.get('time', -1) < row['end']
                                 for r in report['trace']),
            flick_rejections=dict(Counter(r.get('reason') for r in report['trace'] if r['kind']=='flick_qualification')),
            target_flick_rejections=dict(Counter(r.get('reason') for r in report['trace']
                if r['kind']=='flick_qualification' and row['start'] <= r.get('time', -1) < row['end'])),
            stop=report.get('stop'), report=str(target))
        summary.append(stats)
        print(json.dumps(stats, ensure_ascii=False), flush=True)
        write(output/'summary.json', dict(source_files=fingerprints, clips=summary, full_prelude=args.full_prelude,
            limitations='Compressed NumPy visual replay, real runtime loop with fixed simulated median costs; no live input or game grades. Event ID uniqueness cannot prove physical uniqueness.'))
    after = {str(p.relative_to(args.branch_root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((args.branch_root/'agent').rglob('*.py'))}
    if after != fingerprints:
        raise RuntimeError('Source changed during replay; evidence invalid.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
