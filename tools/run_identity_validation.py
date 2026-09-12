"""Run local, user-recorded clip manifests without opening an input backend."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--baseline-root', type=Path, default=ROOT/'backup/hold-identity-v2')
    p.add_argument('--suffix', default=None)
    args = p.parse_args()
    manifest = args.manifest.resolve()
    if not manifest.is_relative_to(ROOT/'temp'):
        p.error('Song-specific manifests must remain under Double/temp')
    root = args.baseline_root.resolve() if args.baseline else ROOT
    suffix = args.suffix or ('old' if args.baseline else 'new')
    if not suffix.replace('-', '').replace('_', '').isalnum():
        p.error('Suffix must be a simple local label')
    def run(case):
        output = manifest.parent/(case['id']+'-'+suffix+'.json')
        cmd = [sys.executable, '-B', str(ROOT/'tools/tap_replay.py'), '--branch-root', str(root),
               '--video', case['video'], '--start', str(case['start']), '--duration', str(case['duration']),
               '--calibration', str(args.calibration.resolve()), '--output', str(output), '--trace-observations']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
        report = json.loads(output.read_text(encoding='utf-8'))
        print(case['id'], suffix, 'heads', len(report['heads']), 'frames', report['frames'], flush=True)
        return {'case': case['id'], 'report': str(output), 'heads': report['heads'],
                'timing': report['timing_ms'], 'note': 'Parallel clip timings are not a performance acceptance benchmark.'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(run,json.loads(manifest.read_text(encoding='utf-8'))))
    (manifest.parent/('summary-'+suffix+'.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__ == '__main__':
    main()
