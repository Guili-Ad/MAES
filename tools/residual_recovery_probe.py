"""Compare recovery pixel results and function costs on fixed original PTS.

Instrumented timings are NOT admission evidence. No controller is connected.
Reference and candidate are counterbalanced within each immutable frame;
shared same-frame contour caches can still affect individual timings.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import point_strict_pts as strict

ROWS = []


def install(reference_path):
    import agent.music.head_recovery as candidate
    spec = importlib.util.spec_from_file_location('agent.music._residual_recovery_reference', reference_path)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    original = candidate.recover_point_heads

    def recovery(tracks, frame, calibration, project, *, entries=None, trace=None):
        results, timing = {}, {}
        order = ('reference', 'candidate') if len(ROWS) % 2 == 0 else ('candidate', 'reference')
        for name in order:
            method = reference.recover_point_heads if name == 'reference' else original
            start = time.perf_counter()
            result = method(tracks, frame, calibration, project, entries=entries, trace=None)
            timing[name] = (time.perf_counter()-start)*1000.
            results[name] = result
        def identity(result):
            return {tid: (asdict(c), asdict(p)) for tid, (c, p) in result.items()}
        if identity(results['candidate']) != identity(results['reference']):
            raise AssertionError(f'Pixel recovery changed at PTS {frame.midpoint} sequence {frame.sequence}')
        ROWS.append({'pts': frame.midpoint, 'sequence': frame.sequence,
            'order': order, 'equal': True, 'owners': sorted(results['candidate']), 'timing_ms': timing})
        # Search diagnostics are intentionally excluded in this probe; no
        # measured frame is used to claim production-loop/trace equivalence.
        return results['candidate']
    candidate.recover_point_heads = recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--branch-root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--clips', nargs='+', required=True)
    args = parser.parse_args()
    if not args.reference.is_file():
        parser.error('Explicit preserved reference required')
    source = strict.prepare(args.work_root.resolve(), args.inventory.resolve())
    source = strict.substitute(source, '    logging.disable(logging.CRITICAL)',
        '    logging.disable(logging.CRITICAL)\n    _install_recovery_probe(_reference_path)')
    source = strict.substitute(source, "'mode':'point-strict-source-PTS-qualification-wall'",
                              "'mode':'developer-recovery-equivalence-NOT-ADMISSION'")
    source = strict.substitute(source, "            'inventory_hash':inventory_hash,",
        "            'developer_probe_rows': list(_probe_rows),\n"
        "            'reference_hash':_reference_hash,\n"
        "            'developer_limitations':'Counterbalanced old/new calls; instrumented frame times invalid for admission.',\n"
        "            'inventory_hash':inventory_hash,")
    argv = sys.argv
    sys.argv = [str(strict.HERE), '--branch-root', str(args.branch_root.resolve()), '--label', args.label,
                '--clips', *args.clips]
    namespace = {'__file__': str(strict.HERE), '__name__': 'residual_recovery_probe_implementation',
        'TEMPLATE_PATH': strict.TEMPLATE, '_install_recovery_probe': install,
        '_reference_path': args.reference.resolve(), '_probe_rows': ROWS,
        '_reference_hash': hashlib.sha256(args.reference.read_bytes()).hexdigest()}
    try:
        exec(compile(source, str(strict.HERE), 'exec'), namespace)
        return namespace['main']()
    finally:
        sys.argv = argv


if __name__ == '__main__':
    raise SystemExit(main())
