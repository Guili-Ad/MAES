"""Run the frozen hold tests against either import root and capture contracts."""
import argparse
import functools
import json
import logging
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / 'temp'):
        parser.error('Output must stay under Double/temp')
    # The selected hold test definitions are unchanged from the checkpoint.
    # Run each root in a separate process to avoid cross-branch import caches.
    sys.path.insert(0, str(args.branch_root))
    sys.path.insert(0, str(args.branch_root / 'tests'))
    import test_longtap_branch as cases
    from agent.music.tracking import MusicVisionEngine
    logging.disable(logging.CRITICAL)
    records = []
    active = ['']
    for name in ('update', 'release_events', 'refine_pending'):
        original = getattr(MusicVisionEngine, name)
        def make_wrapper(method, name):
            @functools.wraps(method)
            def wrapper(*args, **kwargs):
                result = method(*args, **kwargs)
                for event in result:
                    if event.gesture.value.startswith('Hold'):
                        records.append({'test': active[0], 'method': name, 'track': event.track_id,
                                        'gesture': event.gesture.value, 'lane': event.lane,
                                        'deadline': round(event.deadline, 9), 'coordinate': event.coordinate})
                return result
            return wrapper
        setattr(MusicVisionEngine, name, make_wrapper(original, name))
    names = [name for name in unittest.defaultTestLoader.getTestCaseNames(cases.LongTapBranchTests)
             if any(word in name for word in ('hold', 'tail', 'ribbon', 'route', 'sustain', 'cap_locked', 'terminal_target'))]
    result = unittest.TestResult()
    for name in names:
        active[0] = name
        cases.LongTapBranchTests(name).run(result)
    payload = {'branch': str(args.branch_root), 'tests': names, 'records': records,
               'errors': [(str(case), error) for case, error in result.errors + result.failures]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'tests': len(names), 'records': len(records), 'errors': len(payload['errors'])}))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
