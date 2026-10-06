"""Counterbalanced adjacent-frame wall-cost comparison, no controller.

Two independent module graphs, engines, clocks and queues consume the same
immutable decoded frames. Switch the active agent graph outside measurement;
alternate which implementation processes each frame first. This controls slow
host-load drift without changing power settings, user processes or timings.
It is still a fixed-PTS mock-transport fixture, not game-grade evidence.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import point_strict_pts as strict


def agent_modules():
    return {k: v for k, v in sys.modules.items() if k == 'agent' or k.startswith('agent.')}


def activate(state, paths):
    for name in list(agent_modules()):
        del sys.modules[name]
    sys.modules.update(state)
    sys.path[:] = paths


def prepare(work, inventory):
    source = strict.prepare(work, inventory)
    source = strict.substitute(source, '        decoded = list(video_frames(options))',
        '        decoded = _paired_decode(options, name, video_frames)')
    source = strict.substitute(source, '        for sequence,(pts,image,unused) in enumerate(decoded):',
        "        yield ('ready', name, len(decoded))\n"
        '        for sequence,(pts,image,unused) in enumerate(decoded):')
    source = strict.substitute(source, "                'candidates':len(candidates),'pending':len(pending)})",
        "                'candidates':len(candidates),'pending':len(pending)})\n"
        "            yield ('frame', sequence, pts)")
    source = strict.substitute(source,
        "(Path(__file__),TEMPLATE_PATH,APP_ROOT/'tools/tap_replay.py',APP_ROOT/'tools/workspace_paths.py')",
        "(Path(__file__),TEMPLATE_PATH,PAIR_TOOL_PATH,APP_ROOT/'tools/tap_replay.py',APP_ROOT/'tools/workspace_paths.py')",
        count=2)
    source = strict.substitute(source, "'version':3,'source_clock':", "'version':4,'source_clock':")
    source = strict.substitute(source, "'primary_timer':'perf_counter elapsed wall seconds; not process CPU',",
        "'primary_timer':'perf_counter elapsed wall seconds; not process CPU',\n"
        "    'pairing':'Same process, adjacent same-PTS frames, independent module graphs; order alternates each frame; switching outside timers',")
    return compile(source, str(strict.HERE), 'exec')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--baseline-label', required=True)
    parser.add_argument('--candidate-label', required=True)
    parser.add_argument('--clips', nargs='*')
    args = parser.parse_args()
    roots = [args.baseline_root.resolve(), args.candidate_root.resolve()]
    labels = [args.baseline_label, args.candidate_label]
    if roots[0] == roots[1] or labels[0] == labels[1]:
        parser.error('Independent baseline and candidate required')
    if any(not (r/'agent').is_dir() for r in roots):
        parser.error('Both source roots must contain agent')
    if any(not label.replace('-', '').replace('_', '').isalnum() for label in labels):
        parser.error('Invalid report label')
    work, inventory = args.work_root.resolve(), args.inventory.resolve()
    fixtures = json.loads(inventory.read_text(encoding='utf-8'))
    chosen = [r for r in fixtures if not args.clips or r['name'] in args.clips]
    if not chosen or (args.clips and set(args.clips) != {r['name'] for r in chosen}):
        parser.error('Unknown or empty fixture selection')
    for label in labels:
        if (work/'reports'/label).exists() or (work/'strict-source-pts'/label).exists():
            raise FileExistsError('Preserve previous evidence; use fresh labels')
    code = prepare(work, inventory)
    original_paths, original_argv, original_modules = list(sys.path), sys.argv, agent_modules()
    try:
        for fixture in chosen:
            decoded = {}
            def decode(options, name, method):
                if name not in decoded:
                    decoded[name] = list(method(options))
                    for _, image, _ in decoded[name]:
                        image.setflags(write=False)
                return decoded[name]
            states, paths, generators = [], [], []
            ready = []
            for index, (root, label) in enumerate(zip(roots, labels)):
                path = [str(root), *[p for p in original_paths if p not in {str(r) for r in roots}]]
                activate({}, path)
                namespace = {'__file__': str(strict.HERE), '__name__': 'paired_pts_implementation',
                    'TEMPLATE_PATH': strict.TEMPLATE, 'PAIR_TOOL_PATH': Path(__file__).resolve(),
                    '_paired_decode': decode}
                sys.argv = [str(strict.HERE), '--branch-root', str(root), '--label', label,
                            '--clips', fixture['name']]
                if index:
                    sys.argv.extend(['--compare-label', labels[0]])
                exec(code, namespace)
                generator = namespace['main']()
                ready.append(next(generator))
                states.append(agent_modules()); paths.append(list(sys.path)); generators.append(generator)
                imported = Path(states[-1]['agent'].__file__).resolve()
                if not imported.is_relative_to(root):
                    raise RuntimeError(f'Wrong source graph loaded: {imported}')
            if ready[0] != ready[1] or ready[0][0] != 'ready':
                raise RuntimeError('Decoded frame inventory differs')
            for sequence in range(ready[0][2]):
                pair = []
                for index in ((0, 1) if sequence % 2 == 0 else (1, 0)):
                    activate(states[index], paths[index])
                    pair.append(next(generators[index]))
                    states[index] = agent_modules()
                if pair[0] != pair[1] or pair[0][:2] != ('frame', sequence):
                    raise RuntimeError('Frame phases diverged')
            # Baseline writes its immutable report before candidate compares.
            for index in (0, 1):
                activate(states[index], paths[index])
                try:
                    next(generators[index])
                    raise RuntimeError('Unexpected extra frames')
                except StopIteration as finished:
                    if finished.value != 0:
                        raise RuntimeError('Benchmark failed')
            print(json.dumps({'paired_clip': fixture['name'], 'frames': ready[0][2],
                'counterbalanced': True, 'source_graphs_independent': True}), flush=True)
            del generators, states, decoded
    finally:
        activate(original_modules, original_paths)
        sys.argv = original_argv
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
