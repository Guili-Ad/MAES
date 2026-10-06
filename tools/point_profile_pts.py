"""Developer-only, targeted cProfile over the unchanged strict PTS fixture.

The original strict tool and runtime files are never edited. A temporary
in-memory observer captures association graph/solver sizes; these instrumented
timings are diagnostic only and MUST NOT replace any admission measurement.
"""
from __future__ import annotations

import argparse
import ast
import cProfile
from dataclasses import asdict
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import pstats
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import point_strict_pts as strict


RECORDS = []
ACTIVE = None
INPUT_ROOT = None


def state_summary(engine, frame):
    chain = getattr(engine, 'tap_hold_chain', None)
    tracker = getattr(chain, 'tracker', None)
    markers = getattr(tracker, 'markers', {})
    rows = []
    for marker in markers.values():
        observations = list(marker.observations)
        last = observations[-1] if observations else None
        rows.append({'id': marker.marker_id, 'owner': marker.owner,
                     'observations': len(observations),
                     'lane': last.lane if last is not None else None,
                     'progress': last.progress if last is not None else None,
                     'last_frame': marker.last_seen_frame,
                     'age': frame.midpoint-marker.last_seen_time})
    return {'tracks': len(engine.tracks), 'markers': len(markers),
            'anchors': len(getattr(chain, 'anchors', {})),
            'stationary': sorted(getattr(tracker, 'stationary_origins', set())),
            'moving': sorted(getattr(tracker, 'moving_origins', set())),
            'marker_rows': rows}


def graph_observer(edges, prior, detections, frame):
    if ACTIVE is None:
        return
    by_i, by_j = {}, {}
    for i, j in edges:
        by_i.setdefault(i, set()).add(j)
        by_j.setdefault(j, set()).add(i)
    components, visited = [], set()
    for seed in by_i:
        if seed in visited:
            continue
        ci, cj, todo = {seed}, set(), [seed]
        while todo:
            i = todo.pop()
            for j in by_i[i]-cj:
                cj.add(j)
                added = by_j[j]-ci
                ci.update(added)
                todo.extend(added)
        visited.update(ci)
        components.append({'detection_indices': sorted(ci),
                           'marker_ids': [prior[j].marker_id for j in sorted(cj)],
                           'detections': len(ci), 'tracks': len(cj),
                           'edges': sum(i in ci and j in cj for i, j in edges)})
    ACTIVE['association'].append({'frame': frame.sequence, 'pts': frame.midpoint,
                                 'detections': len(detections), 'prior': len(prior),
                                 'edges': len(edges), 'components': components})


def solver_observer(current, previous, solve, frame):
    if ACTIVE is None:
        return
    cache = solve.cache_info()
    ACTIVE['solver'].append({'frame': frame.sequence, 'pts': frame.midpoint,
                            'detections': len(current), 'tracks': len(previous),
                            'cache_hits': cache.hits, 'states': cache.misses,
                            'cache_size': cache.currsize})


def install_association_observer():
    import agent.music.holds as holds
    from agent.music.tap_hold_chain import GoldMarkerTracker
    original = GoldMarkerTracker.associate_detections
    source = textwrap.dedent(inspect.getsource(original))
    # The instrumented function is compiled outside its original class body.
    # Spell the equivalent class explicitly instead of losing its __class__ cell.
    source = strict.substitute(source, 'super().observe(',
                               'super(GoldMarkerTracker, self).observe(')
    anchor = '    # Solve connected conflicts only. Most rings have one possible edge.'
    source = strict.substitute(source, anchor,
        '    _point_graph_observer(edges, prior, detections, frame)\n'+anchor)
    anchor = '        for i, j in solve(0, 0)[2]:'
    source = strict.substitute(source, anchor,
        '        _point_solution = solve(0, 0)\n'
        '        _point_solver_observer(current, previous, solve, frame)\n'
        '        for i, j in _point_solution[2]:')
    tree = ast.parse(source)
    ast.increment_lineno(tree, original.__code__.co_firstlineno-1)
    namespace = dict(original.__globals__)
    namespace.update(_point_graph_observer=graph_observer,
                     _point_solver_observer=solver_observer)
    exec(compile(tree, original.__code__.co_filename, 'exec'), namespace)
    GoldMarkerTracker.associate_detections = namespace['associate_detections']
    original_ribbon = holds.bonus_hold_ribbon_present

    @wraps(original_ribbon)
    def observed_ribbon(image, candidate, tangent, **kwargs):
        record = None
        if ACTIVE is not None:
            caller = inspect.currentframe().f_back
            track = caller.f_locals.get('track')
            record = {'candidate': asdict(candidate), 'tangent': list(tangent),
                      'caller': caller.f_code.co_name,
                      'track_id': getattr(track, 'track_id', None),
                      'track_gesture': getattr(getattr(track, 'gesture', None), 'value', None),
                      'track_visual_family': getattr(track, 'visual_family', None),
                      'track_speed': getattr(track, 'speed', None),
                      'track_state': getattr(getattr(track, 'state', None), 'value', None)}
            ACTIVE['ribbon_calls'].append(record)
            del caller
        result = original_ribbon(image, candidate, tangent, **kwargs)
        if record is not None:
            record['result'] = result
            record['metadata_evidence_requested'] = kwargs.get('evidence') is not None
        return result

    holds.bonus_hold_ribbon_present = observed_ribbon


def begin_stage(stage, pts, engine, frame):
    global ACTIVE
    target = {'update': 100.300000, 'prepass': 100.333333}[stage]
    if abs(pts-target) > 1e-6:
        return
    image_reference = None
    if INPUT_ROOT is not None:
        import numpy as np
        INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        target = INPUT_ROOT/f'{stage}-{pts:.6f}.npy'
        if target.exists():
            raise FileExistsError(f'Preserve original pixel evidence: {target}')
        np.save(target, np.asarray(frame.image), allow_pickle=False)
        image_reference = {'path': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                           'shape': list(np.asarray(frame.image).shape),
                           'format': 'Original unmodified BGR NumPy array at decoded recorded PTS'}
    ACTIVE = {'stage': stage, 'pts': pts, 'before': state_summary(engine, frame),
              'association': [], 'solver': [], 'ribbon_calls': [],
              'image_reference': image_reference, '_profiler': cProfile.Profile()}
    ACTIVE['_profiler'].enable()


def end_stage(stage, pts, engine, frame):
    global ACTIVE
    if ACTIVE is None or ACTIVE['stage'] != stage:
        return
    profiler = ACTIVE.pop('_profiler')
    profiler.disable()
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, function), (primitive, total, own, cumulative, _) in stats.stats.items():
        rows.append({'file': filename, 'line': line, 'function': function,
                     'primitive_calls': primitive, 'total_calls': total,
                     'own_ms': own*1000, 'cumulative_ms': cumulative*1000})
    ACTIVE.update(after=state_summary(engine, frame),
                  top_cumulative=sorted(rows, key=lambda row: -row['cumulative_ms'])[:50],
                  selected_functions=[row for row in rows if row['function'] in
                                      ('solve', 'lane', 'speed', 'associate_detections',
                                       'recover_missing', 'detect_hold_tails')])
    RECORDS.append(ACTIVE)
    ACTIVE = None


def main():
    global INPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--branch-root', type=Path, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    INPUT_ROOT = args.work_root.resolve()/'reports'/args.label/'pixel-inputs'
    source = strict.prepare(args.work_root.resolve(), args.inventory.resolve())
    source = strict.substitute(source,
        '    logging.disable(logging.CRITICAL)',
        '    logging.disable(logging.CRITICAL)\n    _point_install_association_observer()')
    source = strict.substitute(source,
        '            chain = getattr(engine,\'tap_hold_chain\',None)',
        '            chain = getattr(engine,\'tap_hold_chain\',None)')
    source = strict.substitute(source,
        "            begin = time.perf_counter()\n            begin_cpu = time.process_time()\n            chain =",
        "            _point_begin('prepass',pts,engine,frame)\n"
        "            begin = time.perf_counter()\n            begin_cpu = time.process_time()\n            chain =")
    source = strict.substitute(source,
        '            prepass_process_ms = (time.process_time()-begin_cpu)*1000.',
        '            prepass_process_ms = (time.process_time()-begin_cpu)*1000.\n'
        "            _point_end('prepass',pts,engine,frame)")
    source = strict.substitute(source,
        '            begin = time.perf_counter()\n            begin_cpu = time.process_time()\n            events = engine.update',
        "            _point_begin('update',pts,engine,frame)\n"
        '            begin = time.perf_counter()\n            begin_cpu = time.process_time()\n            events = engine.update')
    source = strict.substitute(source,
        '            update_process_ms = (time.process_time()-begin_cpu)*1000.',
        '            update_process_ms = (time.process_time()-begin_cpu)*1000.\n'
        "            _point_end('update',pts,engine,frame)")
    source = strict.substitute(source,
        "'mode':'point-strict-source-PTS-qualification-wall'",
        "'mode':'developer-profile-instrumented-PTS-NOT-ADMISSION'")
    source = strict.substitute(source,
        "            'inventory_hash':inventory_hash,",
        "            'developer_profile':_POINT_PROFILE_RECORDS,\n"
        "            'developer_tool_hash':_POINT_PROFILE_TOOL_HASH,\n"
        "            'developer_limitations':'Targeted cProfile and observer overhead; all frame times invalid as admission evidence.',\n"
        "            'inventory_hash':inventory_hash,")
    argv = sys.argv
    sys.argv = [str(strict.HERE), '--branch-root', str(args.branch_root.resolve()),
                '--label', args.label, '--clips', 'yellow-hud-v1']
    namespace = {'__file__': str(strict.HERE), '__name__': 'point_profile_implementation',
                 'TEMPLATE_PATH': strict.TEMPLATE, '_point_begin': begin_stage,
                 '_point_end': end_stage, '_point_install_association_observer': install_association_observer,
                 '_POINT_PROFILE_RECORDS': RECORDS,
                 '_POINT_PROFILE_TOOL_HASH': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        exec(compile(source, str(strict.HERE), 'exec'), namespace)
        return namespace['main']()
    finally:
        sys.argv = argv


if __name__ == '__main__':
    raise SystemExit(main())
