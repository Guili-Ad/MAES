"""Compare the real ribbon-marker pipeline without a controller or video decode.

Each invocation imports exactly one source root. Precomputed detector output is
the identical fixture boundary for both roots; the changed association/planning/
refinement and the real synchronous executor plus receipt handling are measured.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
import math
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def percentile(values, quantile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def describe(values):
    return {'count': len(values), 'mean_ms': statistics.mean(values),
            'p50_ms': percentile(values, .5), 'p95_ms': percentile(values, .95),
            'max_ms': max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--compare', type=Path, help='Previously generated baseline report')
    args = parser.parse_args()
    if args.iterations < 10:
        parser.error('At least 10 measured repetitions are required')
    if not args.output.resolve().is_relative_to((ROOT / 'temp').resolve()):
        parser.error('Reports must stay under app/temp')
    branch = args.branch_root.resolve()
    if not (branch / 'agent/music/tracking.py').is_file():
        parser.error('The selected source root is missing the tracking module')
    sys.path.insert(0, str(branch))
    import numpy as np
    import agent.music.tracking as tracking
    from agent.music.executor import MusicActionExecutor
    from agent.music.holds import HoldTailDetection
    from agent.music.models import (MusicCalibrationData, MusicConfig, MusicFrame,
                                    NoteGesture, NoteTrack, TrackState)
    from agent.music.runtime import MusicRuntime, RuntimeMetrics
    logging.disable(logging.CRITICAL)

    points = [[160 + lane * 160, 620] for lane in range(7)]
    calibration = MusicCalibrationData(
        version=4, lane_count=7, width=1280, height=720, points=points,
        lane_centerlines=[[[float(x), 140.], [float(x), 380.], [float(x), float(y)]]
                          for x, y in points],
        corridor_widths=[52.] * 7, trigger_progress=1., candidate_roi=[0, 100, 1280, 590],
        exclusion_rois=[], baseline_version='maes-music-v4-2026-08', action_advance_ms=125.,
        color_lower=[[0, 45, 110]], color_upper=[[179, 255, 255]], candidate_min_pixels=12,
        hold_min_length=100., created_at='')
    config = MusicConfig(hold_notes_as_taps=True, hold_sustain_enabled=False,
                         enable_holds=True, lane_count=7)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector_output = [[]]
    # Only detector output is mocked. Every edited marker/tracking/scheduling
    # operation and all unchanged ribbon/route checks run through production.
    tracking.detect_hold_tails = lambda *unused: detector_output[0]

    class Clock:
        def __init__(self):
            self.now = 1.4

        def __call__(self):
            self.now += .000001
            return self.now

        def sleep(self, duration):
            self.now += duration

    def sequence(lanes):
        frames = []
        for index, moment in enumerate((1.40, 1.44, 1.48, 1.52, 1.56, 1.60, 1.64, 1.68, 1.90)):
            tails = []
            for owner_index, lane in enumerate(lanes):
                # Two physical rings per owner, paired with its reciprocal
                # linked owner. A 10 ms raw skew exercises shared deadlines.
                for hit_base in (2., 2.20):
                    hit = hit_base + (owner_index % 2) * .01
                    progress = 1. - (hit - moment) * .5
                    tails.append(HoldTailDetection(progress, .8, 100, lane,
                        (float(points[lane][0]), 140. + progress * 480.), 0.,
                        2, 'checkpoint', (lane,)))
            frames.append((MusicFrame(index + 1, moment, moment, moment, image), tails))
        return frames

    def prepare(lanes, fixtures):
        clock = Clock()
        runtime = MusicRuntime(SimpleNamespace(), config, clock=clock, sleeper=clock.sleep)
        engine = tracking.MusicVisionEngine(calibration, config, tap_trace=runtime.tap_trace)
        for index, lane in enumerate(lanes):
            owner_id = index + 1
            owner = NoteTrack(owner_id, lane, gesture=NoteGesture.HOLD_START,
                state=TrackState.HOLDING, predicted_hit_time=.5, hold_release_time=5.)
            owner.linked_partner_id = (index + 2 if index % 2 == 0 else index)
            engine.tracks[owner_id] = owner
        executor = MusicActionExecutor(SimpleNamespace(), 1280, 720, config,
            advanced=True, multi_touch=True, clock=clock, sleeper=clock.sleep)
        executor._bindings()  # Maa type import/binding excluded; no controller.
        executor.begin_segment(runtime.tap_trace.run_id, runtime.tap_trace.segment_id)
        actions = []

        def mock_input(kind, param, *unused, **unused_kw):
            # _run is the sole replaced input boundary. It never calls a
            # controller; real allocation, deduplication, up and receipts run.
            actions.append({'kind': kind.value, 'contact': param.contact,
                            'target': getattr(param, 'target', None)})

        executor._run = mock_input
        pending = []
        for frame, tails in fixtures[:-1]:
            clock.now = frame.midpoint
            detector_output[0] = tails
            engine.last_frame_sequence = frame.sequence
            engine._update_active_hold_tails(frame)
            pending.extend(engine.release_events(clock()))
            pending = engine.refine_pending(pending, clock())
        return clock, runtime, engine, executor, pending, actions

    scenarios = {}
    for name, lanes in (('four_markers_two_owners', (1, 5)),
                        ('eight_markers_four_owners', (0, 6, 2, 4))):
        fixtures = sequence(lanes)
        stage_samples = {key: [] for key in ('tail_update', 'release_events', 'refine_pending',
                                            'sync_input_and_receipts', 'total')}
        signatures = []
        first_contract = None
        for repetition in range(args.iterations + 20):
            clock, runtime, engine, executor, pending, actions = prepare(lanes, fixtures)
            frame, tails = fixtures[-1]
            clock.now = frame.midpoint
            detector_output[0] = tails
            engine.last_frame_sequence = frame.sequence
            moments = [time.perf_counter_ns()]
            engine._update_active_hold_tails(frame)
            moments.append(time.perf_counter_ns())
            pending.extend(engine.release_events(clock()))
            moments.append(time.perf_counter_ns())
            pending = engine.refine_pending(pending, clock())
            moments.append(time.perf_counter_ns())
            before_input = list(pending)
            runtime._execute_due(executor, pending, clock(), RuntimeMetrics(), engine, wait=False)
            moments.append(time.perf_counter_ns())
            if repetition < 20:
                continue
            for index, key in enumerate(('tail_update', 'release_events', 'refine_pending',
                                         'sync_input_and_receipts')):
                stage_samples[key].append((moments[index + 1] - moments[index]) / 1_000_000.)
            stage_samples['total'].append((moments[-1] - moments[0]) / 1_000_000.)
            contract = {'markers_retained': len(engine.sustain_tracker.markers),
                'events': [{'id': event.event_id, 'lane': event.lane,
                            'gesture': event.gesture.value, 'deadline': event.deadline,
                            'coordinate': event.coordinate, 'group': event.tap_group_id}
                           for event in before_input],
                'actions': actions,
                'pending_after_input': [event.event_id for event in pending],
                'input_receipts': sum(row.get('kind') == 'input' for row in runtime.tap_trace.records),
                'retained_owner_states': [owner.state.value for owner in engine.tracks.values()]}
            signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            signatures.append(signature)
            if first_contract is None:
                first_contract = contract
        if len(set(signatures)) != 1:
            raise RuntimeError(f'{name}: repeated input contracts differ')
        if first_contract['markers_retained'] != len(lanes) * 2:
            raise RuntimeError(f'{name}: moving markers were lost in benchmark fixture')
        if not first_contract['input_receipts'] or not first_contract['actions']:
            raise RuntimeError(f'{name}: fixture did not exercise actual sync input receipts')
        if first_contract['input_receipts'] != len(lanes):
            raise RuntimeError(f'{name}: leading marker input receipts were lost or repeated')
        if len(first_contract['pending_after_input']) != len(lanes):
            raise RuntimeError(f'{name}: the later marker sequence was lost or prematurely sent')
        scenarios[name] = {'markers': len(lanes) * 2, 'owners': len(lanes),
                           'timings': {key: describe(values) for key, values in stage_samples.items()},
                           'stable_contract_sha256': signatures[0], 'contract': first_contract}

    report = {'schema': 1, 'branch_root': str(branch), 'iterations': args.iterations,
        'warmup_iterations': 20, 'measured_unit': 'one production marker tick',
        'included': ['_update_active_hold_tails with real motion/identity/owner recovery',
                     'release_events', 'refine_pending',
                     '_execute_due(wait=False), synchronous executor, mock input, real receipts'],
        'excluded': ['fixture/state construction', 'detector image classification',
                     'Maa bindings import', 'video decode', 'controller I/O', 'disk output'],
        'limitations': 'A deterministic detector-boundary CPU benchmark, not an image-recognition or game-grade result.',
        'scenarios': scenarios}
    if args.compare:
        baseline = json.loads(args.compare.read_text(encoding='utf-8'))
        comparisons = {}
        for name, candidate in scenarios.items():
            original = baseline['scenarios'][name]
            differences = []
            left = original['contract']
            # Baseline came from JSON; normalize tuples in the in-memory
            # candidate before comparison so coordinates are not false deltas.
            right = json.loads(json.dumps(candidate['contract']))
            for key in ('markers_retained', 'actions', 'input_receipts', 'retained_owner_states'):
                if left[key] != right[key]:
                    differences.append({'field': key, 'baseline': left[key], 'candidate': right[key]})
            by_id = {event['id']: event for event in left['events']}
            for event in right['events']:
                previous = by_id.pop(event['id'], None)
                if previous != event:
                    differences.append({'field': 'event', 'id': event['id'],
                                        'baseline': previous, 'candidate': event})
            for event in by_id.values():
                differences.append({'field': 'removed_event', 'baseline': event})
            delta = candidate['timings']['total']['p95_ms'] - original['timings']['total']['p95_ms']
            # Paired input legitimately changes Down/Up interleaving and
            # temporary contact numbers. Compare semantic input positions,
            # event identity/order/gesture/coordinate, and retained owners.
            def semantic_events(contract):
                return [{key: event[key] for key in ('id', 'lane', 'gesture', 'coordinate')}
                        for event in contract['events']]

            def semantic_input(contract):
                return Counter((action['kind'], json.dumps(action['target']))
                               for action in contract['actions'])

            invariant = (semantic_events(left) == semantic_events(right)
                         and semantic_input(left) == semantic_input(right)
                         and left['markers_retained'] == right['markers_retained']
                         and left['input_receipts'] == right['input_receipts']
                         and left['pending_after_input'] == right['pending_after_input']
                         and left['retained_owner_states'] == right['retained_owner_states'])
            baseline_events = {event['id']: event for event in left['events']}
            shared_groups = {}
            for event in right['events']:
                if event['group']:
                    shared_groups.setdefault(event['group'], []).append(event)
            deadlines_are_targeted = all(
                len(members) == 2
                and group.startswith('holdnote-pair-')
                and all(member['id'] in baseline_events for member in members)
                and all(abs(member['deadline'] - sum(baseline_events[item['id']]['deadline']
                    for item in members) / 2.) < 1e-9 for member in members)
                for group, members in shared_groups.items())
            deadlines_are_targeted = deadlines_are_targeted and all(
                event['group'] or abs(event['deadline'] - baseline_events[event['id']]['deadline']) <= .0001
                for event in right['events'] if event['id'] in baseline_events)
            comparisons[name] = {'p95_increment_ms': delta, 'within_2ms_budget': delta < 2.,
                'stage_p95_increments_ms': {stage: candidate['timings'][stage]['p95_ms']
                    - original['timings'][stage]['p95_ms'] for stage in candidate['timings']},
                'contract_differences': differences,
                'semantic_contract_preserved': invariant,
                'targeted_changes_only': invariant and deadlines_are_targeted,
                'expected_targeted_change': 'Gold-ring pair grouping uses one shared mean deadline and an explicit hold-note group; no game-grade inference.'}
        report['comparison'] = {'baseline_report': str(args.compare), 'scenarios': comparisons,
                                'within_2ms_budget': all(item['within_2ms_budget'] for item in comparisons.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'scenarios': {
        name: scenario['timings']['total'] for name, scenario in scenarios.items()},
        'comparison': report.get('comparison')}, ensure_ascii=False))
    if args.compare and not all(item['within_2ms_budget'] and item['targeted_changes_only']
                               for item in report['comparison']['scenarios'].values()):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
