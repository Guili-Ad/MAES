"""Process-isolated source/action contracts; no simulator or game judgments.

Run this tool with explicit baseline/candidate roots. The worker imports only
its selected source tree. Reports are development evidence, not song actions.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys


def source_fingerprint(source):
    files = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((source/'agent').rglob('*.py'))}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def worker(source):
    sys.path[:0] = [str(source), str(source/'tests')]
    import unittest
    from dataclasses import replace
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    import test_longtap_branch as hold_cases
    import test_sustain_chain as sustain
    import test_hold_note_taps as small_notes
    from test_flick_session import Clock, Context
    from agent.music.tracking import MusicVisionEngine
    from agent.music.models import MusicActionEvent, MusicConfig, MusicFrame, NoteGesture, NoteTrack, TrackObservation
    from agent.music.runtime import MusicRuntime, RuntimeMetrics
    from agent.music.executor import MusicActionExecutor
    from agent.music.vision import VisualMask
    logging.disable(logging.CRITICAL)
    records, gold_records, active = [], [], ['']
    originals = {}
    for name in ('update', 'release_events', 'refine_pending'):
        original = getattr(MusicVisionEngine, name)
        originals[name] = original
        def wrap(method, label):
            @functools.wraps(method)
            def run(engine, *args, **kwargs):
                events = method(engine, *args, **kwargs)
                if not engine.config.hold_notes_as_taps:
                    for event in events:
                        if event.gesture.value.startswith(('Hold', 'Sustain')):
                            records.append(dict(test=active[0], method=label, track=event.track_id,
                                gesture=event.gesture.value, lane=event.lane,
                                deadline=event.deadline, coordinate=event.coordinate))
                else:
                    for event in events:
                        if event.origin == 'hold_note':
                            gold_records.append(dict(test=active[0], method=label, track=event.track_id,
                                gesture=event.gesture.value, lane=event.lane,
                                deadline=event.deadline, coordinate=event.coordinate))
                return events
            return run
        setattr(MusicVisionEngine, name, wrap(original, name))
    classes = [hold_cases.LongTapBranchTests, sustain.SustainTrackerTests,
               sustain.SustainPlannerTests, sustain.SustainRuntimeTests, small_notes.HoldNoteTapTests]
    selected = [(cls, name) for cls in classes
        for name in unittest.defaultTestLoader.getTestCaseNames(cls)
        if cls is not hold_cases.LongTapBranchTests or any(word in name for word in
            ('hold', 'tail', 'ribbon', 'route', 'sustain', 'cap_locked', 'terminal_target'))]
    result = unittest.TestResult()
    for cls, name in selected:
        active[0] = cls.__name__+'.'+name
        cls(name).run(result)
    for name, original in originals.items():
        setattr(MusicVisionEngine, name, original)

    point_contracts = {}
    # The same motion fit and successful visual qualification are injected on
    # both sides. This isolates scheduling/input, not image classifier quality.
    for family, gesture, bonus in (
            ('ordinary', NoteGesture.TAP, False),
            ('bonus', NoteGesture.TAP, True),
            ('yellow_head', NoteGesture.HOLD_START, False),
            ('bonus_yellow_head', NoteGesture.HOLD_START, True)):
        cal = hold_cases.calibration()
        config = MusicConfig(lane_count=7, enable_holds=True, hold_notes_as_taps=True,
                             hold_start_action_advance_ms=175)
        engine = MusicVisionEngine(cal, config)
        track = NoteTrack(1, 3, gesture=gesture, bonus_star=bonus, hold_evidence_frames=3 if gesture == NoteGesture.HOLD_START else 0)
        track.point_mode = True
        track.visual_family = 'bonus' if bonus else family
        track.timing_profile = 'yellow_head' if gesture == NoteGesture.HOLD_START else family
        for sequence in range(5):
            stamp, progress = .6+sequence*.1, .60+sequence*.05
            candidate = hold_cases.candidate_at(cal, 3, progress)
            track.observations.append(TrackObservation(sequence, stamp, candidate.center, progress, candidate))
        engine.tracks[1], engine.next_track_id = track, 2
        engine._update_motion(track)
        frame = MusicFrame(4, 1., 1., 1., np.zeros((720, 1280, 3), np.uint8))
        with patch.object(engine, '_associate_lane'), patch.object(engine, '_ready_to_schedule', return_value=True), \
                patch.object(engine, '_update_active_hold_tails'), \
                patch('agent.music.tracking.detect_bonus_star_notes', return_value=[]), \
                patch.object(engine, '_update_center_color_note', return_value=(None, [])):
            events = engine.update(frame, [], VisualMask.from_image(frame.image, cal))
        if len(events) != 1:
            raise AssertionError(f'{family}: expected one physical point, got {len(events)}')
        event = events[0]
        clock, context = Clock(), None
        clock.now = event.deadline
        context = Context(clock, cost=.008)
        runtime = MusicRuntime(context, config, clock=clock, sleeper=clock.sleep)
        runtime.tap_trace = engine.tap_trace
        executor = MusicActionExecutor(context, 1280, 720, config,
            advanced=True, multi_touch=True, clock=clock, sleeper=clock.sleep)
        executor.begin_segment(engine.tap_trace.run_id, engine.tap_trace.segment_id)
        # New positive evidence at dispatch; do not grant a stale-fit exemption.
        progress = .80+.5*(event.deadline-1.)
        candidate = hold_cases.candidate_at(cal, 3, progress)
        track.observations.append(TrackObservation(5, event.deadline, candidate.center, progress, candidate))
        engine._update_motion(track)
        engine.last_frame_sequence = 5
        engine.last_frame = MusicFrame(5, event.deadline, event.deadline, event.deadline, frame.image)
        pending = [event]
        runtime._execute_due(executor, pending, clock.now, RuntimeMetrics(), engine, wait=False)
        point_contracts[family] = dict(deadline=event.deadline, coordinate=event.coordinate,
            calls=context.calls, pending=len(pending), planned_gesture=event.gesture.value)
        if len(context.calls) != 2 or pending:
            raise AssertionError(f'{family}: expected one full down/up, got {context.calls}')

    # Two simultaneous and slightly offset independent flicks keep actual call
    # order, coordinates, deadline and step waits unchanged.
    flicks = {}
    for skew in (0., .020):
        clock = Clock()
        context = Context(clock, cost=.008)
        config = MusicConfig(lane_count=7, enable_holds=True, hold_notes_as_taps=True,
                             flick_duration_ms=60, flick_steps=3, flick_end_hold_ms=16.)
        runtime = MusicRuntime(context, config, clock=clock, sleeper=clock.sleep)
        executor = MusicActionExecutor(context, 1280, 720, config,
            advanced=True, multi_touch=True, clock=clock, sleeper=clock.sleep)
        executor.begin_segment(runtime.tap_trace.run_id, runtime.tap_trace.segment_id)
        events = [MusicActionEvent('left', 101, 1, NoteGesture.FLICK_LEFT, 10., (250, 516)),
                  MusicActionEvent('right', 105, 5, NoteGesture.FLICK_RIGHT, 10.+skew, (1030, 516))]
        runtime._execute_due(executor, events, clock.now, RuntimeMetrics(), wait=False)
        flicks[str(skew)] = dict(calls=context.calls, pending=len(events), waits=clock.sleeps)

    audit = None
    if (source/'agent/music/point_events.py').exists():
        from agent.music.point_events import point_registry
        engine = MusicVisionEngine(hold_cases.calibration(), MusicConfig(hold_notes_as_taps=True))
        registry = point_registry(engine)
        physical_ids, event_ids = set(), set()
        for i in range(4000):
            family = ('ordinary', 'bonus', 'yellow_head', 'gold_ring')[i % 4]
            event = MusicActionEvent(f'renamed-{i}', 1, 3, NoteGesture.TAP, 1.+i*.001, (640, 620))
            adopted = registry.adopt(event, source='track', key=1, family=family, timing_profile=family)
            physical_ids.add(adopted.physical_id)
            event_ids.add(adopted.event_id)
        registry.acknowledge(engine, adopted, SimpleNamespace(down_call_started=5.,
            down_call_finished=None, up_call_finished=None, error='injected failed down'))
        retry = registry.adopt(adopted, source='track', key=1, family='ordinary', timing_profile='ordinary')
        fixed_identity = len(registry.states) == 1 and len(physical_ids) == len(event_ids) == 1 and retry is None
        engine.tap_trace.segment_id += 1
        reset = point_registry(engine)
        audit = dict(classification_changes=4000, physical_ids=len(physical_ids),
            event_ids=len(event_ids), state_count=len(registry.states), source_count=len(registry.source_ids),
            down_attempt_retry_rejected=retry is None, new_segment_empty=not reset.states,
            passed=fixed_identity and not reset.states,
            memory_policy='states/source aliases are retained per physical birth for the entire run; not a hard-cap table. '
                          'Repeated classifications are constant-space; namespace reset clears the table. '
                          'Input trace buffers are bounded independently.')
    return dict(source=str(source), legacy_tests=len(selected), legacy_records=records,
        healthy_gold_records=gold_records,
        legacy_errors=[dict(test=str(case), error=error) for case, error in result.errors+result.failures],
        point_contracts=point_contracts, flick_contracts=flicks, registry_audit=audit)


def compare(left, right):
    failures = []
    lrecords, rrecords = left['legacy_records'], right['legacy_records']
    if len(lrecords) != len(rrecords):
        failures.append('legacy action count differs')
    max_delta = 0.
    for index, (a, b) in enumerate(zip(lrecords, rrecords)):
        delta = abs(a['deadline']-b['deadline'])
        max_delta = max(max_delta, delta)
        if {k:v for k,v in a.items() if k != 'deadline'} != {k:v for k,v in b.items() if k != 'deadline'} or delta > .0001:
            failures.append(f'legacy action {index} differs')
    for family, a in left['point_contracts'].items():
        b = right['point_contracts'][family]
        delta = abs(a['deadline']-b['deadline'])
        max_delta = max(max_delta, delta)
        # HOLD_START -> TAP is intentional; native Down/Up is the contract.
        if (a['coordinate'] != b['coordinate'] or a['calls'] != b['calls']
                or a['pending'] != b['pending'] or delta > .0001):
            failures.append(f'{family} native point contract differs')
    lgold, rgold = left['healthy_gold_records'], right['healthy_gold_records']
    if len(lgold) != len(rgold):
        failures.append('healthy gold scheduling count differs')
    for index, (a, b) in enumerate(zip(lgold, rgold)):
        delta = abs(a['deadline']-b['deadline'])
        max_delta = max(max_delta, delta)
        if {k:v for k,v in a.items() if k != 'deadline'} != {k:v for k,v in b.items() if k != 'deadline'} or delta > .0001:
            failures.append(f'healthy gold action {index} differs')
    if left['flick_contracts'] != right['flick_contracts']:
        failures.append('interleaved flick native contract differs')
    if left['legacy_errors'] or right['legacy_errors']:
        failures.append('selected legacy tests failed')
    if not right['registry_audit']['passed']:
        failures.append('classification/namespace audit failed')
    return dict(passed=not failures, failures=failures,
        legacy_action_count=len(lrecords), legacy_test_count=left['legacy_tests'],
        healthy_gold_action_count=len(lgold), old_contract_total_actions=len(lrecords)+len(lgold),
        point_families=list(left['point_contracts']), flick_scenarios=2,
        max_deadline_difference_ms=max_delta*1000., deadline_tolerance_ms=.1,
        limitation='Injected healthy observations and synchronous native-call simulation; no game Bad/Miss inference.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker.resolve()), ensure_ascii=False))
        return 0
    if not all((args.baseline, args.candidate, args.output)):
        parser.error('explicit --baseline, --candidate and --output are required')
    snapshots = []
    for source in (args.baseline, args.candidate):
        before = source_fingerprint(source.resolve())
        process = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
            '--worker', str(source.resolve())], capture_output=True, text=True, encoding='utf-8')
        if process.returncode:
            raise RuntimeError(f'Contract worker failed for {source}:\n{process.stderr}')
        snapshot = json.loads(process.stdout)
        snapshot['source_fingerprint'] = before
        snapshot['source_stable'] = source_fingerprint(source.resolve()) == before
        if not snapshot['source_stable']:
            raise RuntimeError(f'Source changed during contract validation: {source}')
        snapshots.append(snapshot)
    payload = dict(schema=1, comparison=compare(*snapshots), baseline=snapshots[0], candidate=snapshots[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload['comparison'], ensure_ascii=False, indent=2))
    return 0 if payload['comparison']['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
