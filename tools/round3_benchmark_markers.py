"""Compare gold tracking/planning using successful mock head input receipts.

The fixture is the previous four/eight mature-marker geometry and timestamps.
Only its invalid synthetic HOLDING owner setup is replaced by production
head dispatch plus Down/Up receipts, identically for both source branches.
No detector or controller work is timed. Reports never infer game grades.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
import math
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

APP_ROOT = Path(__file__).resolve().parents[1]


def quantile(values, fraction):
    values = sorted(values)
    point = (len(values)-1)*fraction
    low, high = math.floor(point), math.ceil(point)
    return values[low]+(values[high]-values[low])*(point-low)


def describe(values):
    return {'count': len(values), 'mean_ms': statistics.mean(values),
            'p50_ms': quantile(values, .5), 'p95_ms': quantile(values, .95), 'max_ms': max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    if args.iterations < 10 or not args.output.resolve().is_relative_to((APP_ROOT/'temp').resolve()):
        parser.error('At least 10 repeats; output must stay under app/temp')
    branch = args.branch_root.resolve()
    sys.path.insert(0, str(branch))
    import numpy as np
    import agent.music.tracking as tracking
    import agent.music.holds as holds
    from agent.music.executor import MusicActionExecutor
    from agent.music.models import (MusicActionEvent, MusicCalibrationData, MusicConfig, MusicFrame,
                                    NoteGesture, NoteTrack, TrackState)
    from agent.music.runtime import MusicRuntime, RuntimeMetrics
    logging.disable(logging.CRITICAL)
    points = [[160+lane*160, 620] for lane in range(7)]
    calibration = MusicCalibrationData(version=4, lane_count=7, width=1280, height=720,
        points=points, lane_centerlines=[[[float(x), 140.], [float(x), 380.], [float(x), float(y)]]
                                      for x,y in points], corridor_widths=[52.]*7,
        trigger_progress=1., candidate_roi=[0, 100, 1280, 590], exclusion_rois=[],
        baseline_version='maes-music-v4-2026-08', action_advance_ms=125.,
        color_lower=[[0,45,110]], color_upper=[[179,255,255]], candidate_min_pixels=12,
        hold_min_length=100., created_at='')
    config = MusicConfig(hold_notes_as_taps=True, hold_sustain_enabled=False, enable_holds=True, lane_count=7)
    image = np.zeros((720,1280,3), np.uint8)
    detector_output = [[]]
    # The sole visual boundary replacement is identical in both roots.
    tracking.detect_hold_tails = lambda *unused: detector_output[0]
    holds.detect_hold_tails = lambda *unused: detector_output[0]

    class Clock:
        def __init__(self):
            self.now = .4
        def __call__(self):
            self.now += .000001
            return self.now
        def sleep(self, duration):
            self.now += duration

    def sequence(lanes):
        frames = []
        for index, moment in enumerate((1.40, 1.44, 1.48, 1.52, 1.56, 1.60, 1.64, 1.68, 1.90)):
            detections = []
            for owner_index, lane in enumerate(lanes):
                for hit_base in (2.,2.20):
                    hit = hit_base+(owner_index%2)*.01
                    progress = 1.-(hit-moment)*.5
                    detections.append(holds.HoldTailDetection(progress,.8,100,lane,
                        (float(points[lane][0]),140.+progress*480.),0.,2,'checkpoint',(lane,)))
            frames.append((MusicFrame(index+1,moment,moment,moment,image),detections))
        return frames

    def prepare(lanes, fixtures):
        clock = Clock()
        context = SimpleNamespace()
        runtime = MusicRuntime(context,config,clock=clock,sleeper=clock.sleep)
        engine = tracking.MusicVisionEngine(calibration,config,tap_trace=runtime.tap_trace)
        executor = MusicActionExecutor(context,1280,720,config,advanced=True,multi_touch=True,
                                       clock=clock,sleeper=clock.sleep)
        executor._bindings()
        executor.begin_segment(runtime.tap_trace.run_id,runtime.tap_trace.segment_id)
        actions = []
        def mock_input(kind,param,*unused,**unused_kw):
            actions.append({'kind':kind.value,'contact':param.contact,'target':getattr(param,'target',None)})
        executor._run = mock_input
        heads = []
        for index,lane in enumerate(lanes):
            owner_id = index+1
            owner = NoteTrack(owner_id,lane,gesture=NoteGesture.HOLD_START,
                              state=TrackState.HOLD_PENDING,predicted_hit_time=.5,hold_release_time=5.)
            owner.linked_partner_id = index+2 if index%2==0 else index
            engine.tracks[owner_id] = owner
            heads.append(MusicActionEvent(f'fixture-head-{owner_id}',owner_id,lane,
                NoteGesture.HOLD_START,.375,tuple(points[lane]),
                source_capture_started=.3,source_capture_finished=.31))
        runtime._execute_due(executor,heads,clock(),RuntimeMetrics(),engine,wait=False)
        receipts = [row['receipt'] for row in runtime.tap_trace.records if row.get('kind')=='input']
        if (heads or len(receipts)!=len(lanes)
                or any(r.get('error') or r.get('up_call_finished') is None for r in receipts)):
            raise RuntimeError('The owner fixture failed production mock head input/receipts')
        if hasattr(engine,'tap_hold_chain') and engine.tap_hold_chain is not None:
            if len(engine.tap_hold_chain.anchors)!=len(lanes):
                raise RuntimeError('Successful head receipts failed to create new virtual anchors')
        head_contract = {'receipts':len(receipts),'actions':list(actions),
                         'owner_states':[owner.state.value for owner in engine.tracks.values()]}
        actions.clear()
        pending = []
        for frame,detections in fixtures[:-1]:
            clock.now = frame.midpoint
            detector_output[0] = detections
            engine.last_frame_sequence = frame.sequence
            engine._update_active_hold_tails(frame)
            pending.extend(engine.release_events(clock()))
            pending = engine.refine_pending(pending,clock())
        return clock,runtime,engine,executor,pending,actions,head_contract

    scenarios = {}
    for name,lanes in (('four_markers_two_owners',(1,5)),('eight_markers_four_owners',(0,6,2,4))):
        fixtures = sequence(lanes)
        stage_samples = {key:[] for key in ('tail_update','release_events','refine_pending',
                                           'sync_input_and_receipts','total')}
        signatures, first_contract = [], None
        for repeat in range(args.iterations+20):
            clock,runtime,engine,executor,pending,actions,head_contract = prepare(lanes,fixtures)
            frame,detections = fixtures[-1]
            clock.now = frame.midpoint
            detector_output[0] = detections
            engine.last_frame_sequence = frame.sequence
            moments = [time.perf_counter_ns()]
            engine._update_active_hold_tails(frame)
            moments.append(time.perf_counter_ns())
            pending.extend(engine.release_events(clock()))
            moments.append(time.perf_counter_ns())
            pending = engine.refine_pending(pending,clock())
            moments.append(time.perf_counter_ns())
            before = list(pending)
            runtime._execute_due(executor,pending,clock(),RuntimeMetrics(),engine,wait=False)
            moments.append(time.perf_counter_ns())
            if repeat < 20:
                continue
            for index,key in enumerate(('tail_update','release_events','refine_pending','sync_input_and_receipts')):
                stage_samples[key].append((moments[index+1]-moments[index])/1e6)
            stage_samples['total'].append((moments[-1]-moments[0])/1e6)
            contract = {'head_contract':head_contract,'markers_retained':len(engine.sustain_tracker.markers),
                'events':[{'id':e.event_id,'lane':e.lane,'gesture':e.gesture.value,
                           'deadline':e.deadline,'coordinate':e.coordinate,'group':e.tap_group_id} for e in before],
                'actions':list(actions),'input_receipts':sum(row.get('kind')=='input' and row.get('origin')=='hold_note'
                                                            for row in runtime.tap_trace.records),
                'pending_after_input':[(e.lane,e.gesture.value,e.deadline) for e in pending],
                'retained_owner_states':[owner.state.value for owner in engine.tracks.values()]}
            signatures.append(hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest())
            if first_contract is None:
                first_contract = contract
        if len(set(signatures))!=1:
            raise RuntimeError(f'{name}: fixed input produced non-deterministic contracts')
        if (first_contract['markers_retained']!=len(lanes)*2
                or first_contract['input_receipts']!=len(lanes)
                or len(first_contract['pending_after_input'])!=len(lanes)):
            raise RuntimeError(f'{name}: gold input contract lost/repeated physical markers or actions: {first_contract}')
        scenarios[name] = {'markers':len(lanes)*2,'owners':len(lanes),
                          'timings':{key:describe(values) for key,values in stage_samples.items()},
                          'stable_contract_sha256':signatures[0],'contract':first_contract}
    report = {'schema':2,'branch_root':str(branch),'iterations':args.iterations,
        'warmup_iterations':20,'fixture':'original mature gold motion plus real successful mock head receipts',
        'included':['gold association/ownership','planning/refinement','sync input receipts'],
        'excluded':['fixture/owner construction','head input','image detection','video decode','controller I/O','disk I/O'],
        'limitations':'Detector-boundary CPU and transport contract; no game grade or image classifier performance.',
        'scenarios':scenarios}
    passed = True
    if args.compare:
        old = json.loads(args.compare.read_text(encoding='utf-8'))
        comparisons = {}
        for name,entry in scenarios.items():
            original = old['scenarios'][name]
            left,right = original['contract'],json.loads(json.dumps(entry['contract']))
            semantic_events = lambda c: [(e['lane'],e['gesture'],e['coordinate']) for e in c['events']]
            event_deltas = [abs(a['deadline']-b['deadline'])*1000.
                            for a,b in zip(left['events'],right['events'])]
            pending_deltas = [abs(a[2]-b[2])*1000.
                              for a,b in zip(left['pending_after_input'],right['pending_after_input'])]
            max_deadline_delta = max([*event_deltas,*pending_deltas],default=0.)
            pending_structure = lambda c: [(p[0],p[1]) for p in c['pending_after_input']]
            invariant = (left['head_contract']==right['head_contract']
                         and left['actions']==right['actions']
                         and semantic_events(left)==semantic_events(right)
                         and max_deadline_delta<=.1
                         and left['markers_retained']==right['markers_retained']
                         and left['input_receipts']==right['input_receipts']
                         and pending_structure(left)==pending_structure(right)
                         and left['retained_owner_states']==right['retained_owner_states'])
            delta = entry['timings']['total']['p95_ms']-original['timings']['total']['p95_ms']
            comparisons[name] = {'p95_increment_ms':delta,'within_2ms_budget':delta<=2.,
                'semantic_contract_preserved':invariant,
                'max_deadline_difference_ms':max_deadline_delta,
                'intentional_event_id_change':'new physical-marker key removes owner from ID; same input position/gesture/deadline required'}
            passed = passed and invariant and delta<=2.
        report['comparison'] = {'baseline':str(args.compare),'scenarios':comparisons,'passed':passed}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'scenarios':{k:v['timings']['total'] for k,v in scenarios.items()},
                      'comparison':report.get('comparison')}))
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
