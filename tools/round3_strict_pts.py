"""Fixed-source-PTS visual-qualification CPU benchmark, no controller.

Every decoded frame is consumed exactly once. Mock transport is instantaneous
so input waits cannot change the next screenshot phase. Real tap/head Down/Up
receipts create owners; flick direction/distance/waypoints remain from the
production executor, but transport timing is deliberately not validated here.
Use production-loop replay separately for scheduling and transport evidence.
"""
from __future__ import annotations
import argparse
import ctypes
from dataclasses import asdict
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import struct
import sys
import time
from types import SimpleNamespace

APP_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = APP_ROOT.parent/'.work/round3-implementation-20261003'
MEASUREMENT = {
    'version':2,'source_clock':'Every decoded PTS consumed once; capture start=finish=midpoint=PTS',
    'decode':'All frames of one clip decoded before measurement; FFmpeg exited',
    'qualification':'engine.update + external gold prepass, no duplicate inline refresh',
    'planning':'pending extension + release_events + refine_pending',
    'dispatch':'both nonwaiting _execute_due calls, including instant mock transport overhead',
    'frame_processing':'mask/provider + qualification + planning + dispatch',
    'transport':'instant mock; real head receipts; actual visual strategy and gesture waypoints',
    'primary_timer':'perf_counter elapsed wall seconds; not process CPU',
    'auxiliary_timer':'process_time CPU seconds; informative, never replaces wall gate',
    'gate':'Guarded wall qualification P95 increment <=2ms, not complete live-loop latency'
}


def memory_available():
    """Read-only memory check; no WMI/service dependency on restricted Windows."""
    status = ctypes.create_string_buffer(64)
    ctypes.c_uint32.from_buffer(status).value = 64
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('Unable to verify memory budget before predecode')
    return ctypes.c_uint64.from_buffer(status,16).value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-root',type=Path,required=True)
    parser.add_argument('--label',required=True)
    parser.add_argument('--clips',nargs='*')
    parser.add_argument('--compare-label')
    args = parser.parse_args()
    if not args.label.replace('-','').replace('_','').isalnum():
        parser.error('Use letters, digits, hyphens or underscores for labels')
    branch = args.branch_root.resolve()
    sys.path.insert(0,str(APP_ROOT/'tools'))
    sys.path.insert(0,str(branch))
    from tap_replay import video_frames,replay_identity,load_replay_config
    from workspace_paths import ffmpeg_binary
    import agent.music.executor as bindings
    from agent.music.executor import MusicActionExecutor
    from agent.music.models import MusicConfig,MusicCalibrationData,MusicFrame
    from agent.music.runtime import MusicRuntime,RuntimeMetrics
    from agent.music.tracking import MusicVisionEngine
    from agent.music.vision import NumpyCandidateProvider,VisualMask
    from agent.music.storage import metric_summary
    logging.disable(logging.CRITICAL)
    output = APP_ROOT/'temp/round3-implementation'/args.label
    persistent = WORK_ROOT/'reports'/args.label
    output.mkdir(parents=True,exist_ok=True)
    persistent.mkdir(parents=True,exist_ok=True)
    inventory = json.loads((WORK_ROOT/'clip-fixtures.json').read_text())
    chosen = [item for item in inventory if not args.clips or item['name'] in args.clips]
    if args.clips and {item['name'] for item in chosen} != set(args.clips):
        parser.error('Unknown clip')
    source_files = {str(p.relative_to(branch)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((branch/'agent').rglob('*.py'))}
    source_hash = hashlib.sha256(json.dumps(source_files,sort_keys=True).encode()).hexdigest()
    tool_files = {file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in
                  (Path(__file__),APP_ROOT/'tools/tap_replay.py',APP_ROOT/'tools/workspace_paths.py')}
    tool_hash = hashlib.sha256(json.dumps(tool_files,sort_keys=True).encode()).hexdigest()
    measurement_hash = hashlib.sha256(json.dumps(MEASUREMENT,sort_keys=True).encode()).hexdigest()
    summaries = []

    class Clock:
        now = 0.
        def __call__(self):
            return self.now
        def sleep(self,seconds):
            if seconds > 0:
                raise RuntimeError('Strict-PTS fixture may not advance the source clock by input waits')

    class FixedTransport(MusicActionExecutor):
        """Only the transport boundary changes; visual strategy stays real."""
        def _instant(self,request,event_id='',track_id=None,deadline=None,on_started=None):
            self._bindings()
            contact = self._allocate_temporary_contact()
            receipt_class = getattr(bindings,'FlickInputReceipt',None)
            receipt = receipt_class(event_id,request.lane,contact,run_id=self.run_id,
                                    segment_id=self.segment_id) if receipt_class else None
            if event_id:
                self._used_event_ids.add(self._event_key(event_id))
            if receipt is not None:
                receipt.down_call_started = self.clock()
                if on_started:
                    submission = bindings.FlickSubmission(request,event_id,track_id,deadline)
                    on_started(submission,receipt)
            self._run(bindings.JActionType.TouchDown,
                bindings.JTouch(contact=contact,target=self._target(request.x,request.y),pressure=1),
                'strict-PTS mock flick down',self.config.max_click_touch_ms)
            if receipt is not None:
                receipt.down_call_finished = self.clock()
            end = self._flick_target(request.x,request.y,request.direction)
            for point in self._flick_waypoints(request.x,request.y,*end):
                if receipt is not None:
                    receipt.move_call_started.append(self.clock())
                self._run(bindings.JActionType.TouchMove,
                    bindings.JTouch(contact=contact,target=self._target(*point),pressure=1),
                    'strict-PTS mock flick move',self.config.max_click_touch_ms)
                if receipt is not None:
                    receipt.move_call_finished.append(self.clock())
            if receipt is not None:
                receipt.up_call_started = self.clock()
            self._run(bindings.JActionType.TouchUp,bindings.JTouchUp(contact=contact),
                      'strict-PTS mock flick up',self.config.max_click_touch_ms)
            if receipt is not None:
                receipt.up_call_finished = self.clock()
            self._temporary_contacts.discard(contact)
            return receipt
        def swipe(self,request,*,event_id='',tick=None,**unused):
            return self._instant(request,event_id)
        def swipe_many(self,submissions,*,on_started=None,**unused):
            self.last_flick_deferred = []
            return [self._instant(s.request,s.event_id,s.track_id,s.deadline,on_started)
                    for s in submissions]

    for item in chosen:
        name = item['name']
        target = output/f'{name}.json'
        if target.exists():
            raise FileExistsError(f'Preserve previous evidence; choose another label: {target}')
        package = WORK_ROOT/'baseline/candidate-state'
        calibration_path = package/'user-data/calibration/music.json'
        config_path = package/f'logs/tap-traces/{item["run"]}.jsonl'
        calibration = MusicCalibrationData(**json.loads(calibration_path.read_text())['profiles']['7@1280x720'])
        config = MusicConfig(**load_replay_config(config_path))
        identity = replay_identity(config,calibration,str(config_path))
        if (identity['config_hash']!='6200cae263aa15a7'
                or identity['effective_calibration_hash']!='20bfdc72be93e1b61d5d2d8803ca39562ac72d77b55d9c6b0be31a6dd3371714'):
            raise ValueError('Strict fixture changed configuration or calibration')
        clock = Clock()
        calls = []
        context = SimpleNamespace(run_action_direct=lambda kind,param:
            (calls.append({'time':clock(),'kind':kind.value,'contact':param.contact})
             or SimpleNamespace(success=True)))
        runtime = MusicRuntime(context,config,clock=clock,monotonic=clock,sleeper=clock.sleep)
        engine = MusicVisionEngine(calibration,config,tap_trace=runtime.tap_trace)
        executor = FixedTransport(context,1280,720,config,advanced=True,multi_touch=True,
                                  clock=clock,sleeper=clock.sleep)
        executor.begin_segment(runtime.tap_trace.run_id,0)
        provider = NumpyCandidateProvider(calibration,config.candidate_iou_threshold,
                                         config.candidate_min_size,config.split_stacked_notes)
        pending, rows, metrics = [], [], RuntimeMetrics()
        digest = hashlib.sha256()
        candidate_digest = hashlib.sha256()
        options = SimpleNamespace(video=APP_ROOT.parent/f'test-materials/music/20261003test3-{item["video_index"]}.mp4',
            start=item['start'],duration=item['end']-item['start'],
            ffmpeg=ffmpeg_binary('ffmpeg.exe'),ffprobe=ffmpeg_binary('ffprobe.exe'))
        available = memory_available()
        estimated_bytes = math.ceil((item['end']-item['start'])*31)*1280*720*3
        if available < estimated_bytes + 2*1024**3:
            raise MemoryError('Predecode budget needs estimated clip bytes plus 2GiB reserve')
        decoded = list(video_frames(options))
        decoded_bytes = sum(image.nbytes for pts,image,unused in decoded)
        for pts,image,unused in decoded:
            digest.update(struct.pack('<d',pts))
            digest.update(image)
        for sequence,(pts,image,unused) in enumerate(decoded):
            clock.now = pts
            frame = MusicFrame(sequence,pts,pts,pts,image)
            runtime._dispatch_frame = frame
            frame_begin = time.perf_counter()
            frame_begin_cpu = time.process_time()
            begin = time.perf_counter()
            begin_cpu = time.process_time()
            chain = getattr(engine,'tap_hold_chain',None)
            if chain is not None and any(e.origin=='hold_note' for e in pending):
                chain.refresh(frame)
            prepass_ms = (time.perf_counter()-begin)*1000.
            prepass_process_ms = (time.process_time()-begin_cpu)*1000.
            begin = time.perf_counter()
            begin_cpu = time.process_time()
            runtime._execute_due(executor,pending,pts,metrics,engine,wait=False)
            before_dispatch_ms = (time.perf_counter()-begin)*1000.
            before_dispatch_process_ms = (time.process_time()-begin_cpu)*1000.
            begin = time.perf_counter()
            begin_cpu = time.process_time()
            visual = VisualMask.from_image(image,calibration)
            candidates = provider.detect(frame,visual)
            detection_ms = (time.perf_counter()-begin)*1000.
            detection_process_ms = (time.process_time()-begin_cpu)*1000.
            begin = time.perf_counter()
            begin_cpu = time.process_time()
            events = engine.update(frame,candidates,visual)
            update_ms = (time.perf_counter()-begin)*1000.
            update_process_ms = (time.process_time()-begin_cpu)*1000.
            begin = time.perf_counter()
            begin_cpu = time.process_time()
            pending.extend(events)
            pending.extend(engine.release_events(pts))
            pending = engine.refine_pending(pending,pts)
            planning_ms = (time.perf_counter()-begin)*1000.
            planning_process_ms = (time.process_time()-begin_cpu)*1000.
            begin = time.perf_counter()
            begin_cpu = time.process_time()
            runtime._execute_due(executor,pending,pts,metrics,engine,wait=False)
            after_dispatch_ms = (time.perf_counter()-begin)*1000.
            after_dispatch_process_ms = (time.process_time()-begin_cpu)*1000.
            frame_cpu_ms = (time.perf_counter()-frame_begin)*1000.
            frame_process_ms = (time.process_time()-frame_begin_cpu)*1000.
            # Outside the measured intervals; preserve candidate order and all geometry/features.
            candidate_digest.update(json.dumps({'pts':pts,'candidates':[asdict(c) for c in candidates]},
                                               sort_keys=True,allow_nan=False).encode())
            rows.append({'sequence':sequence,'pts':pts,'prepass_cpu_ms':prepass_ms,
                'engine_update_cpu_ms':update_ms,'qualification_cpu_ms':prepass_ms+update_ms,
                'nonwaiting_planning_cpu_ms':planning_ms,
                'nonwaiting_dispatch_cpu_ms':before_dispatch_ms+after_dispatch_ms,
                'provider_mask_cpu_ms':detection_ms,'frame_processing_cpu_ms':frame_cpu_ms,
                'qualification_process_cpu_ms':prepass_process_ms+update_process_ms,
                'nonwaiting_planning_process_cpu_ms':planning_process_ms,
                'nonwaiting_dispatch_process_cpu_ms':before_dispatch_process_ms+after_dispatch_process_ms,
                'provider_mask_process_cpu_ms':detection_process_ms,
                'frame_processing_process_cpu_ms':frame_process_ms,
                'candidates':len(candidates),'pending':len(pending)})
        executor.release_all()
        files_after = {str(p.relative_to(branch)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((branch/'agent').rglob('*.py'))}
        tools_after = {file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in
                       (Path(__file__),APP_ROOT/'tools/tap_replay.py',APP_ROOT/'tools/workspace_paths.py')}
        report = {'schema':2,'mode':'strict-source-PTS-qualification-CPU','clip':item,
            'branch':str(branch),'source_hash':source_hash,'source_changed':files_after!=source_files,
            'source_files':source_files,**identity,'input_digest':digest.hexdigest(),
            'candidate_digest':candidate_digest.hexdigest(),'tool_files':tool_files,'tool_hash':tool_hash,
            'tool_changed':tools_after!=tool_files,
            'measurement_definition':MEASUREMENT,'measurement_hash':measurement_hash,
            'predecode':{'available_bytes':available,'estimated_bytes':estimated_bytes,
                         'actual_bytes':decoded_bytes,'ffmpeg_exited_before_measurement':True},
            'frames':len(rows),'timing_ms':metric_summary([x['qualification_cpu_ms'] for x in rows]),
            'update_only_timing_ms':metric_summary([x['engine_update_cpu_ms'] for x in rows]),
            'prepass_timing_ms':metric_summary([x['prepass_cpu_ms'] for x in rows]),'rows':rows,
            'planning_timing_ms':metric_summary([x['nonwaiting_planning_cpu_ms'] for x in rows]),
            'dispatch_timing_ms':metric_summary([x['nonwaiting_dispatch_cpu_ms'] for x in rows]),
            'provider_mask_timing_ms':metric_summary([x['provider_mask_cpu_ms'] for x in rows]),
            'frame_processing_timing_ms':metric_summary([x['frame_processing_cpu_ms'] for x in rows]),
            'process_cpu_timing_ms':{key:metric_summary([x[key] for x in rows]) for key in
                ('qualification_process_cpu_ms','nonwaiting_planning_process_cpu_ms',
                 'nonwaiting_dispatch_process_cpu_ms','provider_mask_process_cpu_ms',
                 'frame_processing_process_cpu_ms')},
            'head_records':runtime.head_action_trace,'input_calls':len(calls),
            'trace_dropped':runtime.tap_trace.dropped,
            'limitations':'Every decoded frame and PTS consumed once; FFmpeg exits before measurement. Main legacy *_cpu_ms fields actually use perf_counter elapsed wall time, subject to scheduling/frequency/cache variance, not process CPU. Auxiliary process_time statistics are explicitly separate and do not replace wall gate; OS timer granularity may affect individual small samples. Controller I/O and flick waits mocked instantaneous, using actual detected events and real head tap receipts. No musical actions injected. Does not validate physical gesture timing, scheduling/transport grades, screenshot phase or game BM. Qualification includes engine.update plus external gold prepass, excludes provider/mask/OCR/input/release/refine. Planning and nonwaiting dispatch separately include Python checks and instant mock input overhead, not real controller latency. Frame processing includes provider/mask, qualification, planning and mock dispatch, excludes decode, OCR, hashing, disk output and final cleanup; it is not an actual runtime loop deadline/latency metric.'}
        if args.compare_label:
            old = json.loads((APP_ROOT/'temp/round3-implementation'/args.compare_label/f'{name}.json').read_text())
            delta = report['timing_ms']['p95']-old['timing_ms']['p95']
            guards = {
                'same_pixels_and_pts':report['input_digest']==old.get('input_digest'),
                'same_qualification_timestamps':[(r['sequence'],r['pts']) for r in rows]==[(r['sequence'],r['pts']) for r in old['rows']],
                'same_candidate_geometry_features_order':report['candidate_digest']==old.get('candidate_digest'),
                'same_config':report['config_hash']==old.get('config_hash'),
                'same_effective_calibration':report['effective_calibration_hash']==old.get('effective_calibration_hash'),
                'source_stable':not report['source_changed'] and not old.get('source_changed',True),
                'same_mode_schema':report['mode']==old.get('mode') and report['schema']==old.get('schema'),
                'same_tool':report['tool_hash']==old.get('tool_hash'),
                'tool_stable':not report['tool_changed'] and not old.get('tool_changed',True),
                'same_measurement_definition':report['measurement_hash']==old.get('measurement_hash')}
            report['comparison'] = {'baseline_label':args.compare_label,
                'same_pixels_and_pts':report['input_digest']==old['input_digest'],
                'same_qualification_timestamps':[(r['sequence'],r['pts']) for r in rows]==[(r['sequence'],r['pts']) for r in old['rows']],
                'p95_increment_ms':delta,'guards':guards,'comparison_valid':all(guards.values()),
                'within_2ms':all(guards.values()) and delta<=2.,
                'planning_p95_increment_ms':report['planning_timing_ms']['p95']-old['planning_timing_ms']['p95'],
                'dispatch_p95_increment_ms':report['dispatch_timing_ms']['p95']-old['dispatch_timing_ms']['p95'],
                'frame_processing_p95_increment_ms':report['frame_processing_timing_ms']['p95']-old['frame_processing_timing_ms']['p95'],
                'action_alignment':'Not compared by ordinal: tracker identities and event counts may differ. This fixture validates qualification CPU only.'}
        target.write_text(json.dumps(report,indent=2),encoding='utf-8')
        shutil.copy2(target,persistent/target.name)
        summaries.append({k:report[k] for k in ('clip','frames','source_hash','source_changed','input_digest','timing_ms','update_only_timing_ms','prepass_timing_ms','planning_timing_ms','dispatch_timing_ms','provider_mask_timing_ms','frame_processing_timing_ms')}
                         | {'comparison':report.get('comparison')})
        (output/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
        shutil.copy2(output/'summary.json',persistent/'summary.json')
        print(json.dumps({'clip':name,'frames':len(rows),'qualification':report['timing_ms'],
                          'comparison':report.get('comparison')},ensure_ascii=False),flush=True)
        del decoded
    return 0


if __name__=='__main__':
    raise SystemExit(main())
