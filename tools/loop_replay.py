"""Replay production MusicRuntime.play, with deterministic delayed/failing stages.

No controller is created. Capture selects the latest frame available at the
simulated capture return time. Costs are synthetic, not measured game timing.
"""
import json
import time
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch


def run_loop(stream, calibration, config, args):
    from agent.music import runtime as runtime_module
    from agent.music.runtime import MusicRuntime
    from agent.music.models import MusicFrame
    from agent.music.executor import MusicActionExecutor
    from agent.music.tracking import MusicVisionEngine
    from agent.music.vision import NumpyCandidateProvider, VisualMask
    from agent.music.storage import metric_summary
    from tap_replay import decode_candidate
    try:
        from agent.music.tap_hold_chain import TapHoldChain
    except ImportError:  # Frozen baseline does not have the new tap-mode chain.
        TapHoldChain = None

    profile = json.loads(args.cost_profile.read_text(encoding='utf-8')) if args.cost_profile else {}
    class Clock:
        now = 0.0
        def __call__(self):
            # Production precision spin needs a progressing host clock.
            self.now += .000001
            return self.now
        def sleep(self, seconds):
            self.now += max(0., seconds)
    clock = Clock()
    iterator = iter(stream)
    following = next(iterator, None)
    if following is None:
        raise ValueError('No replay frames')
    clock.now = following[0]
    class Context:
        tasker = SimpleNamespace(stopping=False)
        actions = []
        captures = 0
        decoded = 0
        latest = None
        costs = {}
        capture_pts = {}
        def capture(self, _context, sequence, _clock, timeout_ms=1200):
            nonlocal following
            begin = clock()
            self.costs = dict(profile.get('default', {}))
            self.costs.update(profile.get('by_capture', {}).get(str(self.captures), {}))
            self.captures += 1
            clock.sleep(self.costs.get('capture_ms', 18.) / 1000.)
            if following is not None and self.latest is None:
                clock.now = max(clock.now, following[0])
            while following is not None and following[0] <= clock.now:
                self.latest = following
                self.decoded += 1
                following = next(iterator, None)
            finish = clock()
            if self.costs.get('capture_failure'):
                return None, (finish-begin)*1000.
            if following is None and (self.latest is None or clock.now > self.latest[0] + .10):
                self.tasker.stopping = True
                return None, (finish-begin)*1000.
            image = self.latest[1]
            self.capture_pts[sequence] = self.latest[0]
            return MusicFrame(sequence, begin, finish, (begin+finish)/2., image), (finish-begin)*1000.
        def run_recognition(self, node, image):
            clock.sleep(self.costs.get('ocr_ms', 0.) / 1000.)
            ui = self.costs.get('ui', 'live')
            hit = ((node == 'MusicPauseDialog' and ui == 'pause') or
                   (node in ('MusicLiveScreen', 'MusicLiveClearScreen') and ui == 'live') or
                   (node == 'MusicResultLoading' and ui == 'loading') or
                   (node == 'MusicResultLive' and ui == 'LIVE'))
            return SimpleNamespace(hit=hit)
        def run_action_direct(self, kind, param):
            index = len(self.actions)
            self.actions.append({'time': clock(), 'action': kind.value,
                                 'contact': getattr(param, 'contact', 0),
                                 'target': getattr(param, 'target', None)})
            clock.sleep(self.costs.get('action_ms', args.action_ms) / 1000.)
            return SimpleNamespace(success=index not in profile.get('action_failures', []))
    context = Context()
    actual_provider = NumpyCandidateProvider(calibration, config.candidate_iou_threshold,
                                             config.candidate_min_size, config.split_stacked_notes)
    class Provider:
        name = 'simulated-numpy' if args.video else 'known-candidates'
        def detect(self, frame, visual):
            clock.sleep(context.costs.get('provider_ms', 12.) / 1000.)
            if context.costs.get('provider_failure'):
                raise RuntimeError('Injected provider failure')
            raw = context.latest[2]
            return actual_provider.detect(frame, visual) if raw is None else [decode_candidate(c) for c in raw]
    runtime = MusicRuntime(context, replace(config, max_duration_seconds=600), clock=clock,
                           monotonic=clock, sleeper=clock.sleep)
    runtime.calibration = calibration
    runtime.provider = Provider()
    runtime.input_mode = 'advanced'
    executor = MusicActionExecutor(context, 1280, 720, config, advanced=True, multi_touch=True,
                                   clock=clock, sleeper=getattr(runtime, '_sleep_interruptibly', clock.sleep))
    engines, scheduled, pending_snapshot, samples, observations = [], [], [], [], []
    cpu_frames, update_depth = {}, [0]
    def cpu_frame(frame):
        segment = getattr(runtime.tap_trace, 'segment_id', 0)
        key = (segment, frame.sequence)
        return cpu_frames.setdefault(key, {
            'segment': segment, 'sequence': frame.sequence,
            'source_pts': context.capture_pts.get(frame.sequence),
            'capture_started': frame.capture_started, 'capture_finished': frame.capture_finished,
            'midpoint': frame.midpoint, 'engine_update_cpu_ms': 0.,
            'identity_prepass_cpu_ms': 0., 'gold_refresh_inline_cpu_ms': 0.,
            'gold_refresh_prepass_calls': 0, 'gold_refresh_inline_calls': 0})
    def refresh(chain, frame, *positional, **keywords):
        # The pre-dispatch qualification lives outside engine.update. Count it
        # separately; nested refresh is already in update's elapsed CPU time.
        nested = update_depth[0] > 0
        begin = time.perf_counter()
        try:
            return original_refresh(chain, frame, *positional, **keywords)
        finally:
            elapsed = (time.perf_counter()-begin)*1000.
            row = cpu_frame(frame)
            row['gold_refresh_inline_cpu_ms' if nested else 'identity_prepass_cpu_ms'] += elapsed
            row['gold_refresh_inline_calls' if nested else 'gold_refresh_prepass_calls'] += 1
    original_refresh = TapHoldChain.refresh if TapHoldChain is not None else None
    def new_engine(*positional, **keywords):
        engine = MusicVisionEngine(*positional, **keywords)
        original = engine.update
        def update(frame, candidates, visual):
            observations.append({'segment': getattr(runtime.tap_trace, 'segment_id', 0), 'sequence':frame.sequence,
                                 'capture_finished':frame.capture_finished,
                                 'source_pts':context.capture_pts.get(frame.sequence)})
            begin = time.perf_counter()
            update_depth[0] += 1
            try:
                events = original(frame, candidates, visual)
            finally:
                update_depth[0] -= 1
                elapsed = (time.perf_counter()-begin)*1000.
                samples.append(elapsed)
                cpu_frame(frame)['engine_update_cpu_ms'] += elapsed
            clock.sleep(context.costs.get('tracking_ms', 0.) / 1000.)
            scheduled.extend({'track': e.track_id, 'gesture': e.gesture.value, 'lane': e.lane,
                              'deadline': e.deadline} for e in events)
            return events
        engine.update = update
        engines.append(engine)
        return engine
    original_execute = runtime._execute_due
    original_mask = VisualMask.from_image
    def mask(image, calibration):
        clock.sleep(context.costs.get('mask_ms', 0.) / 1000.)
        return original_mask(image, calibration)
    def execute(executor, pending, *positional, **keywords):
        result = original_execute(executor, pending, *positional, **keywords)
        pending_snapshot[:] = pending
        return result
    first = context.capture(context, 0, clock)[0]
    if first is None:
        raise ValueError('Initial replay capture failed; production startup has no valid frame')
    runtime._execute_due = execute
    with ExitStack() as stack, \
         patch.object(runtime, 'startup_gate', return_value=None), \
         patch.object(runtime, 'activate_play_provider', return_value=None), \
         patch.object(runtime, '_create_executor', return_value=executor), \
         patch.object(runtime.tap_trace, 'write', return_value='mock/no-disk'), \
         patch.object(VisualMask, 'from_image', side_effect=mask), \
         patch.object(runtime_module, '_capture_frame', side_effect=context.capture), \
         patch.object(runtime_module, 'MusicVisionEngine', side_effect=new_engine):
        if TapHoldChain is not None:
            stack.enter_context(patch.object(TapHoldChain, 'refresh', new=refresh))
        result = runtime.play(first)
    heads = []
    for row in runtime.head_action_trace:
        ordinal, track, lane, gesture, timestamp, late, flags = row.split(':')
        heads.append({'ordinal': int(ordinal), 'track': int(track), 'lane': int(lane), 'gesture': gesture,
                      'time': float(timestamp), 'late_ms': float(late), 'flags': flags})
    cpu_rows = list(cpu_frames.values())
    for row in cpu_rows:
        row['total_tracking_cpu_ms'] = row['engine_update_cpu_ms'] + row['identity_prepass_cpu_ms']
    return {'schema': 3, 'mode': 'production-loop', 'branch': str(args.branch_root.resolve()),
            'source': str(args.video or args.candidates), 'provider': runtime.provider.name,
            'frames': context.decoded, 'captures': context.captures, 'cost_profile': profile,
            'heads': heads, 'actions': context.actions, 'scheduled': scheduled,
            'pending_at_clip_end': len(pending_snapshot), 'timing_ms': metric_summary(samples),
            'full_tracking_timing_ms': metric_summary([row['total_tracking_cpu_ms'] for row in cpu_rows]),
            'identity_prepass_timing_ms': metric_summary([row['identity_prepass_cpu_ms'] for row in cpu_rows]),
            'tracking_cpu_by_frame': cpu_rows,
            'cpu_timing_notes': 'timing_ms is legacy engine.update only; full_tracking_timing_ms adds external gold qualification prepass without counting inline refresh twice. Excludes provider, mask, OCR, input and release_events/refine_pending queue planning. Actual perf_counter CPU time includes probe overhead. metrics_ms is simulated host-clock timing, not measured CPU. Decoded PTS are identical input, but serviced snapshots/action sequences can differ.',
            'metrics_ms': runtime.metrics.summaries(executor.action_durations),
            'stop': {'status': result.status, 'reason': result.reason, 'cleanup_failure': runtime.cleanup_failure},
            'game_bad_miss': 'unavailable: compressed video / known candidates, simulated input and OCR',
            'trace': list(runtime.tap_trace.records), 'observations': observations}
