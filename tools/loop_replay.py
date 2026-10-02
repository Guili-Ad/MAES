"""Replay production MusicRuntime.play, with deterministic delayed/failing stages.

No controller is created. Capture selects the latest frame available at the
simulated capture return time. Costs are synthetic, not measured game timing.
"""
import json
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch


def run_loop(stream, calibration, config, args):
    from agent.music import runtime as runtime_module
    from agent.music.runtime import MusicRuntime
    from agent.music.models import MusicFrame
    from agent.music.executor import MusicActionExecutor
    from agent.music.tracking import MusicVisionEngine
    from agent.music.vision import NumpyCandidateProvider
    from agent.music.storage import metric_summary
    from tap_replay import decode_candidate

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
                                   clock=clock, sleeper=runtime._sleep_interruptibly)
    engines, scheduled, pending_snapshot, samples = [], [], [], []
    def new_engine(*positional, **keywords):
        engine = MusicVisionEngine(*positional, **keywords)
        original = engine.update
        def update(frame, candidates, visual):
            begin = time.perf_counter()
            events = original(frame, candidates, visual)
            samples.append((time.perf_counter()-begin)*1000.)
            clock.sleep(context.costs.get('tracking_ms', 0.) / 1000.)
            scheduled.extend({'track': e.track_id, 'gesture': e.gesture.value, 'lane': e.lane,
                              'deadline': e.deadline} for e in events)
            return events
        engine.update = update
        engines.append(engine)
        return engine
    original_execute = runtime._execute_due
    def execute(executor, pending, *positional, **keywords):
        result = original_execute(executor, pending, *positional, **keywords)
        pending_snapshot[:] = pending
        return result
    first = context.capture(context, 0, clock)[0]
    if first is None:
        raise ValueError('Initial replay capture failed; production startup has no valid frame')
    runtime._execute_due = execute
    with patch.object(runtime, 'startup_gate', return_value=None), \
         patch.object(runtime, 'activate_play_provider', return_value=None), \
         patch.object(runtime, '_create_executor', return_value=executor), \
         patch.object(runtime.tap_trace, 'write', return_value='mock/no-disk'), \
         patch.object(runtime_module, '_capture_frame', side_effect=context.capture), \
         patch.object(runtime_module, 'MusicVisionEngine', side_effect=new_engine):
        result = runtime.play(first)
    heads = []
    for row in runtime.head_action_trace:
        ordinal, track, lane, gesture, timestamp, late, flags = row.split(':')
        heads.append({'ordinal': int(ordinal), 'track': int(track), 'lane': int(lane), 'gesture': gesture,
                      'time': float(timestamp), 'late_ms': float(late), 'flags': flags})
    return {'schema': 2, 'mode': 'production-loop', 'branch': str(args.branch_root.resolve()),
            'source': str(args.video or args.candidates), 'provider': runtime.provider.name,
            'frames': context.decoded, 'captures': context.captures, 'cost_profile': profile,
            'heads': heads, 'actions': context.actions, 'scheduled': scheduled,
            'pending_at_clip_end': len(pending_snapshot), 'timing_ms': metric_summary(samples),
            'metrics_ms': runtime.metrics.summaries(executor.action_durations),
            'stop': {'status': result.status, 'reason': result.reason, 'cleanup_failure': runtime.cleanup_failure},
            'game_bad_miss': 'unavailable: compressed video / known candidates, simulated input and OCR',
            'trace': list(runtime.tap_trace.records), 'observations': []}
