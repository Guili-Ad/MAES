"""Fixed-input comparison of association, grouping, dispatch and trace overhead."""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=200)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / 'temp'):
        parser.error('Output must stay under Double/temp')
    sys.path.insert(0, str(args.branch_root))
    sys.path.insert(0, str(args.branch_root / 'tests'))
    from test_longtap_branch import calibration, candidate_at, linked_pair_frame
    from agent.music.models import MusicConfig, NoteTrack, TrackObservation, MusicFrame, MusicActionEvent, NoteGesture
    from agent.music.vision import VisualMask
    from agent.music.tracking import MusicVisionEngine, LaneProjection
    from agent.music.runtime import MusicRuntime, RuntimeMetrics
    from agent.music.executor import MusicActionExecutor
    from agent.music.storage import metric_summary
    logging.disable(logging.CRITICAL)
    cal, config = calibration(), MusicConfig(lane_count=7, enable_holds=True)
    image = linked_pair_frame(candidate_at(cal, 1, .64), candidate_at(cal, 5, .64))
    visual = VisualMask.from_image(image, cal)
    context = SimpleNamespace(run_action_direct=lambda *args, **kwargs: SimpleNamespace(success=True))
    times = []
    for iteration in range(args.iterations + 20):
        engine = MusicVisionEngine(cal, config)
        runtime = MusicRuntime(context, config, clock=lambda: 1.5)
        executor = MusicActionExecutor(context, 1280, 720, config, advanced=True, multi_touch=True, clock=lambda: 1.5)
        entries, pending = {}, []
        tid = 0
        for lane, positions in ((1, [.6]), (2, [.7, .6, .5]), (3, [.7, .6, .5]), (5, [.6])):
            for progress in positions:
                tid += 1
                track = NoteTrack(tid, lane, speed=.8, predicted_hit_time=1.5)
                for seq in range(3):
                    p = progress - (2 - seq) * .024
                    note = candidate_at(cal, lane, p)
                    track.observations.append(TrackObservation(seq, .94 + seq * .03, note.center, p, note))
                engine.tracks[tid] = track
                p = progress + .04
                entries.setdefault(lane, []).append((candidate_at(cal, lane, p), LaneProjection(lane, p, 0., (0., 1.))))
                pending.append(MusicActionEvent(str(tid), tid, lane, NoteGesture.TAP, 1.5, tuple(cal.points[lane])))
        frame = MusicFrame(3, 1.05, 1.05, 1.05, image)
        if hasattr(engine, 'last_frame_sequence'):
            engine.last_frame_sequence = 3
        started = time.perf_counter()
        for lane, candidates in entries.items():
            engine._associate_lane(lane, candidates, frame, visual)
        engine._update_linked_tap_pairs(frame)
        if hasattr(engine, 'tap_trace'):
            from agent.music.tap_tracking import retire_converged_shadows
            retire_converged_shadows(engine.tracks, frame, engine.tap_trace)
        engine._stabilize_dense_tap_timing(frame)
        pending = engine.refine_pending(pending, 1.05)
        # All fixed-input events are due now; no precision sleeping occurs.
        runtime._execute_due(executor, pending, 1.5, RuntimeMetrics(), engine)
        elapsed = (time.perf_counter() - started) * 1000.
        if iteration >= 20:
            times.append(elapsed)
    result = {'branch': str(args.branch_root), 'input': '8 mature taps, 4 lanes, two dense triples and one arc pair',
              'included': 'association + arc pairing + dense marking + refinement + mock input/ack/trace',
              'excluded': 'screenshot, video decoder, provider, hold route detection, filesystem output',
              'milliseconds': metric_summary(times)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
