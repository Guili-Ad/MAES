"""Two near-line tap recoveries, including trace and unchanged motion fitting."""
import sys
import time
import json
import copy
from pathlib import Path
from dataclasses import replace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
from test_longtap_branch import calibration, candidate_at
from agent.music.models import MusicConfig, MusicFrame, NoteTrack, TrackObservation
from agent.music.tracking import MusicVisionEngine, assign_lane
from agent.music.tap_recovery import recover_masked_taps
from agent.music.storage import metric_summary
from agent.music.vision import VisualMask


def main():
    cal = replace(calibration(), exclusion_rois=[[410, 550, 130, 140], [730, 550, 130, 140]])
    image = np.zeros((720, 1280, 3), np.uint8)
    yy, xx = np.ogrid[:720, :1280]
    tracks = {}
    for lane in (2, 4):
        track = NoteTrack(lane, lane)
        for i, p in enumerate([.76, .80, .84]):
            c = candidate_at(cal, lane, p, 60)
            track.observations.append(TrackObservation(i, i*.05, c.center, p, c))
        tracks[lane] = track
        note = candidate_at(cal, lane, .88, 64)
        d = (xx-note.center[0])**2+(yy-note.center[1])**2
        image[d < 32**2] = (180, 220, 20)
        image[d < 7**2] = 255
    frame = MusicFrame(3, .15, .15, .15, image)
    visual = VisualMask.from_image(image, cal)
    times = []
    for i in range(220):
        engine = MusicVisionEngine(cal, MusicConfig())
        engine.tracks = copy.deepcopy(tracks)
        started = time.perf_counter()
        recovered = recover_masked_taps(engine.tracks, frame, cal, lambda c: assign_lane(c, cal))
        for lane in (2,4):
            engine._associate_lane(lane, [], frame, visual, recovered)
        elapsed = (time.perf_counter()-started)*1000
        assert len(engine.tap_trace.records) == 2
        if i >= 20:
            times.append(elapsed)
    result = {'input': 'two confirmed ordinary heads inside judgement exclusion ROIs',
              'milliseconds': metric_summary(times), 'iterations': len(times),
              'includes': 'crop/color/components/topology/ownership/motion/trace',
              'excludes': 'test setup, capture/provider, disk output', 'real_game_validation': False}
    output = ROOT/'temp/validation/tap-identity-v3/recovery-bench.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
