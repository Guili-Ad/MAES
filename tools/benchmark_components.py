"""Equivalent component hotspot benchmark on actual recorded masks, no input."""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
sys.path.insert(0, str(ROOT/'tools'))
from tap_replay import video_frames
from workspace_paths import ffmpeg_binary
from agent.music.models import MusicCalibrationData
from agent.music.vision import VisualMask, connected_components
from agent.music.storage import metric_summary
from legacy_components import connected_components as original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'temp'):
        parser.error('Output must stay under app/temp')
    raw = json.loads(args.calibration.read_text(encoding='utf-8'))
    if 'profiles' in raw:
        raw = raw['profiles']['7@1280x720']
    calibration = MusicCalibrationData(**raw)
    masks = []
    request = SimpleNamespace(video=args.video, start=25., duration=3.,
                              ffmpeg=ffmpeg_binary('ffmpeg.exe'), ffprobe=ffmpeg_binary('ffprobe.exe'))
    for i, (_, image, _) in enumerate(video_frames(request)):
        if i % 6 == 0:
            masks.append(VisualMask.from_image(image, calibration).mask)
    if not masks:
        raise ValueError('No recorded masks decoded')
    before, after, equivalent = [], [], True
    for _ in range(6):
        for mask in masks:
            started = time.perf_counter()
            a = original(mask, calibration.candidate_min_pixels)
            before.append((time.perf_counter()-started)*1000.)
            started = time.perf_counter()
            b = connected_components(mask, calibration.candidate_min_pixels)
            after.append((time.perf_counter()-started)*1000.)
            equivalent &= a == b
    report = {'source':str(args.video), 'masks':len(masks), 'samples':len(before),
              'equivalent':equivalent, 'baseline_ms':metric_summary(before),
              'candidate_ms':metric_summary(after), 'scope':'CPU component extraction only, not game FPS'}
    report['passed'] = equivalent and report['candidate_ms']['p95'] < report['baseline_ms']['p95']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
