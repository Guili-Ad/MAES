"""Compare hold-marker CPU cost on identical recorded pixels, excluding decode."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from tap_replay import video_frames

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch-root',type=Path,default=ROOT)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--start',type=float,required=True)
    p.add_argument('--duration',type=float,default=2.)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'temp'):
        p.error('Output must remain under Double/temp')
    binary = WORKSPACE/'.work/ffmpeg-7.1.1-extract/ffmpeg-7.1.1-essentials_build/bin'
    args.ffmpeg, args.ffprobe = binary/'ffmpeg.exe',binary/'ffprobe.exe'
    sys.path.insert(0,str(args.branch_root.resolve()))
    from agent.music.holds import detect_hold_tails
    from agent.music.models import MusicCalibrationData, MusicConfig
    from agent.music.storage import metric_summary
    raw=json.loads(args.calibration.read_text(encoding='utf-8'))
    cal=MusicCalibrationData(**raw['profiles']['7@1280x720'])
    config=MusicConfig(lane_count=7,enable_holds=True)
    frames=list(video_frames(args))
    times=[]
    for repeat in range(4):
        for _, image, _ in frames:
            start=time.perf_counter()
            detect_hold_tails(image,cal,config)
            if repeat:
                times.append((time.perf_counter()-start)*1000.)
    report={'branch':str(args.branch_root), 'frames':len(frames), 'timing_ms':metric_summary(times),
            'included':'all marker detection, topology and owner evidence; decoder excluded'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    main()
