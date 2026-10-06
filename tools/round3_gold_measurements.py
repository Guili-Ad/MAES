"""Read actual video frames and measure gold/HUD appearance, with no OCR/input."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

APP_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = APP_ROOT.parent / '.work/round3-implementation-20261003'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-root', type=Path, default=APP_ROOT)
    parser.add_argument('--fixtures', type=Path, default=WORK_ROOT/'gold-fixtures.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--images', action='store_true')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Measured repeated detector calls per fixed frame; first call warms separately')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be at least one')
    if not args.output.resolve().is_relative_to((APP_ROOT/'temp').resolve()):
        parser.error('Analysis output must stay in app/temp')
    sys.path.insert(0,str(args.branch_root))
    import numpy as np
    from agent.music.models import MusicCalibrationData, MusicConfig
    from agent.music import holds
    detect_hold_tails = holds.detect_hold_tails
    gold_ring_coverage = getattr(holds, 'gold_ring_coverage', lambda image, box: None)
    from agent.music.vision import build_color_mask
    def descriptor_controls(frame,box):
        x,y,w,h = box
        patch = frame[y:y+h,x:x+w,:3]
        yy,xx = np.ogrid[:h,:w]
        dx,dy = (xx-(w-1)/2.)/max(1.,w/2.),(yy-(h-1)/2.)/max(1.,h/2.)
        radial = dx*dx+dy*dy
        rim = (radial>=.45**2)&(radial<=1.05**2)
        angles = (np.arctan2(dy,dx)+2*np.pi)%(2*np.pi)
        sectors = np.minimum(11,(angles[rim]*6/np.pi).astype(np.intp))
        totals = np.bincount(sectors,minlength=12)
        gold = build_color_mask(patch,[[7,5,145]],[[45,200,255]])
        b,g,r = (patch[...,i].astype(np.int16) for i in range(3))
        chromatic = (r>=145)&(g>=110)&(np.minimum(r,g)-b>=20)&(r-g<=100)
        answer = {'gold_hsv_fill':float(gold.mean()),'gold_center_fill':float(gold[radial<=.35**2].mean())}
        for name,mask in [('hsv_gold',gold),('chromatic_gold',chromatic)]:
            counts = np.bincount(sectors,weights=mask[rim],minlength=12)
            answer[f'{name}_coverage_25pct'] = float(((totals>0)&(counts>=totals*.25)).mean())
            answer[f'{name}_coverage_5pct'] = float(((totals>0)&(counts>=totals*.05)).mean())
        return answer
    package = WORK_ROOT/'baseline/candidate-state'
    cal = MusicCalibrationData(**json.loads((package/'user-data/calibration/music.json').read_text())['profiles']['7@1280x720'])
    trace = package/'logs/tap-traces/20261003T114448Z-63e80544.jsonl'
    with trace.open() as stream:
        config = MusicConfig(**json.loads(next(stream))['config'])
    ffmpeg = APP_ROOT.parent/'.work/ffmpeg-7.1.1-extract/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe'
    flags = getattr(subprocess,'CREATE_NO_WINDOW',0)
    output = args.output.parent
    output.mkdir(parents=True,exist_ok=True)
    rows = []
    all_samples = []
    for item in json.loads(args.fixtures.read_text()):
        video = APP_ROOT.parent/f'test-materials/music/20261003test3-{item["video_index"]}.mp4'
        command = [str(ffmpeg),'-v','error','-ss',str(item['time']),'-i',str(video),
                   '-frames:v','1','-vf','scale=1280:720:flags=area','-f','rawvideo','-pix_fmt','bgr24','pipe:1']
        raw = subprocess.check_output(command,creationflags=flags)
        if len(raw)!=1280*720*3:
            raise RuntimeError(f'Missing complete video frame at {item}')
        frame = np.frombuffer(raw,np.uint8).reshape(720,1280,3)
        detections = detect_hold_tails(frame,cal,config)
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            detections = detect_hold_tails(frame,cal,config)
            samples.append((time.perf_counter()-start)*1000.)
        all_samples.extend(samples)
        elapsed = float(np.median(samples))
        # These are descriptor controls, not permanent exclusion ROIs.
        controls = [{'name':'central grade wide','box':(600,440,80,32)},
                    {'name':'central grade square component','box':(613,443,36,36)},
                    {'name':'grade left small score','box':(604,465,30,24)}]
        for control in controls:
            control['ring_coverage'] = gold_ring_coverage(frame,control['box'])
            control['alternative_controls'] = descriptor_controls(frame,control['box'])
        detected_rows = []
        for detection in detections:
            detected = asdict(detection)
            box = getattr(detection, 'box', None)
            detected['alternative_controls'] = descriptor_controls(frame,box) if box else None
            detected_rows.append(detected)
        record = {**item,'video':str(video),'detector_cpu_ms':elapsed,
                  'detector_timing_ms':{'count':len(samples),'p50':elapsed,
                                       'p95':float(np.percentile(samples,95)),'max':max(samples)},
                  'detections':detected_rows,'hud_descriptor_controls':controls,
                  'limitations':'Compressed area-scaled frame; controls are not hand-labelled object segmentation. No BM inference.'}
        if args.images:
            image = output/f'gold-{item["video_index"]}-{item["time"]:.2f}.png'
            subprocess.run([str(ffmpeg),'-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','1280x720',
                            '-i','pipe:0','-frames:v','1','-y',str(image)],input=raw,check=True,creationflags=flags)
            record['image'] = str(image)
        rows.append(record)
        print(json.dumps({'video':item['video_index'],'time':item['time'],
                          'detections':[{k:d.get(k) for k in ('center','box','ring_coverage','topology','owner_lanes')}
                                        for d in record['detections']]},ensure_ascii=False),flush=True)
    descriptor_source = args.branch_root/'agent/music/holds.py'
    report = {'descriptor_source':str(descriptor_source),
              'descriptor_source_sha256':hashlib.sha256(descriptor_source.read_bytes()).hexdigest(),
              'fixture_file':str(args.fixtures),'frames':rows,
              'repeats_per_frame':args.repeats,
              'detector_timing_ms':{'count':len(all_samples),'p50':float(np.median(all_samples)),
                                   'p95':float(np.percentile(all_samples,95)),'max':max(all_samples)},
              'meaning':'Ring coverage is an angular occupancy descriptor, not sufficient physical identity or ribbon ownership.'}
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
