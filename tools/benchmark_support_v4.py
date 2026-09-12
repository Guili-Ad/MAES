"""Fixed input, serial-process comparison including owner-only pixel recovery."""
import argparse,copy,json,sys,time,logging
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch-root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'temp'):
        p.error('Output must stay under Double/temp')
    sys.path.insert(0,str(args.branch_root));sys.path.insert(0,str(args.branch_root/'tests'))
    from test_longtap_branch import calibration,candidate_at
    from agent.music.models import MusicConfig,NoteTrack,TrackObservation,MusicFrame
    from agent.music.tracking import MusicVisionEngine,assign_lane
    from agent.music.tap_recovery import recover_masked_taps
    from agent.music.vision import VisualMask
    from agent.music.storage import metric_summary
    logging.disable(logging.CRITICAL)
    cal=calibration();config=MusicConfig(enable_holds=True)
    image=np.zeros((720,1280,3),np.uint8)
    yy,xx=np.ogrid[:720,:1280]
    tracks={}
    for lane in (2,3,4):
        track=NoteTrack(lane,lane,speed=.8)
        for seq,progress in enumerate([.48,.52,.56]):
            c=candidate_at(cal,lane,progress,60)
            track.observations.append(TrackObservation(seq,seq*.05,c.center,progress,c))
        tracks[lane]=track
        c=candidate_at(cal,lane,.60,64)
        d=(xx-c.center[0])**2+(yy-c.center[1])**2
        image[d<32**2]=(180,220,20);image[d<7**2]=255
    for tid in range(10,110):
        track=NoteTrack(tid,tid%7)
        for seq in range(3):
            c=candidate_at(cal,track.lane,.65,20)
            track.observations.append(TrackObservation(seq,seq*.05,c.center,.65,c))
        tracks[tid]=track
    frame=MusicFrame(3,.15,.15,.15,image)
    visual=VisualMask.from_image(image,cal)
    samples=[]
    for iteration in range(220):
        engine=MusicVisionEngine(cal,config);engine.tracks=copy.deepcopy(tracks)
        start=time.perf_counter()
        recovered=recover_masked_taps(engine.tracks,frame,cal,lambda c:assign_lane(c,cal))
        for lane in (2,3,4):engine._associate_lane(lane,[],frame,visual,recovered)
        elapsed=(time.perf_counter()-start)*1000
        if iteration>=20:samples.append(elapsed)
    result={'branch':str(args.branch_root),'iterations':200,'milliseconds':metric_summary(samples),
            'input':'3 mid-flight teal heads, 100 retained stationary fragments, same fixed image',
            'includes':'pixel recovery, continuity/ownership, lane association, motion fit and bounded trace',
            'excludes':'setup, screencap/provider, decoding, filesystem output; not preview FPS'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
