"""Fixed-song repairs: timing is host timing, never a game-grade assertion."""
import os
os.environ.setdefault('MAES_AGENT_TEST_MODE', '1')
import sys
from pathlib import Path
from dataclasses import replace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from agent.music.models import MusicConfig, MusicFrame, MusicCandidate, TrackObservation, NoteGesture
from agent.music.tap_identity import coastable_tap
from agent.music.tap_tracking import associate_taps
from agent.music.tap_trace import TapTrace
from agent.music.tracking import LaneProjection, MusicVisionEngine
from agent.music.runtime import MusicRuntime, RuntimeMetrics, _recognize
from agent.music import runtime as runtime_module
from agent.music.pause_gate import pause_overlay_possible
from test_tap_pipeline import observed_track, tap_event
from test_longtap_branch import calibration


class FixedSongRegressionTests(unittest.TestCase):
    def test_collapsed_two_sample_particle_cannot_coast_into_extra_tap(self):
        track=observed_track(39,4,.818,hit=30.47)
        track.observations.clear()
        for i,(stamp,progress,size) in enumerate(((30.267,.818,86),(30.4,.937,18))):
            note=MusicCandidate((768,484+i*120,size,size),size*size,.8,(768+size/2,484+i*120+size/2))
            track.observations.append(TrackObservation(i,stamp,note.center,progress,note))
        self.assertFalse(coastable_tap(track,MusicConfig(),now=30.533))

    def test_identical_head_pixels_keep_owner_without_creating_shadow(self):
        track=observed_track(34,4,.818,speed=1.5)
        track.tap_input_started=1.01
        previous=track.observations[-1]
        frame=MusicFrame(3,1.033,1.033,1.033,np.zeros((720,1280,3),np.uint8))
        projection=LaneProjection(4,previous.progress,0.,(0.,1.))
        result=associate_taps([track],[(previous.candidate,projection)],frame,
                              MusicConfig(),lambda *_:True,TapTrace(MusicConfig()))
        self.assertEqual({i:t.track_id for i,t in result.items()},{0:34})

    def test_live_pixels_do_not_start_expensive_pause_ocr(self):
        runtime=MusicRuntime(SimpleNamespace(),MusicConfig())
        image=np.full((720,1280,3),[180,130,70],np.uint8)
        with patch('agent.music.runtime._recognize',return_value=False) as recognize:
            check=getattr(runtime,'pause_dialog_present',lambda im:runtime_module._recognize(runtime.context,'MusicPauseDialog',im))
            self.assertFalse(check(image))
        recognize.assert_not_called()

    def test_possible_pause_has_two_independent_positive_routes(self):
        background=np.full((720,1280,3),[180,130,70],np.uint8)
        for patch_box, color in (((240,425,770,930),[210,210,210]),
                                  ((165,210,590,690),[35,130,65])):
            image=background.copy()
            y0,y1,x0,x1=patch_box
            image[y0:y1,x0:x1]=color
            self.assertTrue(pause_overlay_possible(image))
            runtime=MusicRuntime(SimpleNamespace(),MusicConfig())
            with patch('agent.music.runtime._recognize',return_value=True) as recognize:
                self.assertTrue(runtime.pause_dialog_present(image))
            recognize.assert_called_once()

    def test_unknown_dark_formats_and_white_flashes_keep_ocr_confirmation(self):
        for image in (None,'paused',np.zeros((720,1280,3),np.uint8),
                       np.zeros((1080,1920,3),np.uint8),np.full((720,1280,3),240,np.uint8)):
            self.assertTrue(pause_overlay_possible(image))

    def test_contour_repeat_does_not_change_speed_fit_or_add_observation(self):
        cal=calibration()
        engine=MusicVisionEngine(cal,MusicConfig(lane_count=7))
        track=observed_track(1,3,.6,speed=1.,hit=1.4)
        engine.tracks[1]=track
        engine.next_track_id=2
        note=track.observations[-1].candidate
        before=list(track.observations)
        from agent.music.vision import VisualMask
        image=np.zeros((720,1280,3),np.uint8)
        frame=MusicFrame(3,1.03,1.03,1.03,image)
        visual=VisualMask.from_image(image,cal)
        engine._associate_lane(3,[(note,LaneProjection(3,.6,0.,(0.,1.)))],frame,visual)
        self.assertEqual(list(track.observations),before)
        self.assertEqual((track.speed,track.predicted_hit_time),(1.,1.4))
        self.assertEqual(engine.next_track_id,2)

    def test_hold_bonus_and_static_contours_do_not_use_tap_repeat_rule(self):
        from agent.music.tap_tracking import repeated_head_pixels
        frame=MusicFrame(3,1.03,1.03,1.03,np.zeros((720,1280,3),np.uint8))
        for kind in ('hold','bonus','static'):
            track=observed_track(1,3,.6)
            if kind=='hold': track.gesture=NoteGesture.HOLD_START
            elif kind=='bonus': track.bonus_star=True
            else:
                values=[replace(observation,progress=.6) for observation in track.observations]
                track.observations.clear()
                track.observations.extend(values)
            self.assertFalse(repeated_head_pixels(track,track.observations[-1].candidate,frame))

    def test_red_flick_cannot_be_taken_over_by_blue_background_component(self):
        track=observed_track(1,0,.6,speed=1.)
        track.flick=True
        track.flick_color='red'
        track.flick_direction=NoteGesture.FLICK_LEFT
        track.gesture=NoteGesture.FLICK_LEFT
        note=replace(track.observations[-1].candidate,variant='flick',
                     flick_direction=NoteGesture.FLICK_RIGHT,flick_color='blue')
        frame=MusicFrame(3,1.03,1.03,1.03,None)
        matches=associate_taps([track],[(note,LaneProjection(0,.63,0.,(0.,1.)))],frame,
                               MusicConfig(),lambda *_:False,TapTrace(MusicConfig()))
        self.assertEqual(matches,{})

    def test_future_unfrozen_tap_yields_another_frame_beyond_capture_guard(self):
        class Clock:
            now=1.
            def __call__(self):
                self.now+=.00001
                return self.now
            def sleep(self,seconds):
                self.now+=seconds
        clock=Clock()
        runtime=MusicRuntime(SimpleNamespace(),MusicConfig(),clock=clock,sleeper=clock.sleep)
        pending=[tap_event(1,3,1.08)]
        with patch.object(runtime,'_acknowledge_taps'), patch.object(runtime,'sleeper',wraps=clock.sleep) as sleep:
            kwargs={'tap_wait_ms':45.} if 'tap_wait_ms' in runtime._execute_due.__code__.co_varnames else {}
            runtime._execute_due(SimpleNamespace(tap_many=lambda *a,**k:[],supports_holds=True,async_input=False),
                                 pending,1.,RuntimeMetrics(),**kwargs)
        sleep.assert_not_called()
        self.assertEqual(len(pending),1)

    def test_tap_wait_budget_does_not_change_hold_head_window(self):
        from agent.music.models import MusicActionEvent
        class Clock:
            now=1.
            def __call__(self):
                self.now+=.00001
                return self.now
            def sleep(self,seconds): self.now+=seconds
        clock=Clock()
        runtime=MusicRuntime(SimpleNamespace(),MusicConfig(hold_notes_as_taps=True),
                             clock=clock,sleeper=clock.sleep)
        event=MusicActionEvent('hold',1,3,NoteGesture.HOLD_START,1.08,(640,620))
        executor=SimpleNamespace(tap_many=lambda *a,**k:[],supports_holds=True,async_input=False)
        pending=[event]
        runtime._execute_due(executor,pending,1.,RuntimeMetrics(),tap_wait_ms=45.)
        self.assertFalse(pending)
        self.assertGreaterEqual(clock.now,1.08)


if __name__=='__main__':
    unittest.main()
