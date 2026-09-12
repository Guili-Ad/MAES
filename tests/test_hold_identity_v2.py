"""Regressions for the September 6 SUPPORT-converted failures."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_longtap_branch import calibration, candidate_at, tail_frame
from agent.music.models import MusicConfig, MusicFrame, NoteTrack, NoteGesture, TrackState, TrackObservation
from agent.music.tracking import MusicVisionEngine, LaneProjection
from agent.music.holds import HoldTailDetection
from agent.music.tap_tracking import associate_taps
from agent.music.head_identity import unique_head_candidates
from agent.music.hold_topology import marker_evidence


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.cal = calibration()
        self.engine = MusicVisionEngine(self.cal, MusicConfig(lane_count=7, enable_holds=True))

    def frame(self, seq, t, image=None):
        return MusicFrame(seq, t, t, t, np.zeros((720, 1280, 3), np.uint8) if image is None else image)

    def hold(self):
        track = NoteTrack(1, 3, gesture=NoteGesture.HOLD_START, state=TrackState.HOLDING,
                          predicted_hit_time=0., hold_release_time=1.8)
        self.engine.tracks[1] = track
        return track

    def test_mature_checkpoint_revokes_an_acquired_terminal(self):
        track = self.hold()
        for i, p in enumerate([.3, .4, .5], 1):
            self.engine._record_active_hold_tail(track, self.frame(i, i*.1),
                HoldTailDetection(p, .8, 100, 3, (640., 140.+480*p), 0.))
        for i, p in enumerate([.6, .7], 4):
            self.engine._record_active_hold_tail(track, self.frame(i, i*.1),
                HoldTailDetection(p, .8, 100, 3, (640., 140.+480*p), 0., 2))
        self.assertFalse(track.hold_terminal_confirmed)
        self.assertFalse(track.hold_release_locked)
        self.assertFalse(track.hold_tail_observations)

    def test_locked_release_still_receives_sustain_evidence(self):
        track = self.hold()
        track.hold_terminal_confirmed = track.hold_release_locked = True
        track.hold_release_time = 1.1
        # Its actual marker is still high in the ribbon, not at the target.
        for i, p in enumerate([.42, .47, .52], 1):
            tail = HoldTailDetection(p, .8, 100, 3, (640., 140.+480*p), 0., 2)
            with patch('agent.music.tracking.detect_hold_tails', return_value=[tail]):
                self.engine._update_active_hold_tails(self.frame(i, .7+i*.1))
        self.assertFalse(track.hold_release_locked)
        self.assertGreater(track.hold_release_time, 1.1)

    def test_static_track_cannot_capture_a_moving_head(self):
        static, real = NoteTrack(1, 3), NoteTrack(2, 3)
        for i, p in enumerate([.65, .65, .65, .65]):
            c = candidate_at(self.cal, 3, p)
            static.observations.append(TrackObservation(i, i*.05, c.center, p, c))
        for i, p in enumerate([.44, .48, .52, .56]):
            c = candidate_at(self.cal, 3, p)
            real.observations.append(TrackObservation(i, i*.05, c.center, p, c))
        real.speed = .8
        c = candidate_at(self.cal, 3, .68)
        entries = [(c, LaneProjection(3, .68, 0., (0., 1.)))]
        found = associate_taps([static, real], entries, self.frame(6, .25), self.engine.config,
                               lambda *_: True, self.engine.tap_trace)
        self.assertIs(found[0], real)

    def test_two_stationary_samples_then_dropout_cannot_steal_head(self):
        static = NoteTrack(1, 3)
        for i in range(2):
            c = candidate_at(self.cal, 3, .65)
            static.observations.append(TrackObservation(i, i*.033, c.center, .65, c))
        c = candidate_at(self.cal, 3, .70)
        found = associate_taps([static], [(c, LaneProjection(3, .70, 0., (0., 1.)))],
                               self.frame(8, .27), self.engine.config, lambda *_: True, self.engine.tap_trace)
        self.assertEqual(found, {})

    def test_concentric_hold_core_is_one_candidate_but_nearby_notes_survive(self):
        outer = candidate_at(self.cal, 3, .5, 60)
        core = candidate_at(self.cal, 3, .5, 20)
        next_note = candidate_at(self.cal, 3, .62, 60)
        entries = [(c, LaneProjection(3, p, 0., (0., 1.)))
                   for c, p in [(outer,.5), (core,.5), (next_note,.62)]]
        result = unique_head_candidates(entries, self.engine.tap_trace, self.frame(2,.2))
        self.assertEqual({c.box for c, _ in result}, {outer.box, next_note.box})

    def test_temporary_checkpoint_classification_can_recover_after_two_terminals(self):
        for i, kind in enumerate(['checkpoint', 'checkpoint', 'terminal', 'terminal']):
            tail = HoldTailDetection(.3+i*.03,.8,100,3,(640.,280.+i*12),0.,2,
                                     topology=kind)
            moving, streaks, flags = self.engine._moving_hold_tails([tail])
            self.engine.previous_hold_tails = [tail]
            self.engine.previous_hold_tail_streaks = streaks
            self.engine.previous_hold_tail_checkpoint_flags = flags
            self.engine.previous_hold_tail_terminal_streaks = self.engine._current_terminal_streaks
            if i == 2:
                self.assertTrue(flags[0])  # one ambiguous frame is not enough
        self.assertFalse(flags[0])

    def test_unlinked_staggered_holds_cannot_claim_opposite_ribbons(self):
        left = self.hold()
        left.lane = 1
        right = NoteTrack(2,5,gesture=NoteGesture.HOLD_START,state=TrackState.HOLDING,predicted_hit_time=.1)
        self.engine.tracks[2] = right
        # Ambiguous provisional lane labels must not override ribbon ownership.
        previous = [HoldTailDetection(.25,.8,100,3,(620.,250.),0.,1,'terminal',(1,)),
                    HoldTailDetection(.25,.8,100,5,(890.,250.),0.,1,'terminal',(5,))]
        self.engine.previous_hold_tails = previous
        self.engine.previous_hold_tail_streaks = [1,1]
        current = [replace(t, center=(t.center[0],260.), progress=.28) for t in previous]
        with patch('agent.music.tracking.detect_hold_tails', return_value=current):
            self.engine._update_active_hold_tails(self.frame(3,.6))
        self.assertEqual(left.hold_tail_observations[-1].center, (620.,260.))
        self.assertEqual(right.hold_tail_observations[-1].center, (890.,260.))

    def test_empty_ribbon_ownership_cannot_be_acquired_by_old_hold(self):
        track = self.hold()
        tail = HoldTailDetection(.4,.8,100,3,(640.,330.),0.,1,'terminal',())
        self.engine.previous_hold_tails = [replace(tail, center=(640.,320.))]
        self.engine.previous_hold_tail_streaks = [2]
        with patch('agent.music.tracking.detect_hold_tails', return_value=[tail]):
            self.engine._update_active_hold_tails(self.frame(5,3.))
        self.assertFalse(track.hold_tail_observations)

    def test_broad_background_light_is_not_an_upstream_ribbon(self):
        image = tail_frame(self.cal,3,.45)
        center = candidate_at(self.cal,3,.45,24).center
        y = int(center[1])
        image[y-70:y-15, 580:700] = 220  # broad stationary scenery
        topology, _, _, _ = marker_evidence(image,center,12.,3,self.cal)
        self.assertEqual(topology, 'terminal')

    def test_pending_release_is_retimed_after_revocation_even_inside_freeze(self):
        from agent.music.models import MusicActionEvent
        track = self.hold()
        event = MusicActionEvent('release-1',1,3,NoteGesture.HOLD_END,1.01,(640,620))
        track.hold_release_time = 1.01
        track.hold_release_locked = track.hold_terminal_confirmed = True
        self.engine._revoke_terminal(track,self.frame(10,1.))
        result = self.engine.refine_pending([event],1.)
        self.assertGreater(result[0].deadline,2.)
        self.assertEqual(result[0].event_id,event.event_id)

    def test_repeated_capture_does_not_delay_a_moving_heads_prediction(self):
        track = NoteTrack(1,3,gesture=NoteGesture.HOLD_START)
        for i,p in enumerate([.4,.5,.6,.7]):
            c=candidate_at(self.cal,3,p)
            track.observations.append(TrackObservation(i,i*.1,c.center,p,c))
        self.engine._update_motion(track)
        prediction=track.predicted_hit_time
        last=track.observations[-1]
        track.observations.append(replace(last,frame_sequence=4,timestamp=.333))
        self.engine._update_motion(track)
        self.assertAlmostEqual(track.predicted_hit_time,prediction)

    def test_blue_background_corners_do_not_make_a_yellow_head_rainbow(self):
        from agent.music.vision import detect_center_color_note
        from test_longtap_branch import center_color_frame
        self.assertTrue(detect_center_color_note(center_color_frame(240)))
        image=np.zeros((720,1280,3),np.uint8)
        image[180:300,580:700]=[230,150,80]
        yy,xx=np.ogrid[:720,:1280]
        disc=(xx-640)**2+(yy-240)**2 <= 34**2
        image[disc]=[10,220,245]
        self.assertEqual(detect_center_color_note(image),[])

    def test_shrinking_off_corridor_blob_cannot_corrupt_a_hold_head(self):
        from agent.music.head_identity import identity_continuity_allowed
        from agent.music.models import MusicCandidate
        track=NoteTrack(1,4,gesture=NoteGesture.HOLD_START,speed=.8)
        for i,p in enumerate([.44,.49,.54]):
            c=candidate_at(self.cal,4,p,60)
            track.observations.append(TrackObservation(i,i*.05,c.center,p,c))
        blob=MusicCandidate((780,368,48,44),1500,.8,(804,390))
        self.assertFalse(identity_continuity_allowed(track,blob,LaneProjection(4,.6,35.,(0.,1.)),self.frame(5,.2)))

    def test_late_hold_evidence_promotes_one_unsent_tap_event(self):
        from agent.music.models import MusicActionEvent
        from agent.music.vision import VisualMask
        from test_longtap_branch import frame_image
        for sent in (False,True):
            engine=MusicVisionEngine(self.cal,MusicConfig(lane_count=7,enable_holds=True))
            track=NoteTrack(1,3,gesture=NoteGesture.TAP,state=TrackState.TAP_PENDING,
                            action_executed=True,action_event_id='same-head',predicted_hit_time=1.)
            track.tap_input_started=.1 if sent else None
            engine.tracks[1]=track
            for i,p in enumerate([.3,.4,.5]):
                c=candidate_at(self.cal,3,p)
                track.observations.append(TrackObservation(i,i*.1,c.center,p,c))
            engine._update_motion(track)
            event=MusicActionEvent('same-head',1,3,NoteGesture.TAP,.875,(640,620))
            for seq,p in [(3,.6),(4,.7)]:
                c=candidate_at(self.cal,3,p)
                image=frame_image(self.cal,c,None)
                self.assertEqual(engine.update(self.frame(seq,seq*.1,image),[c],VisualMask.from_image(image,self.cal)),[])
            result=engine.refine_pending([event],.4)
            if not sent:
                self.assertEqual(len(result),1)
                self.assertEqual(result[0].event_id,'same-head')
                self.assertEqual(result[0].gesture,NoteGesture.HOLD_START)
                self.assertEqual(result[0].contact_policy,'persistent')
            else:
                self.assertEqual(track.gesture,NoteGesture.TAP)

    def test_white_ring_and_corners_do_not_dilute_orange_head(self):
        from agent.music.holds import hold_head_color_ratio
        from agent.music.models import MusicCandidate
        image=np.zeros((720,1280,3),np.uint8)
        yy,xx=np.ogrid[:720,:1280]
        d=(xx-640)**2+(yy-300)**2
        image[d<=40**2]=245
        image[d<=32**2]=[0,165,255]
        image[d<=10**2]=245
        c=MusicCandidate((600,260,80,80),3000,.5,(640.,300.))
        self.assertGreater(hold_head_color_ratio(image,c),.55)

    def test_missing_cap_frame_cannot_jump_to_the_next_checkpoint(self):
        from agent.music.models import HoldTailObservation
        track=self.hold()
        track.hold_terminal_confirmed=True
        for seq,p in enumerate([.4,.45,.5]):
            c=candidate_at(self.cal,3,p)
            track.hold_tail_observations.append(HoldTailObservation(seq,seq*.05,p,.8,100,3,c.center))
        wrong=HoldTailDetection(.7,.8,100,3,(640.,476.),0.,2,'checkpoint',(3,))
        with patch('agent.music.tracking.detect_hold_tails',return_value=[wrong]):
            self.engine._update_active_hold_tails(self.frame(3,.133))
        self.assertEqual(len(track.hold_tail_observations),3)
        true=HoldTailDetection(.57,.8,100,3,(640.,413.6),0.,1,'terminal',(3,))
        with patch('agent.music.tracking.detect_hold_tails',return_value=[true]):
            self.engine._update_active_hold_tails(self.frame(4,.17))
        self.assertEqual(track.hold_tail_observations[-1].center,true.center)

    def test_same_height_jitter_does_not_reverse_curved_cap_velocity(self):
        from agent.music.models import HoldTailObservation
        track=self.hold()
        track.hold_terminal_confirmed=True
        for seq,center in enumerate([(459.,280.),(441.,300.),(423.,319.),(424.5,319.)]):
            track.hold_tail_observations.append(
                HoldTailObservation(seq,seq/30.,.3+seq*.02,.8,100,2,center))
        for seq,center in [(4,(408.,338.5)),(5,(396.,350.5))]:
            true=HoldTailDetection(.4,.8,100,2,center,0.,1,'terminal',(3,))
            with patch('agent.music.tracking.detect_hold_tails',return_value=[true]):
                self.engine._update_active_hold_tails(self.frame(seq,seq/30.))
            self.assertEqual(track.hold_tail_observations[-1].center,true.center)


if __name__ == '__main__':
    unittest.main()
