"""Regressions for the staged optimization candidate, never a live controller."""
import os
os.environ.setdefault('MAES_AGENT_TEST_MODE', '1')
import unittest
import sys
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.music.build_identity import create_manifest, verify_manifest
from agent.music.executor import MusicActionExecutor, MusicTouchError, TapInputReceipt
from agent.music.models import MusicConfig, LaneInputState


class BuildIdentityTests(unittest.TestCase):
    def test_source_manifest_is_deterministic_and_detects_tampering(self):
        root = Path(__file__).resolve().parents[1]
        first = create_manifest(root)
        self.assertEqual(first, create_manifest(root))
        verify_manifest(root, first)
        first['files']['interface.json'] = 'bad'
        with self.assertRaises(ValueError):
            verify_manifest(root, first)

    def test_packaged_dependency_bytes_are_verified_separately_from_startup(self):
        from agent.music.build_identity import seal_package
        with tempfile.TemporaryDirectory(prefix='maes-manifest-test-') as temporary:
            root = Path(temporary)
            (root/'runtime').mkdir()
            dependency = root/'runtime/python.exe'
            dependency.write_bytes(b'test dependency')
            manifest = seal_package(root, create_manifest(root))
            verify_manifest(root, manifest)
            dependency.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_manifest(root, manifest)
            # Startup checks its code, not the whole runtime on every invocation.
            verify_manifest(root, manifest, dependencies=False)

    def test_candidate_data_is_isolated_and_explicit_override_wins(self):
        from agent.common import data_root
        with tempfile.TemporaryDirectory(prefix='maes-isolation-test-') as temporary:
            root = Path(temporary)/'package'
            root.mkdir()
            (root/'candidate-package.marker').touch()
            original = Path(temporary)/'original'
            with patch('agent.common.project_root', return_value=root), \
                 patch.dict(os.environ, {'MAES_DATA_DIR':'', 'LOCALAPPDATA':str(original)}):
                self.assertEqual(data_root(), root/'user-data')
                self.assertFalse(original.exists())
                with patch.dict(os.environ, {'MAES_DATA_DIR':str(Path(temporary)/'override')}):
                    self.assertEqual(data_root(), Path(temporary)/'override')

    def test_source_data_default_remains_compatible(self):
        from agent.common import data_root
        with tempfile.TemporaryDirectory(prefix='maes-isolation-test-') as temporary:
            root = Path(temporary)
            with patch('agent.common.project_root', return_value=root), \
                 patch.dict(os.environ, {'MAES_DATA_DIR':'', 'LOCALAPPDATA':str(root/'local-app')}):
                self.assertEqual(data_root(), root/'local-app/MAES')

    def test_archive_checks_resolved_temp_root_without_rejecting_valid_package(self):
        import zipfile
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
        from build_manifest import verify_archive
        with tempfile.TemporaryDirectory(prefix='maes-archive-test-') as temporary:
            root = Path(temporary)
            manifest = create_manifest(root)
            archive = root/'package.zip'
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('build-manifest.json', json.dumps(manifest))
            self.assertEqual(verify_archive(archive), manifest)

    def test_archive_rejects_traversal_before_extracting_anything(self):
        import zipfile
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
        from build_manifest import verify_archive
        with tempfile.TemporaryDirectory(prefix='maes-archive-test-') as temporary:
            archive = Path(temporary)/'bad.zip'
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('../outside.txt', 'not allowed')
            with self.assertRaisesRegex(ValueError, 'Unsafe archive path'):
                verify_archive(archive)


class InputLifecycleTests(unittest.TestCase):
    def executor(self):
        return MusicActionExecutor(SimpleNamespace(), 1280, 720,
                                   MusicConfig(lane_count=7, enable_holds=True),
                                   advanced=True, multi_touch=True)

    def test_pause_segment_allows_new_special_note_but_not_duplicate(self):
        for name in ('center-color-1', 'holdnote-1-1'):
            executor = self.executor()
            with patch.object(executor, '_run'):
                executor.begin_segment('run', 0)
                self.assertEqual(len(executor.tap(3, 640, 620, event_id=name)), 1)
                self.assertEqual(executor.tap(3, 640, 620, event_id=name), [])
                executor.release_all()
                executor.begin_segment('run', 1)
                receipt = executor.tap(3, 640, 620, event_id=name)[0]
                self.assertEqual((receipt.run_id, receipt.segment_id), ('run', 1))

    def test_segment_cannot_advance_with_unreleased_contact(self):
        executor = self.executor()
        executor.lanes[3] = LaneInputState(3, contact=0, hold_track_id=1)
        with self.assertRaises(MusicTouchError):
            executor.begin_segment('run', 1)

    def test_unconfirmed_contact_is_never_allocated_even_without_lane_owner(self):
        executor = self.executor()
        executor.release_unconfirmed.add(0)
        self.assertEqual(executor._allocate_temporary_contact(), 1)
        self.assertEqual(executor._allocate_contact(3), 2)

    def test_chord_capacity_counts_unconfirmed_contacts_and_falls_back_serially(self):
        from dataclasses import replace
        executor = self.executor()
        executor.config = replace(executor.config, max_contacts=2)
        executor.release_unconfirmed.add(0)
        with patch.object(executor, '_run') as run:
            executor.tap_many([(2,480,620),(4,800,620)], event_ids=['a','b'])
        self.assertEqual([c.args[0].value for c in run.call_args_list],
                         ['TouchDown','TouchUp','TouchDown','TouchUp'])
        self.assertIn('contact-capacity', executor.tap_fallbacks)
        self.assertEqual(executor.release_unconfirmed, {0})

    def test_failed_lane_release_keeps_contact_and_reports_when_fused(self):
        executor = self.executor()
        executor.healthy = False
        executor.lanes[3] = LaneInputState(3, contact=0, hold_track_id=1)
        with patch.object(executor, '_run', side_effect=MusicTouchError('native failed')) as run:
            with self.assertRaises(MusicTouchError):
                executor.release_all()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(executor.active_contacts, {3: 0})
        with patch.object(executor, '_run'):
            executor.release_all()
        self.assertEqual(executor.active_contacts, {})

    def test_cleanup_retries_up_only(self):
        executor = self.executor()
        executor.lanes[3] = LaneInputState(3, contact=0, hold_track_id=1)
        with patch.object(executor, '_run', side_effect=[MusicTouchError('first'), 0]) as run:
            executor.release_all()
        self.assertEqual([call.args[0].value for call in run.call_args_list], ['TouchUp', 'TouchUp'])
        self.assertFalse(executor.active_contacts)

    def test_old_receipt_cannot_ack_new_segment(self):
        from agent.music.runtime import MusicRuntime, RuntimeMetrics
        from test_tap_pipeline import tap_event
        runtime = MusicRuntime(SimpleNamespace(), MusicConfig(lane_count=7))
        runtime.tap_trace.segment_id = 1
        old = TapInputReceipt('1', 3, 0, 1., 1.001, 1.002, run_id=runtime.tap_trace.run_id, segment_id=0)
        runtime._acknowledge_taps([old], [tap_event(1, 3, 1.)], None, RuntimeMetrics())
        self.assertFalse(runtime.head_action_trace)

    def test_paused_strict_result_needs_two_confirmations(self):
        from agent.music.runtime import MusicRuntime
        from agent.music.models import MusicFrame
        from test_longtap_branch import ReleaseProbe
        runtime = MusicRuntime(SimpleNamespace(), MusicConfig(lane_count=7), sleeper=lambda _: None)
        frame = MusicFrame(0, 0., 0., 0., None)
        with patch('agent.music.runtime._capture_frame', return_value=(frame, 0.)), \
             patch('agent.music.runtime.terminal_state', return_value='result'):
            resumed, sequence, _, result = runtime._wait_until_resumed(ReleaseProbe(), 0, 7)
        self.assertIsNone(resumed)
        self.assertEqual(sequence, 2)
        self.assertEqual(result.status, 'succeeded')

    def test_wait_slices_observe_cancel(self):
        from agent.music.runtime import MusicRuntime, MusicCancelled
        sleeps = []
        runtime = MusicRuntime(SimpleNamespace(), MusicConfig(), sleeper=sleeps.append)
        with patch('agent.music.runtime.is_stopping', side_effect=[False, False, True]):
            with self.assertRaises(MusicCancelled):
                runtime._sleep_interruptibly(.1)
        self.assertEqual(sleeps, [.01, .01])


class MotionAndMetricTests(unittest.TestCase):
    def test_motion_fit_is_time_translation_invariant(self):
        from agent.music.tracking import regression_slope
        from agent.music.sustain import _recency_slope
        for origin in (0., 13600., 86400., 604800., 2592000., 1e7):
            values = [SimpleNamespace(timestamp=origin+i*.02, progress=.60+i*.01) for i in range(3)]
            for function in (regression_slope, _recency_slope):
                self.assertAlmostEqual(function(values), .5, places=6)

    def test_fit_rejects_nonfinite_and_degenerate_samples(self):
        from agent.music.tracking import regression_slope
        for timestamps in ((1., 1.), (float('nan'), 1.), (1., float('inf'))):
            values = [SimpleNamespace(timestamp=t, progress=.5+i*.01) for i, t in enumerate(timestamps)]
            self.assertEqual(regression_slope(values), 0.)

    def test_missing_capture_time_does_not_become_host_uptime_latency(self):
        from agent.music.runtime import MusicRuntime, RuntimeMetrics
        from test_tap_pipeline import tap_event
        runtime = MusicRuntime(SimpleNamespace(), MusicConfig(lane_count=7))
        metrics = RuntimeMetrics()
        runtime._acknowledge_taps([TapInputReceipt('1', 3, 0, 13600., 13600.001, 13600.002)],
                                  [tap_event(1, 3, 13599.)], None, metrics)
        self.assertEqual(len(metrics.perception_to_action), 0)

    def test_metrics_are_bounded_and_full_run_overflow_is_explicit(self):
        from agent.music.metrics import MetricSeries
        series = MetricSeries()
        for _ in range(1000):
            series.append(2.2)
        series.append(6000.)
        series.append(float('nan'))
        self.assertEqual(len(series), 60)
        self.assertEqual(series.summary()['count'], 1001)
        self.assertEqual(series.summary()['p95_upper_ms'], 3)
        self.assertEqual(series.summary()['overflow_count'], 1)
        self.assertEqual(series.summary()['invalid_count'], 1)


class CalibrationSafetyTests(unittest.TestCase):
    def test_confirmed_nine_lanes_never_loads_seven_lane_profile(self):
        from agent.music.runtime import _resolve_calibration
        context = SimpleNamespace(run_recognition=lambda *_: object())
        boxes = [SimpleNamespace(box=[100+i*100, 600, 10, 10]) for i in range(9)]
        with patch('agent.music.runtime.recognition_results', return_value=boxes), \
             patch('agent.music.runtime.load_calibration') as load:
            with self.assertRaises(ValueError):
                _resolve_calibration(context, MusicConfig(lane_count=7), None)
        load.assert_not_called()

    def test_geometry_rejects_invalid_or_duplicate_targets(self):
        from agent.music.calibration import _validate
        from test_longtap_branch import calibration
        for value in ([-1, 620], [float('nan'), 620], [160, 620]):
            cal = calibration()
            cal.points[1] = value
            with self.assertRaises(ValueError):
                _validate(cal)


class ReplayToolTests(unittest.TestCase):
    def test_candidate_preserves_flick_direction_and_color(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
        from tap_replay import decode_candidate
        from agent.music.models import NoteGesture
        note = decode_candidate({'box':[1,2,30,30], 'center':[16,17], 'pixel_count':900,
                                 'fill_ratio':1., 'variant':'flick', 'flick_direction':'FlickLeft',
                                 'flick_color':'red'})
        self.assertEqual(note.flick_direction, NoteGesture.FLICK_LEFT)
        self.assertEqual(note.flick_color, 'red')

    def test_ffmpeg_workspace_is_resolved_from_project_not_drive_root(self):
        from tools import workspace_paths
        with tempfile.TemporaryDirectory(prefix='maes-workspace-test-') as temporary:
            workspace = Path(temporary) / 'workspace'
            (workspace / 'test-materials').mkdir(parents=True)
            (workspace / '.work').mkdir()
            for relative in ('app', '.work/source-snapshot/app'):
                app_root = workspace / relative
                app_root.mkdir(parents=True)
                with self.subTest(location=relative), patch.object(
                    workspace_paths, 'APP_ROOT', app_root
                ):
                    self.assertEqual(workspace_paths.workspace_root(), workspace)

    def test_workspace_root_prefers_nearest_complete_workspace(self):
        from tools import workspace_paths
        with tempfile.TemporaryDirectory(prefix='maes-workspace-test-') as temporary:
            outer = Path(temporary) / 'outer'
            inner = outer / 'inner'
            for workspace in (outer, inner):
                (workspace / 'test-materials').mkdir(parents=True)
                (workspace / '.work').mkdir()
            app_root = inner / 'app'
            app_root.mkdir()
            with patch.object(workspace_paths, 'APP_ROOT', app_root):
                self.assertEqual(workspace_paths.workspace_root(), inner)

    def test_workspace_root_falls_back_without_both_markers(self):
        from tools import workspace_paths
        for markers in ((), ('test-materials',), ('.work',)):
            with self.subTest(markers=markers), tempfile.TemporaryDirectory(
                prefix='maes-workspace-test-'
            ) as temporary:
                fixture = Path(temporary)
                workspace = fixture / 'workspace'
                app_root = workspace / 'snapshots/app'
                app_root.mkdir(parents=True)
                for marker in markers:
                    (workspace / marker).mkdir()
                original_is_dir = Path.is_dir

                def fixture_is_dir(path):
                    # Ignore unrelated markers on the host's ancestor directories.
                    return path.is_relative_to(fixture) and original_is_dir(path)

                with patch.object(workspace_paths, 'APP_ROOT', app_root), patch.object(
                    Path, 'is_dir', autospec=True, side_effect=fixture_is_dir
                ):
                    self.assertEqual(workspace_paths.workspace_root(), app_root.parent)

    def test_ffmpeg_prefers_bundled_tool_to_path(self):
        from tools import workspace_paths
        with tempfile.TemporaryDirectory(prefix='maes-ffmpeg-test-') as temporary:
            workspace = Path(temporary)
            bundled = workspace / (
                '.work/ffmpeg-7.1.1-extract/'
                'ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe'
            )
            bundled.parent.mkdir(parents=True)
            bundled.touch()
            with patch.object(workspace_paths, 'workspace_root', return_value=workspace), \
                    patch.object(workspace_paths.shutil, 'which') as which:
                self.assertEqual(workspace_paths.ffmpeg_binary('ffmpeg.exe'), bundled)
                which.assert_not_called()

    def test_ffmpeg_falls_back_to_path_or_command_name(self):
        from tools import workspace_paths
        with tempfile.TemporaryDirectory(prefix='maes-ffmpeg-test-') as temporary:
            workspace = Path(temporary)
            on_path = str(workspace / 'system-bin/ffmpeg.exe')
            for located in (on_path, None):
                with self.subTest(located=located), patch.object(
                    workspace_paths, 'workspace_root', return_value=workspace
                ), patch.object(workspace_paths.shutil, 'which', return_value=located) as which:
                    self.assertEqual(
                        workspace_paths.ffmpeg_binary('ffmpeg.exe'),
                        Path(located or 'ffmpeg.exe'),
                    )
                    which.assert_called_once_with('ffmpeg.exe')

    def test_preflight_metrics_accept_extend(self):
        from agent.music.metrics import MetricSeries
        values = MetricSeries()
        values.extend([1.,2.,3.])
        self.assertEqual(values[:], [1.,2.,3.])


class PixelAndDispatchTests(unittest.TestCase):
    def test_component_output_order_pixels_and_boxes_match_reference(self):
        import numpy as np
        from legacy_components import connected_components as original
        from agent.music.vision import connected_components
        generator = np.random.default_rng(35)
        masks = [np.zeros((0,0),bool), np.zeros((9,11),bool), np.ones((9,11),bool),
                 np.eye(14,dtype=bool), np.indices((17,19)).sum(axis=0)%2 == 0]
        masks += [generator.random((31,53)) < p for p in (.01,.1,.3,.6,.9) for _ in range(8)]
        for mask in masks:
            for threshold in (1,3,10):
                self.assertEqual(connected_components(mask, threshold), original(mask, threshold))

    def test_post_capture_dispatch_never_waits_for_future_input(self):
        from agent.music.runtime import MusicRuntime, RuntimeMetrics
        from test_tap_pipeline import tap_event
        runtime = MusicRuntime(SimpleNamespace(), MusicConfig(lane_count=7), clock=lambda: 1.)
        pending = [tap_event(1,3,1.05)]
        with patch.object(runtime, 'sleeper') as sleep:
            runtime._execute_due(SimpleNamespace(), pending, 1., RuntimeMetrics(), wait=False)
        sleep.assert_not_called()
        self.assertEqual(len(pending), 1)


class ProductionLoopReplayTests(unittest.TestCase):
    def replay(self, cost_profile=None):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
        from tap_replay import candidate_frames
        from loop_replay import run_loop
        from agent.music.models import MusicCalibrationData
        root = Path(__file__).resolve().parents[1]
        fixture = root/'tests/fixtures/optimization/synthetic.jsonl'
        header = json.loads(fixture.read_text(encoding='utf-8').splitlines()[0])
        args = SimpleNamespace(candidates=fixture, video=None, branch_root=root,
                               cost_profile=cost_profile, action_ms=0.)
        return run_loop(candidate_frames(args), MusicCalibrationData(**header['calibration']),
                        MusicConfig(**header['config']), args)

    def test_real_loop_delays_and_failures_do_not_duplicate_or_leave_contacts(self):
        root = Path(__file__).resolve().parents[1]
        result = self.replay(root/'tests/fixtures/optimization/delays.json')
        self.assertGreater(len(result['heads']), 0)
        identifiers = [head['track'] for head in result['heads']]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        active = set()
        for action in result['actions']:
            if action['action'] == 'TouchDown':
                self.assertNotIn(action['contact'], active)
                active.add(action['contact'])
            elif action['action'] == 'TouchUp':
                active.discard(action['contact'])
        self.assertFalse(active)
        self.assertEqual(result['stop']['status'], 'cancelled')
        self.assertEqual(result['stop']['cleanup_failure'], '')

    def test_failed_input_is_not_resent_and_records_original_error(self):
        with tempfile.TemporaryDirectory(prefix='maes-cost-test-') as temporary:
            profile = Path(temporary)/'cost.json'
            profile.write_text(json.dumps({'action_failures':[0]}), encoding='utf-8')
            result = self.replay(profile)
        self.assertEqual(result['stop']['status'], 'failed')
        self.assertIn('success=false', result['stop']['reason'])
        self.assertEqual(sum(a['action'] == 'TouchDown' for a in result['actions']), 1)

    def test_initial_capture_failure_fails_explicitly(self):
        with tempfile.TemporaryDirectory(prefix='maes-cost-test-') as temporary:
            profile = Path(temporary)/'cost.json'
            profile.write_text(json.dumps({'by_capture':{'0':{'capture_failure':True}}}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Initial replay capture failed'):
                self.replay(profile)

    def test_first_resumed_capture_reaches_new_tracker_without_another_capture(self):
        with tempfile.TemporaryDirectory(prefix='maes-cost-test-') as temporary:
            profile = Path(temporary)/'cost.json'
            profile.write_text(json.dumps({'by_capture':{'1':{'ui':'pause'},'2':{'ui':'pause'}}}), encoding='utf-8')
            result = self.replay(profile)
        resets = [row for row in result['trace'] if row['kind'] == 'pause_reset']
        self.assertEqual(len(resets), 1)
        resumed = [row for row in result['observations'] if row['segment'] == 1]
        self.assertTrue(resumed)
        self.assertLess(resumed[0]['capture_finished'], resets[0]['time'])
        self.assertEqual(resumed[0]['sequence'], 3)
