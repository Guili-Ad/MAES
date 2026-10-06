"""Protect replay configuration provenance; no real controller is constructed."""
import io
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tap_replay import load_replay_config, replay_identity


class ReplayConfigTests(unittest.TestCase):
    def test_plain_json_config_is_not_replaced_by_defaults(self):
        with patch.object(Path, 'read_text', return_value='{"tap_action_advance_ms": 95, "coast_adaptive_speed": true}'):
            self.assertEqual(load_replay_config('live.json'),
                             {'tap_action_advance_ms': 95, 'coast_adaptive_speed': True})

    def test_wrapped_json_config_is_unwrapped(self):
        with patch.object(Path, 'read_text', return_value='{"run_id":"test", "config":{"split_stacked_notes":true}}'):
            self.assertEqual(load_replay_config('header.json'), {'split_stacked_notes': True})

    def test_jsonl_reads_first_nonempty_header_not_visual_records(self):
        with patch.object(Path, 'open', return_value=io.StringIO(
                '\n{"config":{"coast_adaptive_speed":true}}\nthis is not part of the config\n')):
            self.assertEqual(load_replay_config('live.jsonl'), {'coast_adaptive_speed': True})

    def test_jsonl_missing_settings_fails_explicitly(self):
        with patch.object(Path, 'open', return_value=io.StringIO('{"run_id":"unknown"}\n')):
            with self.assertRaisesRegex(ValueError, 'must contain a config object'):
                load_replay_config('invalid.jsonl')

    def test_empty_or_nonobject_settings_fail_explicitly(self):
        for text in ('[]', '{"config":null}'):
            with self.subTest(text=text), patch.object(Path, 'read_text', return_value=text):
                with self.assertRaisesRegex(ValueError, 'JSON object'):
                    load_replay_config('invalid.json')
        with patch.object(Path, 'open', return_value=io.StringIO('\n')):
            with self.assertRaisesRegex(ValueError, 'Empty replay config'):
                load_replay_config('empty.jsonl')

    def test_report_preserves_settings_and_distinguishes_geometry_from_timestamp(self):
        from agent.music.models import MusicConfig
        from test_longtap_branch import calibration
        config = MusicConfig(lane_count=7, coast_adaptive_speed=True, split_stacked_notes=True)
        cal = calibration()
        identity = replay_identity(config, cal, 'live.jsonl')
        self.assertEqual(identity['config'], asdict(config))
        self.assertEqual(identity['config_source'], 'live.jsonl')
        before = identity['effective_calibration_hash']
        cal.created_at = 'changed metadata, not geometry'
        changed = replay_identity(config, cal, 'live.jsonl')
        self.assertEqual(changed['effective_calibration_hash'], before)
        self.assertNotEqual(changed['calibration_hash'], identity['calibration_hash'])


if __name__ == '__main__':
    unittest.main()
