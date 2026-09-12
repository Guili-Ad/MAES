from __future__ import annotations

import json
import unittest
from pathlib import Path


BRANCH_ROOT = Path(__file__).resolve().parents[1]


class StartupControllerGuardTests(unittest.TestCase):
    def test_separate_screenshot_tasker_is_enabled(self) -> None:
        instance_config = json.loads(
            (BRANCH_ROOT / "config" / "instances" / "default.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertIs(instance_config.get("UseSeparateScreenshotTasker"), True)

    def test_live_view_uses_an_isolated_screenshot_tasker(self) -> None:
        global_config = json.loads(
            (BRANCH_ROOT / "config" / "config.json").read_text(encoding="utf-8")
        )
        instance_config = json.loads(
            (BRANCH_ROOT / "config" / "instances" / "default.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertIs(global_config.get("UI.LiveView.EnableLiveView"), True)
        self.assertIs(instance_config.get("UI.LiveView.EnableLiveView"), True)
        self.assertIs(instance_config.get("UseSeparateScreenshotTasker"), True)

    def test_agent_calls_have_a_bounded_timeout(self) -> None:
        interface = json.loads(
            (BRANCH_ROOT / "interface.json").read_text(encoding="utf-8")
        )

        self.assertEqual(interface["agent"].get("timeout"), 8)


if __name__ == "__main__":
    unittest.main()
