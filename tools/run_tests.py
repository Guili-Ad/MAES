from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MAES_AGENT_TEST_MODE", "1")
sys.path.insert(0, str(APP_ROOT))


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(APP_ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
