from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.common import LOGGER, configure_logging, data_root


def main() -> int:
    configure_logging()
    if len(sys.argv) < 2:
        LOGGER.error("Missing MaaFramework Agent socket identifier")
        return 2

    from agent.compat import AgentServer, MAA_AVAILABLE, Tasker
    import agent.actions  # noqa: F401 - imports register all custom actions

    if not MAA_AVAILABLE:
        LOGGER.error("MaaFramework Python bindings are missing from the bundled runtime")
        return 2

    log_dir = data_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    Tasker.set_log_dir(str(log_dir))

    socket_id = sys.argv[-1]
    AgentServer.start_up(socket_id)
    try:
        AgentServer.join()
    finally:
        AgentServer.shut_down()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
