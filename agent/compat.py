from __future__ import annotations

import os
from typing import Any, Callable


MAA_AVAILABLE = True

try:
    if os.environ.get("MAES_AGENT_TEST_MODE") == "1":
        raise ModuleNotFoundError("Maa agent registration disabled for unit tests")
    from maa.agent.agent_server import AgentServer
    from maa.context import Context
    from maa.custom_action import CustomAction
    from maa.custom_recognition import CustomRecognition
    from maa.tasker import Tasker
except ModuleNotFoundError:
    MAA_AVAILABLE = False

    class _RegistryStub:
        @staticmethod
        def custom_action(_name: str) -> Callable[[type], type]:
            return lambda cls: cls

        @staticmethod
        def custom_recognition(_name: str) -> Callable[[type], type]:
            return lambda cls: cls

        @staticmethod
        def start_up(_socket_id: str) -> None:
            raise RuntimeError("The MaaFramework Python package is not installed")

        @staticmethod
        def join() -> None:
            return None

        @staticmethod
        def shut_down() -> None:
            return None

    class _CustomActionStub:
        RunArg = Any

    class _CustomRecognitionStub:
        AnalyzeArg = Any

        class AnalyzeResult:
            def __init__(self, box: Any, detail: dict[str, Any]) -> None:
                self.box = box
                self.detail = detail

    class _TaskerStub:
        @staticmethod
        def set_log_dir(_path: str) -> None:
            return None

    AgentServer = _RegistryStub()  # type: ignore[assignment]
    Context = Any  # type: ignore[assignment,misc]
    CustomAction = _CustomActionStub  # type: ignore[assignment,misc]
    CustomRecognition = _CustomRecognitionStub  # type: ignore[assignment,misc]
    Tasker = _TaskerStub  # type: ignore[assignment,misc]


__all__ = [
    "AgentServer",
    "Context",
    "CustomAction",
    "CustomRecognition",
    "MAA_AVAILABLE",
    "Tasker",
]
