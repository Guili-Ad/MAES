from __future__ import annotations

from agent.common import LOGGER, parse_custom_param
from agent.compat import AgentServer, Context, CustomAction


@AgentServer.custom_action("ReportTaskFailure")
class ReportTaskFailure(CustomAction):
    """Turn a recognized terminal error state into a failed Maa task."""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        del context
        try:
            params = parse_custom_param(getattr(argv, "custom_action_param", None))
            message = str(params.get("message", "Pipeline reached a failure state"))
        except (TypeError, ValueError, AttributeError):
            LOGGER.exception("Invalid ReportTaskFailure parameters")
            return False
        LOGGER.error("%s", message)
        return False
