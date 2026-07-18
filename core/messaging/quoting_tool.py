"""A ``send_message_to_user`` variant that quotes the triggering message.

This tool is mounted only for the proactive "wake AI to deliver the generation
result" turn (see :mod:`core.llm.result_handler`). It keeps the builtin tool's
name so the delivery prompt still matches, and it reuses all of the builtin
media / path handling. Only the final ``send_message`` is wrapped so a ``Reply``
component is prepended to the outgoing chain — without any global monkey patch.

The wake turn runs on a synthetic ``CronMessageEvent`` whose ``message_id`` is a
random uuid, so the real message id to quote must come from the original user
event and is passed in via ``reply_message_id``.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic.dataclasses import dataclass as pydantic_dataclass

from astrbot.core.agent.tool import ToolExecResult
from astrbot.core.tools.message_tools import SendMessageToUserTool

from .interaction import prepend_reply

QUOTE_PARAMETER_SCHEMA: dict[str, Any] = {
    "type": "boolean",
    "description": (
        "Whether to quote/reply to the user's original message that triggered "
        "this image generation. Defaults to true; set to false to send without "
        "quoting."
    ),
    "default": True,
}


@pydantic_dataclass
class QuotingSendMessageToUserTool(SendMessageToUserTool):
    """``send_message_to_user`` that quotes the user's original message."""

    reply_message_id: Any = ""

    @classmethod
    def build(
        cls,
        *,
        reply_message_id: Any,
        base_parameters: dict[str, Any] | None = None,
    ) -> QuotingSendMessageToUserTool:
        """Construct a quoting delivery tool.

        Args:
            reply_message_id: Real message id of the user's triggering message.
            base_parameters: The builtin tool's parameter schema to extend with an
                optional ``quote`` flag. When omitted the inherited schema is used.

        Returns:
            A configured :class:`QuotingSendMessageToUserTool`.
        """
        kwargs: dict[str, Any] = {"reply_message_id": reply_message_id or ""}
        if isinstance(base_parameters, dict):
            parameters = copy.deepcopy(base_parameters)
            properties = parameters.get("properties")
            if isinstance(properties, dict) and reply_message_id:
                properties.setdefault("quote", copy.deepcopy(QUOTE_PARAMETER_SCHEMA))
            kwargs["parameters"] = parameters
        return cls(**kwargs)

    async def call(self, context: Any, **kwargs: Any) -> ToolExecResult:
        """Deliver the message, prepending a reply quote when requested.

        The builtin ``call`` sends through ``context.context.context.send_message``.
        Only that shared ``Context`` (``AstrAgentContext.context``) is temporarily
        swapped for a forwarding proxy that prepends the reply; everything else is
        delegated to the builtin implementation and restored in ``finally``.
        """
        reply_id = self.reply_message_id
        # The LLM may opt out via the optional ``quote`` argument.
        if not kwargs.get("quote", True):
            reply_id = ""

        agent_ctx = getattr(context, "context", None)
        real_ctx = getattr(agent_ctx, "context", None) if agent_ctx is not None else None
        if not reply_id or agent_ctx is None or real_ctx is None:
            return await super().call(context, **kwargs)

        class _QuotingContextProxy:
            """Forward everything to the real Context, quoting on send_message."""

            def __getattr__(self, name: str) -> Any:
                return getattr(real_ctx, name)

            async def send_message(self, session: Any, message_chain: Any) -> Any:
                try:
                    prepend_reply(message_chain, reply_id)
                except Exception:
                    # Never let quoting failures block delivery.
                    pass
                return await real_ctx.send_message(session, message_chain)

        agent_ctx.context = _QuotingContextProxy()
        try:
            return await super().call(context, **kwargs)
        finally:
            agent_ctx.context = real_ctx


__all__ = ("QuotingSendMessageToUserTool",)
