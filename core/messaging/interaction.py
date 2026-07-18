"""Reply quoting and emoji reaction helpers for outgoing generation messages.

These helpers are platform aware. Reply quoting relies on the AstrBot ``Reply``
message component and works on any platform that supports it. Emoji reactions use
the aiocqhttp (OneBot / NapCat / Lagrange) ``set_msg_emoji_like`` action and are
silently skipped on platforms that do not provide it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import astrbot.api.message_components as Comp
from astrbot.api import logger

from ..shared.logging import log_prefix, safe_log_error_body

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent, MessageChain

LOG = log_prefix("Interaction")

AIOCQHTTP_PLATFORM_NAME = "aiocqhttp"


def get_reply_message_id(event: AstrMessageEvent | None) -> Any | None:
    """Return the triggering message id used to quote a reply.

    Args:
        event: Source message event, or ``None`` when unavailable.

    Returns:
        The raw platform message id, or ``None`` when it cannot be resolved.
    """
    message_obj = getattr(event, "message_obj", None)
    message_id = getattr(message_obj, "message_id", None)
    if message_id is None or message_id == "":
        return None
    return message_id


def prepend_reply(chain: MessageChain, message_id: Any | None) -> None:
    """Prepend a ``Reply`` component so an outgoing chain quotes a message.

    The chain is left unchanged when there is no message id, when the chain is
    not a builder with a component list, or when it already contains a reply.

    Args:
        chain: Outgoing message chain to mutate in place.
        message_id: Raw platform message id to quote.
    """
    if message_id is None or message_id == "":
        return
    components = getattr(chain, "chain", None)
    if not isinstance(components, list):
        return
    if any(isinstance(component, Comp.Reply) for component in components):
        return
    components.insert(0, Comp.Reply(id=message_id))


def supports_emoji_reaction(event: AstrMessageEvent | None) -> bool:
    """Return whether the event platform supports emoji reactions.

    Args:
        event: Source message event, or ``None`` when unavailable.

    Returns:
        Whether an aiocqhttp bot client exposing ``set_msg_emoji_like`` is
        available for the event.
    """
    if event is None or not getattr(event, "platform_meta", None):
        return False
    try:
        if event.get_platform_name() != AIOCQHTTP_PLATFORM_NAME:
            return False
    except Exception:
        return False
    bot = getattr(event, "bot", None)
    return bool(bot is not None and hasattr(bot, "set_msg_emoji_like"))


def _normalize_emoji_ids(emoji_ids: Iterable[Any] | None) -> tuple[Any, ...]:
    """Deduplicate emoji ids while preserving configuration order."""
    if not emoji_ids:
        return ()
    return tuple(dict.fromkeys(emoji_ids))


@dataclass(frozen=True)
class EmojiReaction:
    """A pending emoji reaction bound to one triggering message.

    The bot client and message id are captured up front so the reaction can be
    removed later without holding onto the full event.
    """

    bot: Any
    message_id: Any
    emoji_ids: tuple[Any, ...]

    @classmethod
    def from_event(
        cls,
        event: AstrMessageEvent | None,
        emoji_ids: Iterable[Any] | None,
    ) -> EmojiReaction | None:
        """Build an emoji reaction from a source event when supported.

        Args:
            event: Source message event to react to.
            emoji_ids: Emoji ids to apply while the task is running.

        Returns:
            An :class:`EmojiReaction`, or ``None`` when reactions are not
            supported or no emoji ids are configured.
        """
        normalized_ids = _normalize_emoji_ids(emoji_ids)
        if not normalized_ids or not supports_emoji_reaction(event):
            return None
        message_id = get_reply_message_id(event)
        if message_id is None:
            return None
        return cls(
            bot=getattr(event, "bot", None),
            message_id=message_id,
            emoji_ids=normalized_ids,
        )

    async def _apply(self, set_reaction: bool) -> None:
        """Add or remove all configured emoji reactions, ignoring failures."""
        bot = self.bot
        if bot is None or not hasattr(bot, "set_msg_emoji_like"):
            return
        for emoji_id in self.emoji_ids:
            try:
                await bot.set_msg_emoji_like(
                    message_id=self.message_id,
                    emoji_id=emoji_id,
                    set=set_reaction,
                )
            except Exception as exc:
                logger.debug(
                    f"{LOG} {'贴' if set_reaction else '移除'}表情反应失败 "
                    f"(emoji_id={emoji_id}): {safe_log_error_body(exc, 160)}"
                )

    async def add(self) -> None:
        """Add the progress emoji reactions to the triggering message."""
        await self._apply(True)

    async def remove(self) -> None:
        """Remove the progress emoji reactions from the triggering message."""
        await self._apply(False)


__all__ = (
    "EmojiReaction",
    "get_reply_message_id",
    "prepend_reply",
    "supports_emoji_reaction",
)
