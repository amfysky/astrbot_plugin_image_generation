"""Platform message interaction helpers (reply quoting and emoji reactions)."""

from .interaction import (
    EmojiReaction,
    get_reply_message_id,
    prepend_reply,
    supports_emoji_reaction,
)

__all__ = (
    "EmojiReaction",
    "get_reply_message_id",
    "prepend_reply",
    "supports_emoji_reaction",
)
