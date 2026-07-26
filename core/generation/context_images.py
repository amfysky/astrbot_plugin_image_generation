"""Text handles for images reachable from the triggering message.

A chat model without image input never receives the bytes of the picture a user
just sent, so it has no way to say "redraw *that* image". Injecting the image
into its context only works for multimodal providers, and describing the image
in words destroys exactly the pixel-level identity that image-to-image needs.

This module implements the third option: index every image reachable from the
message and expose each one as a short text handle (``message:1``, ``reply:1``,
``avatar:10001``). The handle list is injected into the prompt as plain text,
the model echoes a handle back through ``generate_image``, and the plugin
resolves it into real bytes on the way to the image model. The bytes never
enter the chat model's context, so text-only providers work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

HANDLE_MESSAGE = "message"
"""Handle kind for images attached to the current message."""

HANDLE_REPLY = "reply"
"""Handle kind for images inside the quoted message."""

HANDLE_AVATAR = "avatar"
"""Handle kind for user avatars."""

CONTEXT_HANDLE_KINDS = frozenset({HANDLE_MESSAGE, HANDLE_REPLY, HANDLE_AVATAR})
"""All recognized handle kinds."""

ALL_SELECTORS = frozenset({"", "all", "*"})
"""Selectors that mean "every image of this kind"."""

HANDLE_HINT_TAG = "image_reference_handles"
"""XML-ish tag wrapping the injected handle list."""


@dataclass
class ImageSource:
    """One image reachable from a message, recorded in message-chain order."""

    kind: str
    """Handle kind: ``message``, ``reply``, or ``avatar``."""

    value: str
    """Image URL for ``message``/``reply``, or user id for ``avatar``."""

    label: str = ""
    """Display name, currently the mentioned user's nickname."""


@dataclass
class EventImageIndex:
    """Every image source found in one message event.

    Sources are kept in message-chain order so reference images reach the image
    model in the same order the user arranged them.
    """

    sources: list[ImageSource] = field(default_factory=list)

    def of_kind(self, kind: str) -> list[ImageSource]:
        """Return sources of one handle kind, preserving order."""
        return [source for source in self.sources if source.kind == kind]

    @property
    def message_images(self) -> list[str]:
        """Return URLs of images attached to the current message."""
        return [source.value for source in self.of_kind(HANDLE_MESSAGE)]

    @property
    def reply_images(self) -> list[str]:
        """Return URLs of images inside the quoted message."""
        return [source.value for source in self.of_kind(HANDLE_REPLY)]

    @property
    def mentioned_users(self) -> list[ImageSource]:
        """Return mentioned users whose avatar can be used as a reference."""
        return self.of_kind(HANDLE_AVATAR)

    @property
    def is_empty(self) -> bool:
        """Return whether the message exposes no usable image source."""
        return not self.sources


@dataclass
class ResolvedReferences:
    """Tool ``reference_images`` values split by how they must be fetched."""

    urls: list[str] = field(default_factory=list)
    """Image URLs resolved from ``message``/``reply`` handles."""

    avatar_refs: list[str] = field(default_factory=list)
    """User references resolved from ``avatar`` handles."""

    plain_refs: list[str] = field(default_factory=list)
    """Values that are not handles: network URLs or allowed local paths."""

    unresolved: list[str] = field(default_factory=list)
    """Well-formed handles that point at nothing in the current message."""


def split_handle(reference: str) -> tuple[str, str] | None:
    """Split a reference into ``(kind, selector)`` when it is a context handle.

    Returns None for ordinary values. Network URLs (``https:``) and Windows
    paths (``C:``) also contain a colon, so the prefix is only treated as a
    handle when it names a known kind.
    """
    normalized = reference.strip()
    if not normalized:
        return None

    kind, _, selector = normalized.partition(":")
    kind = kind.strip().lower()
    if kind not in CONTEXT_HANDLE_KINDS:
        return None
    return kind, selector.strip()


def _select_images(images: list[str], selector: str) -> list[str] | None:
    """Resolve a ``message``/``reply`` selector into image URLs.

    Returns None when the selector is a valid 1-based index that the current
    message does not have, so the caller can report it as unresolved.
    """
    if selector.lower() in ALL_SELECTORS:
        return list(images)
    if not selector.isdigit():
        return None
    position = int(selector)
    if 1 <= position <= len(images):
        return [images[position - 1]]
    return None


def resolve_context_references(
    index: EventImageIndex,
    references: list[str],
) -> ResolvedReferences:
    """Split tool references into context handles and ordinary references."""
    resolved = ResolvedReferences()
    for reference in references:
        handle = split_handle(reference)
        if handle is None:
            resolved.plain_refs.append(reference)
            continue

        kind, selector = handle
        if kind == HANDLE_AVATAR:
            if selector:
                resolved.avatar_refs.append(selector)
            else:
                resolved.unresolved.append(reference)
            continue

        images = index.message_images if kind == HANDLE_MESSAGE else index.reply_images
        selected = _select_images(images, selector)
        if selected:
            resolved.urls.extend(selected)
        else:
            resolved.unresolved.append(reference)
    return resolved


def format_context_handle_hint(index: EventImageIndex) -> str:
    """Render the handle list injected into the chat model's system prompt.

    Returns an empty string when the message carries no image source, so plain
    text conversations are left untouched.
    """
    if index.is_empty:
        return ""

    # Only handles are exposed; the underlying URLs stay server-side.
    lines: list[str] = []
    for position in range(1, len(index.message_images) + 1):
        lines.append(
            f"- {HANDLE_MESSAGE}:{position} = 用户本条消息中的第 {position} 张图片"
        )
    for position in range(1, len(index.reply_images) + 1):
        lines.append(
            f"- {HANDLE_REPLY}:{position} = 被引用消息中的第 {position} 张图片"
        )
    for source in index.mentioned_users:
        who = f"@{source.label}" if source.label else f"用户 {source.value}"
        lines.append(f"- {HANDLE_AVATAR}:{source.value} = {who} 的头像")

    if not lines:
        return ""

    return (
        f"<{HANDLE_HINT_TAG}>\n"
        "以下是本条消息中可用于图生图的图片句柄。这些图片不会出现在你的上下文里，"
        "你无需“看到”它们，只要把句柄原样填进生图工具的参数即可：\n"
        + "\n".join(lines)
        + "\n用法：需要以某张已有图片为基础作画、改图、换风格时，"
        f"把上面的句柄填入 generate_image 的 reference_images；"
        f"头像句柄也可以填入 avatar_references。"
        f"另外始终可用 {HANDLE_AVATAR}:sender（发送者头像）和 {HANDLE_AVATAR}:self（你自己的头像）。\n"
        f"</{HANDLE_HINT_TAG}>"
    )


__all__ = (
    "ALL_SELECTORS",
    "CONTEXT_HANDLE_KINDS",
    "EventImageIndex",
    "HANDLE_AVATAR",
    "HANDLE_MESSAGE",
    "HANDLE_REPLY",
    "ImageSource",
    "ResolvedReferences",
    "format_context_handle_hint",
    "resolve_context_references",
    "split_handle",
)
