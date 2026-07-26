"""Reference image collection helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

from ..shared.logging import log_prefix, mask_sensitive, safe_log_text, safe_log_url
from ..shared.types import ImageCapability, ImageData
from .context_images import resolve_context_references

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

    from .image_processor import ImageProcessor


LOG = log_prefix("Reference")


def ensure_image_data(item: ImageData | tuple[bytes, str]) -> ImageData:
    """Normalize legacy (data, mime) tuples into ImageData."""
    if isinstance(item, ImageData):
        return item
    data, mime = item
    return ImageData(data=data, mime_type=mime)


def normalize_string_items(raw: Any) -> list[str]:
    """Normalize one or many string-like values from tool arguments."""
    if raw is None:
        return []
    if isinstance(raw, str):
        item = raw.strip()
        return [item] if item else []
    if isinstance(raw, dict):
        for key in ("url", "path", "file", "name"):
            if items := normalize_string_items(raw.get(key)):
                return items
        return []
    if isinstance(raw, Iterable):
        items: list[str] = []
        for value in raw:
            items.extend(normalize_string_items(value))
        return items
    item = str(raw).strip()
    return [item] if item else []


def resolve_avatar_user_id(event: Any, ref: str) -> str | None:
    """Resolve an avatar reference into a platform user id.

    Accepts ``self``, ``sender``, a bare user id, and the ``avatar:``/``qq:``/
    ``@`` prefixed forms the LLM may echo back from an injected handle list.
    """
    normalized = ref.strip().lower()
    for prefix in ("avatar:", "qq:", "@"):
        normalized = normalized.removeprefix(prefix).strip()
    if not normalized:
        return None

    if normalized == "self" and hasattr(event, "get_self_id"):
        # An empty id would build a request for a nonexistent avatar.
        return str(event.get_self_id() or "") or None
    if normalized == "sender" and hasattr(event, "get_sender_id"):
        return str(event.get_sender_id() or "") or None

    if normalized.isdigit():
        return normalized
    return None


def deduplicate_reference_images(
    images_data: list[ImageData],
    *,
    task_id: str | None = None,
    log_context: str = "Reference",
) -> list[ImageData]:
    """Remove duplicate reference images by content hash."""
    if len(images_data) < 2:
        return images_data

    unique_images: list[ImageData] = []
    seen_hashes: set[str] = set()
    duplicate_count = 0
    for image in images_data:
        image = ensure_image_data(image)
        digest = hashlib.sha256(image.data).hexdigest()
        if digest in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(digest)
        unique_images.append(image)

    if duplicate_count:
        task_log = log_prefix(log_context, task_id) if task_id else LOG
        logger.debug(f"{task_log} 已忽略 {duplicate_count} 张重复参考图")
    return unique_images


async def collect_reference_images_from_personas(
    image_processor: ImageProcessor,
    persona_images: list[tuple[str, str]],
    *,
    task_id: str | None = None,
    log_context: str = "Reference",
    workspace_dir: str | None = None,
) -> list[ImageData]:
    """Download all configured persona reference images."""
    images_data: list[ImageData] = []
    task_log = log_prefix(log_context, task_id) if task_id else LOG
    for persona_name, persona_image in persona_images:
        if persona_image_data := await image_processor.download_image(
            persona_image,
            workspace_dir=workspace_dir,
        ):
            images_data.append(persona_image_data)
        else:
            logger.warning(
                f"{task_log} 人设参考图获取失败: {safe_log_text(persona_name)}"
            )
    return images_data


async def download_reference_images(
    image_processor: ImageProcessor,
    references: Any,
    *,
    reference_label: str,
    task_id: str | None = None,
    log_context: str = "Reference",
    workspace_dir: str | None = None,
) -> list[ImageData]:
    """Download explicit reference images from URLs or local file paths."""
    images_data: list[ImageData] = []
    task_log = log_prefix(log_context, task_id) if task_id else LOG
    for reference in normalize_string_items(references):
        if image_data := await image_processor.download_image(
            reference,
            workspace_dir=workspace_dir,
        ):
            images_data.append(image_data)
        else:
            logger.warning(
                f"{task_log} {reference_label}参考图获取失败: {safe_log_url(reference)}"
            )
    return images_data


async def collect_command_reference_images(
    image_processor: ImageProcessor,
    event: AstrMessageEvent,
    persona_images: list[tuple[str, str]],
    *,
    task_id: str,
) -> list[ImageData]:
    """Collect command persona and message reference images."""
    workspace_dir = image_processor.workspace_dir_for_origin(
        getattr(event, "unified_msg_origin", None)
    )
    images_data = await collect_reference_images_from_personas(
        image_processor,
        persona_images,
        task_id=task_id,
        log_context="Task",
        workspace_dir=workspace_dir,
    )
    images_data.extend(await image_processor.fetch_images_from_event(event))
    return deduplicate_reference_images(
        images_data,
        task_id=task_id,
        log_context="Task",
    )


async def collect_tool_reference_images(
    image_processor: ImageProcessor,
    event: Any,
    *,
    capabilities: ImageCapability,
    reference_images: Any = None,
    avatar_references: Any = None,
    persona_images: list[tuple[str, str]] | None = None,
    task_id: str | None = None,
    auto_context_reference: bool = True,
) -> list[ImageData]:
    """Collect LLM tool persona, handle, URL/path, and avatar reference images.

    ``reference_images`` may contain context handles (``message:1``, ``reply:1``,
    ``avatar:10001``) resolved against the triggering message. This is the only
    way a chat model without image input can request image-to-image generation,
    since it never receives the message images themselves.

    When ``auto_context_reference`` is set and the model asked for no reference
    at all, images found in the message are used as a fallback.
    """
    task_log = log_prefix("LLMTool", task_id) if task_id else LOG
    if not (capabilities & ImageCapability.IMAGE_TO_IMAGE):
        if reference_images or avatar_references or persona_images:
            logger.warning(f"{task_log} 当前适配器不支持参考图，已忽略工具参考图参数")
        return []

    raw_references = normalize_string_items(reference_images)
    raw_avatar_references = normalize_string_items(avatar_references)
    has_explicit_request = bool(
        raw_references or raw_avatar_references or persona_images
    )

    images_data: list[ImageData] = []
    avatar_user_ids: set[str] = set()
    workspace_dir = image_processor.workspace_dir_for_origin(
        getattr(event, "unified_msg_origin", None)
    )

    if persona_images:
        images_data.extend(
            await collect_reference_images_from_personas(
                image_processor,
                persona_images,
                task_id=task_id,
                log_context="LLMTool",
                workspace_dir=workspace_dir,
            )
        )

    index = image_processor.scan_event_image_sources(event)
    resolved = resolve_context_references(index, raw_references)
    if resolved.unresolved:
        logger.warning(
            f"{task_log} 无法解析的图片句柄: "
            f"{safe_log_text('、'.join(resolved.unresolved))}"
        )

    for references, label in (
        (resolved.urls, "句柄"),
        (resolved.plain_refs, "显式"),
    ):
        images_data.extend(
            await download_reference_images(
                image_processor,
                references,
                reference_label=label,
                task_id=task_id,
                log_context="LLMTool",
                workspace_dir=workspace_dir,
            )
        )

    for ref in (*raw_avatar_references, *resolved.avatar_refs):
        user_id = resolve_avatar_user_id(event, ref)
        if not user_id or user_id in avatar_user_ids:
            continue
        avatar_user_ids.add(user_id)
        if avatar_data := await image_processor.get_avatar(user_id):
            images_data.append(ImageData(data=avatar_data, mime_type="image/jpeg"))
            logger.debug(
                f"{task_log} 已添加 {mask_sensitive(user_id)} 的头像作为参考图"
            )

    if auto_context_reference and not has_explicit_request and not index.is_empty:
        # A text-only model may simply not mention the picture it cannot see.
        if auto_images := await image_processor.fetch_indexed_images(
            event,
            index,
            avatar_user_ids=avatar_user_ids,
        ):
            images_data.extend(auto_images)
            logger.debug(
                f"{task_log} 工具未指定参考图，已自动引用消息中的 "
                f"{len(auto_images)} 张图片"
            )

    return deduplicate_reference_images(
        images_data,
        task_id=task_id,
        log_context="LLMTool",
    )
