"""Image generation execution package."""

from .context_images import (
    EventImageIndex,
    ImageSource,
    format_context_handle_hint,
    resolve_context_references,
)
from .image_processor import ImageProcessor
from .image_utils import (
    convert_image_format,
    convert_images_batch,
    detect_mime_type,
    validate_aspect_ratio,
    validate_resolution,
)
from .reference_collector import (
    collect_command_reference_images,
    collect_reference_images_from_personas,
    collect_tool_reference_images,
    deduplicate_reference_images,
    download_reference_images,
    ensure_image_data,
    normalize_string_items,
)


def __getattr__(name: str):
    """Lazily expose heavy generation components to avoid import cycles."""
    if name == "GenerationExecutor":
        from .executor import GenerationExecutor

        return GenerationExecutor
    raise AttributeError(name)


__all__ = (
    "EventImageIndex",
    "GenerationExecutor",
    "ImageProcessor",
    "ImageSource",
    "collect_command_reference_images",
    "collect_reference_images_from_personas",
    "collect_tool_reference_images",
    "convert_image_format",
    "convert_images_batch",
    "deduplicate_reference_images",
    "detect_mime_type",
    "download_reference_images",
    "ensure_image_data",
    "format_context_handle_hint",
    "normalize_string_items",
    "resolve_context_references",
    "validate_aspect_ratio",
    "validate_resolution",
)
