"""Image detection, resizing, base64 encoding for vision support."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

MAX_IMAGE_DIMENSION = 1024

_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def is_image(path: str) -> bool:
    """Check if a file path has an image extension."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def guess_mime(path: str) -> str:
    """Guess MIME type from file extension."""
    return _MIME_MAP.get(Path(path).suffix.lower(), "application/octet-stream")


def resize_image_bytes(data: bytes, max_dim: int = MAX_IMAGE_DIMENSION) -> bytes:
    """Resize image bytes if either dimension exceeds max_dim. Returns JPEG bytes.

    Requires Pillow. Falls back to original bytes if Pillow is unavailable or
    the image is already small enough.
    """
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        w, h = img.size
        if w <= max_dim and h <= max_dim:
            return data  # already small enough

        # Calculate new dimensions preserving aspect ratio
        if w > h:
            new_w = max_dim
            new_h = int(h * max_dim / w)
        else:
            new_h = max_dim
            new_w = int(w * max_dim / h)

        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Convert to RGB if needed (e.g. RGBA PNGs)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except ImportError:
        logger.warning("Pillow not available, skipping image resize")
        return data
    except Exception as e:
        logger.warning(f"Image resize failed, using original: {e}")
        return data


def encode_image_to_data_uri(
    path: str,
    *,
    resize: bool = True,
    max_dim: int = MAX_IMAGE_DIMENSION,
) -> str:
    """Read an image file and return a base64 data URI."""
    data = Path(path).read_bytes()
    if resize:
        resized = resize_image_bytes(data, max_dim)
        # If resized, MIME is JPEG; if not resized, use original MIME
        if resized is not data:
            mime = "image/jpeg"
            data = resized
        else:
            mime = guess_mime(path)
    else:
        mime = guess_mime(path)
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def encode_bytes_to_data_uri(
    data: bytes,
    mime: str = "image/jpeg",
    *,
    resize: bool = True,
    max_dim: int = MAX_IMAGE_DIMENSION,
) -> str:
    """Encode raw image bytes to a base64 data URI."""
    if resize:
        resized = resize_image_bytes(data, max_dim)
        if resized is not data:
            mime = "image/jpeg"
            data = resized
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_image_content_blocks(
    path: str,
    *,
    annotation: str | None = None,
    resize: bool = True,
) -> list[dict]:
    """Build OpenAI-format content blocks for an image file.

    Returns a list with a text annotation block and an image_url block.
    """
    data_uri = encode_image_to_data_uri(path, resize=resize)
    blocks: list[dict] = []
    if annotation:
        blocks.append({"type": "text", "text": annotation})
    blocks.append({
        "type": "image_url",
        "image_url": {"url": data_uri},
    })
    return blocks


def extract_text_from_content(content: str | list[dict] | None) -> str:
    """Extract only text from content (string or content blocks).

    Used for FTS indexing and display fallbacks.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if block.get("type") == "text" and block.get("text"):
            parts.append(block["text"])
    return "\n".join(parts)


def content_has_images(content: str | list[dict] | None) -> bool:
    """Check if content contains any image blocks."""
    if not isinstance(content, list):
        return False
    return any(b.get("type") == "image_url" for b in content)


def strip_images_from_content(content: str | list[dict] | None) -> str | list[dict] | None:
    """Replace image blocks with [image] text placeholders.

    Used before sending messages to the compaction LLM.
    """
    if not isinstance(content, list):
        return content
    result = []
    for block in content:
        if block.get("type") == "image_url":
            result.append({"type": "text", "text": "[image]"})
        else:
            result.append(block)
    return result
