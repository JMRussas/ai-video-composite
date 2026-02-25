#  ai-video-composite - Image Compositing Module
#
#  Single-image and batch compositing of characters onto backgrounds.
#  Handles green-screen detection and automatic background removal.
#
#  Depends on: removal.py, numpy, Pillow
#  Used by:    __main__.py, __init__.py

import logging
import tempfile
from pathlib import Path

from PIL import Image

from .removal import remove_background

log = logging.getLogger(__name__)

# Minimum image dimension for corner sampling in green-screen detection
_MIN_SAMPLE_SIZE = 11


# ---------------------------------------------------------------------------
# Image Compositing
# ---------------------------------------------------------------------------

def composite_character(
    character_path: Path,
    background_path: Path,
    output_path: Path,
    position: str = "center",
    scale: float = 0.75,
    offset_x: int = 0,
    offset_y: int = 0,
) -> Path:
    """Composite a green-screen character onto a background image.

    Args:
        character_path: Path to character image (green screen or transparent)
        background_path: Path to background/scene image
        output_path: Where to save the composite
        position: "left", "center", "right" — horizontal anchor
        scale: Character height as fraction of background height (0.0-1.0)
        offset_x: Additional horizontal pixel offset (positive = right)
        offset_y: Additional vertical pixel offset (positive = up, from bottom)

    Returns:
        Path to saved composite image.

    Raises:
        OSError: If character or background image cannot be opened.
    """
    try:
        char_img = Image.open(character_path)
    except (OSError, ValueError) as e:
        raise OSError(f"Cannot open character image '{character_path}': {e}") from e

    try:
        bg_img = Image.open(background_path).convert("RGBA")
    except (OSError, ValueError) as e:
        raise OSError(f"Cannot open background image '{background_path}': {e}") from e

    # Remove green background if not already transparent
    if char_img.mode != "RGBA" or _is_green_background(char_img):
        log.info("Removing green background...")
        char_img = remove_background(char_img)
    else:
        char_img = char_img.convert("RGBA")

    # Scale character relative to background height
    bg_w, bg_h = bg_img.size
    char_w, char_h = char_img.size

    target_h = int(bg_h * scale)
    aspect = char_w / char_h
    target_w = int(target_h * aspect)
    char_img = char_img.resize((target_w, target_h), Image.LANCZOS)

    # Position character
    if position == "left":
        x = int(bg_w * 0.15) - target_w // 2
    elif position == "right":
        x = int(bg_w * 0.85) - target_w // 2
    else:  # center
        x = (bg_w - target_w) // 2

    x += offset_x

    # Anchor to bottom of frame (characters stand on ground)
    y = bg_h - target_h - offset_y

    # Composite
    composite = bg_img.copy()
    composite.paste(char_img, (x, y), char_img)

    # Save as RGB (no need for alpha in final output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite.convert("RGB").save(output_path, "PNG")

    log.info("Composite: %s (%dx%d)", output_path.name, bg_w, bg_h)
    return output_path


def _is_green_background(img: Image.Image) -> bool:
    """Quick check if image has a predominantly green background (chroma key)."""
    w, h = img.size
    if w < _MIN_SAMPLE_SIZE or h < _MIN_SAMPLE_SIZE:
        return False

    img_rgb = img.convert("RGB")
    corners = [
        img_rgb.getpixel((5, 5)),
        img_rgb.getpixel((w - 5, 5)),
        img_rgb.getpixel((5, h - 5)),
        img_rgb.getpixel((w - 5, h - 5)),
    ]
    green_count = sum(1 for r, g, b in corners if g > 100 and g > r * 1.3 and g > b * 1.3)
    return green_count >= 3


# ---------------------------------------------------------------------------
# Batch Compositing
# ---------------------------------------------------------------------------

def batch_composite(
    character_path: Path,
    backgrounds_dir: Path,
    output_dir: Path,
    **kwargs,
) -> list[Path]:
    """Composite one character onto every background in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # Pre-remove background once (reuse transparent version via tempfile)
    tmp_path = None
    try:
        char_img = Image.open(character_path)
    except (OSError, ValueError) as e:
        raise OSError(f"Cannot open character image '{character_path}': {e}") from e

    if char_img.mode != "RGBA" or _is_green_background(char_img):
        log.info("Removing background from character (one-time)...")
        char_img = remove_background(char_img)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        char_img.save(tmp_path, "PNG")
        character_path = tmp_path

    try:
        backgrounds = (
            sorted(backgrounds_dir.glob("*.png"))
            + sorted(backgrounds_dir.glob("*.jpg"))
            + sorted(backgrounds_dir.glob("*.jpeg"))
        )
        log.info("Compositing onto %d backgrounds...", len(backgrounds))

        for bg_path in backgrounds:
            out_path = output_dir / f"scene_{bg_path.stem}.png"
            composite_character(character_path, bg_path, out_path, **kwargs)
            results.append(out_path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return results
