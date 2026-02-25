#  ai-video-composite - Video Compositing Module
#
#  FFmpeg-based video compositing: single-layer chromakey, multi-layer
#  compositing with independent per-layer settings, and scene config loading.
#
#  Depends on: Pillow (for _probe_dimensions fallback), FFmpeg (external)
#  Used by:    __main__.py, __init__.py

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FFmpeg auto-detection (system PATH or imageio-ffmpeg fallback)
# ---------------------------------------------------------------------------


def _find_ffmpeg() -> str:
    """Find ffmpeg binary — system PATH first, then imageio-ffmpeg bundled copy."""
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"  # Hope for the best


FFMPEG = _find_ffmpeg()


# ---------------------------------------------------------------------------
# Single-Layer Video Compositing
# ---------------------------------------------------------------------------

def composite_video(
    video_path: Path,
    background_path: Path,
    output_path: Path,
    similarity: float = 0.3,
    blend: float = 0.1,
    chroma_color: str = "0x00b140",
) -> Path:
    """Composite a green-screen video onto a background image using FFmpeg chromakey.

    Args:
        video_path: Path to green-screen video (.mp4)
        background_path: Path to background image (.png/.jpg)
        output_path: Where to save the composited video
        similarity: Chromakey color similarity threshold (0.01-1.0, lower = stricter)
        blend: Edge blending amount (0.0-1.0)
        chroma_color: Green-screen color in FFmpeg hex format (default: 0x00b140)

    Returns:
        Path to saved composite video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg filter: scale background to video size, chromakey the video, overlay
    filter_complex = (
        f"[0:v]chromakey={chroma_color}:{similarity}:{blend}[fg];"
        f"[1:v]scale=iw:ih[bg];"
        f"[bg][fg]overlay=0:0:shortest=1"
    )

    cmd = [
        FFMPEG, "-y",
        "-i", str(video_path),
        "-i", str(background_path),
        "-filter_complex", filter_complex,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        str(output_path),
    ]

    log.info("FFmpeg compositing: %s + %s", video_path.name, background_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        log.error("FFmpeg error: %s", result.stderr[-500:])
        raise RuntimeError(f"FFmpeg failed with return code {result.returncode}")

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        log.info("Saved: %s (%.1fMB)", output_path.name, size_mb)
    return output_path


# ---------------------------------------------------------------------------
# Multi-Layer Video Compositing
# ---------------------------------------------------------------------------

@dataclass
class VideoLayer:
    """A single video layer for multi-layer compositing.

    Each layer is a video (typically with green-screen or transparent background)
    that gets chromakeyed, scaled, and positioned onto the composite. Layers are
    rendered bottom-to-top (first in list = closest to background).
    """
    video: str                      # Path to video file
    position: str = "center"        # "left", "center", "right", or "x,y" pixels
    scale: float = 0.75             # Height as fraction of background height
    offset_x: int = 0              # Horizontal pixel offset (positive = right)
    offset_y: int = 0              # Vertical pixel offset from bottom (positive = up)
    chromakey: bool = True          # Apply chromakey removal (False = video has alpha)
    chroma_color: str = "0x00b140"  # Green screen color
    similarity: float = 0.3        # Chromakey similarity threshold
    blend: float = 0.1             # Chromakey edge blend
    audio: bool = False            # Include this layer's audio in output


def _probe_dimensions(path: Path) -> tuple[int, int]:
    """Get width and height of a video or image.

    Uses PIL for images, falls back to ffmpeg stream info for videos.
    """
    # Try PIL first (fast, works for all image formats)
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except (OSError, ValueError):
        pass  # Not an image PIL can open — fall back to ffmpeg

    # Fall back to ffmpeg -i (parses stderr for stream dimensions)
    cmd = [FFMPEG, "-i", str(path), "-hide_banner"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    # Parse: "Stream #0:0: Video: h264 ..., 512x768, ..."
    match = re.search(r"Stream.*Video.* (\d{2,5})x(\d{2,5})", result.stderr)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise RuntimeError(f"Could not determine dimensions of {path}")


def _compute_position(
    position: str,
    bg_w: int, bg_h: int,
    layer_w: int, layer_h: int,
    offset_x: int = 0, offset_y: int = 0,
) -> tuple[int, int]:
    """Convert named position + offsets to pixel coordinates.

    Characters are anchored to the bottom of the frame (standing on ground).
    """
    if "," in str(position):
        px, py = position.split(",")
        x, y = int(px.strip()), int(py.strip())
    elif position == "left":
        x = int(bg_w * 0.15) - layer_w // 2
        y = bg_h - layer_h
    elif position == "right":
        x = int(bg_w * 0.85) - layer_w // 2
        y = bg_h - layer_h
    else:  # center
        x = (bg_w - layer_w) // 2
        y = bg_h - layer_h

    return x + offset_x, y - offset_y


def composite_video_layers(
    background: Path,
    layers: list[VideoLayer],
    output: Path,
    audio_mode: str = "auto",
) -> Path:
    """Composite multiple video layers onto a background.

    Each layer is independently chromakeyed (or alpha-blended), scaled, and
    positioned. Layers are rendered bottom-to-top.

    Args:
        background: Path to background image or video.
        layers: List of VideoLayer specs.
        output: Output video path.
        audio_mode: How to handle audio:
            "auto" — use the first layer with audio=True (or first layer)
            "mix"  — mix all layers' audio tracks
            "none" — no audio output

    Returns:
        Path to output video.
    """
    if not layers:
        raise ValueError("Need at least one layer")

    output.parent.mkdir(parents=True, exist_ok=True)

    # Probe background dimensions
    bg_w, bg_h = _probe_dimensions(background)
    log.info("Background: %dx%d", bg_w, bg_h)

    # Detect if background is a static image (needs -loop 1)
    bg_is_image = background.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    # Build FFmpeg inputs
    if bg_is_image:
        inputs = ["-loop", "1", "-i", str(background)]
    else:
        inputs = ["-i", str(background)]
    for layer in layers:
        inputs.extend(["-i", str(layer.video)])

    # Probe each layer and compute scale/position
    layer_info = []  # (target_w, target_h, x, y) per layer
    for layer in layers:
        src_w, src_h = _probe_dimensions(Path(layer.video))
        target_h = int(bg_h * layer.scale)
        target_w = int(target_h * src_w / src_h)
        # Round to even (required by many codecs)
        target_w += target_w % 2
        target_h += target_h % 2
        x, y = _compute_position(
            layer.position, bg_w, bg_h,
            target_w, target_h,
            layer.offset_x, layer.offset_y,
        )
        layer_info.append((target_w, target_h, x, y))
        log.info("Layer %d: %s @ %s (%dx%d, pos=%d,%d)",
                 len(layer_info) - 1, Path(layer.video).name,
                 layer.position, target_w, target_h, x, y)

    # Build filter chain
    filters = []

    # Per-layer: chromakey (optional) + scale
    for i, layer in enumerate(layers):
        stream_idx = i + 1  # 0 is background
        tw, th, _, _ = layer_info[i]

        layer_filters = []
        if layer.chromakey:
            layer_filters.append(
                f"chromakey={layer.chroma_color}:{layer.similarity}:{layer.blend}"
            )
        layer_filters.append(f"scale={tw}:{th}")

        filters.append(f"[{stream_idx}:v]{','.join(layer_filters)}[l{i}]")

    # Chain overlays: bg + l0 → c0, c0 + l1 → c1, ...
    prev = "0:v"
    for i in range(len(layers)):
        _, _, x, y = layer_info[i]
        is_last = (i == len(layers) - 1)
        out_label = "out" if is_last else f"c{i}"
        filters.append(f"[{prev}][l{i}]overlay={x}:{y}:shortest=1[{out_label}]")
        prev = out_label

    filter_complex = ";\n    ".join(filters)

    # Audio mapping
    audio_maps = []
    if audio_mode == "none":
        audio_maps = ["-an"]
    elif audio_mode == "mix":
        audio_inputs = [f"[{i+1}:a]" for i in range(len(layers))]
        if len(audio_inputs) > 1:
            mix_filter = (
                f"{''.join(audio_inputs)}"
                f"amix=inputs={len(audio_inputs)}:duration=longest[aout]"
            )
            filter_complex += ";\n    " + mix_filter
            audio_maps = ["-map", "[aout]"]
        elif audio_inputs:
            audio_maps = ["-map", "1:a?"]
    else:  # auto
        audio_idx = None
        for i, layer in enumerate(layers):
            if layer.audio:
                audio_idx = i + 1
                break
        if audio_idx is None:
            audio_idx = 1  # default to first layer
        audio_maps = ["-map", f"{audio_idx}:a?"]

    # Build full command
    cmd = [
        FFMPEG, "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        *audio_maps,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    ]
    if audio_mode != "none":
        cmd.extend(["-c:a", "aac"])
    cmd.append(str(output))

    log.info("Compositing %d layers onto background...", len(layers))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        log.error("FFmpeg error: %s", result.stderr[-500:])
        raise RuntimeError(f"FFmpeg failed with return code {result.returncode}")

    if output.exists():
        size_mb = output.stat().st_size / (1024 * 1024)
        log.info("Saved: %s (%.1fMB)", output.name, size_mb)
    return output


# ---------------------------------------------------------------------------
# Scene Config Loading
# ---------------------------------------------------------------------------

def load_scene_config(config_path: Path) -> tuple[Path, list[VideoLayer], str]:
    """Load a scene compositing config from JSON.

    JSON format:
    {
        "name": "scene_name",
        "background": "backgrounds/scene.png",
        "audio": "auto",
        "layers": [
            {
                "name": "character_a",
                "video": "characters/char_a_idle.mp4",
                "position": "left",
                "scale": 0.4,
                "offset_y": 20,
                "chromakey": true,
                "audio": false
            }
        ]
    }

    Paths are resolved relative to the JSON file's directory.

    Returns:
        Tuple of (background_path, layers, audio_mode).

    Raises:
        FileNotFoundError: If config_path does not exist.
        ValueError: If the JSON is malformed or missing required fields.
    """
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in scene config '{config_path}': {e}") from e

    base_dir = config_path.parent

    background = base_dir / config["background"]
    audio_mode = config.get("audio", "auto")

    layers = []
    for layer_cfg in config["layers"]:
        layers.append(VideoLayer(
            video=str(base_dir / layer_cfg["video"]),
            position=layer_cfg.get("position", "center"),
            scale=layer_cfg.get("scale", 0.75),
            offset_x=layer_cfg.get("offset_x", 0),
            offset_y=layer_cfg.get("offset_y", 0),
            chromakey=layer_cfg.get("chromakey", True),
            chroma_color=layer_cfg.get("chroma_color", "0x00b140"),
            similarity=layer_cfg.get("similarity", 0.3),
            blend=layer_cfg.get("blend", 0.1),
            audio=layer_cfg.get("audio", False),
        ))

    layer_names = ", ".join(l.get("name", f"layer{i}") for i, l in enumerate(config["layers"]))
    log.info("Scene: %s", config.get("name", config_path.stem))
    log.info("Background: %s", background.name)
    log.info("Layers: %d (%s)", len(layers), layer_names)

    return background, layers, audio_mode
