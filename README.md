# ai-video-composite

Green-screen removal and compositing toolkit for AI-generated images and video.

## The Problem

AI image and video generators (FLUX, Gemini, Kling, Stable Diffusion, etc.) produce character images on green-screen backgrounds, but the green is never broadcast-standard chroma green — it's pastel sage, lime, or uneven. Traditional FFmpeg chromakey breaks on these outputs.

Even when you use AI background removal (rembg), you get:
- **Semi-transparent ghost pixels** around edges
- **Tiny floating pixel islands** (isolated dots, wisps)
- **Green color spill** bleeding into hair, skin, and clothing edges

## The Solution

A three-stage cleanup pipeline that handles what rembg alone can't:

1. **AI Background Removal** — rembg with the `isnet-anime` model (optimized for illustrated/cartoon characters)
2. **Alpha Cleanup** — threshold low-alpha pixels, remove small isolated clusters (scipy connected components), morphological opening to clean thin wisps
3. **Green Defringing** — hue rotation of green-spill pixels toward the character's natural colors. Unlike naive green channel reduction (which turns edges brown), this preserves brightness and produces natural-looking edges.

Plus compositing tools for both still images and video:
- Single image compositing with positioning and scaling
- Batch compositing (one character onto N backgrounds)
- Green-screen standardization (any background → exact #00b140 chroma green)
- FFmpeg chromakey video compositing (single-layer and multi-layer)

## Installation

Requires **Python 3.9+**.

```bash
# Install from source
pip install -e .

# For GPU-accelerated background removal (CUDA)
pip install -e ".[gpu]"

# All optional dependencies (scipy + bundled FFmpeg)
pip install -e ".[full]"
```

**FFmpeg** is required for video compositing commands. Install it system-wide or use `imageio-ffmpeg`.

The `isnet-anime` rembg model (~170MB) is downloaded automatically on first use.

## Quick Start

```bash
# Remove background → transparent PNG
python -m ai_video_composite remove-bg character.png -o character_transparent.png

# Standardize any green to chroma green (#00b140)
python -m ai_video_composite standardize character.png -o character_green.png

# Composite character onto a background
python -m ai_video_composite compose character.png background.png -o scene.png \
    --position right --scale 0.6

# Multi-layer video composite from JSON config
python -m ai_video_composite video-layers scene.json -o final.mp4
```

## CLI Reference

### `remove-bg` — Background removal

```bash
python -m ai_video_composite remove-bg <input> [-o output]
```

Removes the background using AI (rembg isnet-anime), then cleans alpha channel and removes green spill. Output is a transparent RGBA PNG.

If `-o` is not specified, saves as `{input_stem}_transparent.png`.

### `standardize` — Green-screen standardization

```bash
python -m ai_video_composite standardize <input> [-o output]
```

Converts any-background character image to exact chroma green (#00b140). Useful when your AI generator produces non-standard green that FFmpeg chromakey can't handle. Uses rembg internally, so the background color doesn't matter — it removes whatever is there and composites onto standard green.

### `compose` — Image compositing

```bash
python -m ai_video_composite compose <character> <background> -o <output> \
    [--position left|center|right] \
    [--scale 0.75] \
    [--offset-x 0] \
    [--offset-y 0]
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--position` | `center` | Horizontal anchor: `left` (15%), `center` (50%), `right` (85%) |
| `--scale` | `0.75` | Character height as fraction of background height |
| `--offset-x` | `0` | Horizontal pixel offset (positive = right) |
| `--offset-y` | `0` | Vertical pixel offset from bottom (positive = up) |

Characters are anchored to the bottom of the frame (standing on ground). Automatically detects and removes green backgrounds.

### `batch` — Batch compositing

```bash
python -m ai_video_composite batch <character> <backgrounds_dir> -o <output_dir> \
    [--position center] [--scale 0.75]
```

Composites one character onto every PNG/JPG in a directory. Background removal is done once and reused for all composites.

### `video` — Single-layer video compositing

```bash
python -m ai_video_composite video <video> <background> -o <output> \
    [--similarity 0.3] [--blend 0.1] [--chroma-color 0x00b140]
```

Composites a green-screen video onto a background image using FFmpeg chromakey.

| Arg | Default | Description |
|-----|---------|-------------|
| `--similarity` | `0.3` | Color match tolerance (0.01–1.0, lower = stricter) |
| `--blend` | `0.1` | Edge blending amount (0.0–1.0) |
| `--chroma-color` | `0x00b140` | Green-screen color (FFmpeg hex) |

**Tip:** Use `similarity=0.15` for cleaner results when the green screen is standardized (#00b140).

### `video-layers` — Multi-layer video compositing

```bash
python -m ai_video_composite video-layers <scene.json> -o <output> \
    [--audio auto|mix|none]
```

Composites multiple video layers onto a background from a JSON config file. Each layer is independently chromakeyed, scaled, and positioned.

Audio modes:
- `auto` — Use the first layer with `audio: true` (default)
- `mix` — Mix all layers' audio tracks together
- `none` — Strip all audio

## Scene Config JSON

```json
{
    "name": "tavern_scene",
    "background": "backgrounds/tavern.png",
    "audio": "auto",
    "layers": [
        {
            "name": "bartender",
            "video": "characters/bartender_idle.mp4",
            "position": "left",
            "scale": 0.6,
            "offset_y": 10,
            "chromakey": true,
            "similarity": 0.15,
            "blend": 0.08,
            "audio": false
        },
        {
            "name": "hero",
            "video": "characters/hero_talking.mp4",
            "position": "right",
            "scale": 0.7,
            "chromakey": true,
            "audio": true
        }
    ]
}
```

All paths are relative to the JSON file's directory. See [scene_config_schema.json](scene_config_schema.json) for the full schema.

| Layer Field | Default | Description |
|-------------|---------|-------------|
| `video` | *(required)* | Path to video file |
| `position` | `"center"` | `"left"`, `"center"`, `"right"`, or `"x,y"` pixels |
| `scale` | `0.75` | Height as fraction of background height |
| `offset_x` | `0` | Horizontal pixel offset |
| `offset_y` | `0` | Vertical offset from bottom |
| `chromakey` | `true` | Apply green-screen removal |
| `chroma_color` | `"0x00b140"` | Green-screen color (FFmpeg hex) |
| `similarity` | `0.3` | Chromakey tolerance |
| `blend` | `0.1` | Edge blend amount |
| `audio` | `false` | Include this layer's audio |

## Python API

```python
from ai_video_composite import (
    remove_background,       # Image → transparent RGBA
    clean_alpha,             # RGBA → cleaned RGBA (remove stray pixels)
    defringe_green,          # RGBA → defringed RGBA (remove green spill)
    standardize_greenscreen, # File → standard chroma green file
    composite_character,     # Character + background → composited image
    batch_composite,         # Character + N backgrounds → N composites
    composite_video,         # Green-screen video + background → composited video
    composite_video_layers,  # N videos + background → multi-layer video
    VideoLayer,              # Layer config dataclass
    load_scene_config,       # JSON → (background, layers, audio_mode)
)
```

See [examples/](examples/) for full usage examples.

## How It Works

### Background Removal Pipeline

```
Input image (any background)
    ↓
rembg (isnet-anime model) → raw RGBA with rough alpha
    ↓
clean_alpha()
  ├─ Threshold: alpha < 10 → fully transparent
  ├─ Island removal: connected clusters < 50px deleted (scipy)
  └─ Morphological opening: erode → dilate removes thin wisps
    ↓
defringe_green()
  ├─ Compute HSV for all pixels (vectorized numpy)
  ├─ Find target hue from non-green pixels (circular mean)
  ├─ Identify green-spill pixels (hue 80-170°, excess green)
  ├─ Edge pixels (semi-transparent): aggressive hue rotation
  └─ Interior pixels (opaque): conservative hue rotation
    ↓
Clean transparent RGBA output
```

### Why Hue Rotation Instead of Green Channel Reduction

Naive approaches reduce the green channel of fringe pixels, but this produces brown/dark edges that look worse than the original green spill. Hue rotation shifts green pixels toward the character's natural colors (skin tone, hair color) while preserving brightness, producing edges that blend naturally with the character.

### FFmpeg Multi-Layer Filter Chain

The `video-layers` command builds a dynamic FFmpeg filter graph:

```
[1:v] chromakey → scale [l0]      # Layer 0: remove green, resize
[2:v] chromakey → scale [l1]      # Layer 1: remove green, resize
[0:v][l0] overlay=x:y [c0]       # Stack layer 0 on background
[c0][l1] overlay=x:y [out]       # Stack layer 1 on result
```

Each layer can have independent chromakey settings (color, similarity, blend), scale, and position.

## Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| `numpy` | Yes | Vectorized pixel operations |
| `Pillow` | Yes | Image loading, compositing, filters |
| `rembg` | Yes | AI background removal (isnet-anime model) |
| `scipy` | No | Better alpha cleanup (connected component island removal) |
| `imageio-ffmpeg` | No | Bundled FFmpeg binary (if not on system PATH) |
| **FFmpeg** | For video | Video compositing, chromakey, encoding |

## License

MIT
