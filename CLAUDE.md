# ai-video-composite

Green-screen removal and compositing toolkit for AI-generated images and video.
Handles the messy edges, green spill, and non-standard green that AI generators
(FLUX, Gemini, Kling) produce — things rembg and FFmpeg chromakey can't fix alone.

## Quick Reference

| Item | Value |
|------|-------|
| **Language** | Python 3.9+ |
| **Dependencies** | numpy, Pillow, rembg, scipy (optional), FFmpeg (for video) |
| **Install** | `pip install -e .` |
| **Install (GPU)** | `pip install -e ".[gpu]"` |
| **Install (all)** | `pip install -e ".[full]"` |
| **Run CLI** | `python -m ai_video_composite <command>` |
| **Run tests** | `python -m pytest tests/ -v` |

## Project Structure

```
ai-video-composite/
  CLAUDE.md                    This file
  pyproject.toml               Package config, deps, entry points
  requirements.txt             Legacy deps file (prefer pyproject.toml)
  scene_config_schema.json     JSON schema for video-layers scene configs
  ai_video_composite/
    __init__.py                Public API re-exports
    __main__.py                CLI (argparse)
    removal.py                 BG removal, alpha cleanup, green defringe
    compositing.py             Image compositing (single + batch)
    video.py                   FFmpeg video compositing, scene config
  examples/
    example_compose.py         Python API: single image composite
    example_multilayer.py      Python API: multi-layer video from JSON
    example_video.py           Python API: video composite + standardize
    scene_config.json          Example scene config for video-layers
  tests/
    test_removal.py            Tests for clean_alpha, defringe_green
    test_compositing.py        Tests for green detection, compositing
    test_video.py              Tests for position calc, scene config
```

## Module Dependency Map

| Module | Role | Depends On | Used By |
|--------|------|------------|---------|
| `removal.py` | AI background removal, alpha cleanup, green defringe | numpy, Pillow, rembg | compositing.py, __main__.py |
| `compositing.py` | Image compositing, green-screen detection | removal.py, Pillow | __main__.py |
| `video.py` | FFmpeg chromakey, multi-layer compositing, scene config | Pillow, FFmpeg (external) | __main__.py |
| `__main__.py` | CLI entry point | removal.py, compositing.py, video.py | — |

## Conventions

- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for module constants, `_prefix` for private/internal functions
- **Logging**: Use `logging.getLogger(__name__)` — no print statements
- **File I/O**: All file operations use `pathlib.Path`, never raw strings
- **Image format**: Internal processing uses RGBA PIL Images; final outputs convert to RGB
- **Error handling**: `Image.open()` calls wrapped with context messages; JSON parsing catches `JSONDecodeError`
- **FFmpeg calls**: Always use `subprocess.run()` with list args (never `shell=True`)
- **Optional deps**: `scipy` and `imageio-ffmpeg` degrade gracefully with fallback paths

## CLI Commands

| Command | Description |
|---------|-------------|
| `remove-bg` | AI background removal → transparent PNG |
| `standardize` | Any background → standard chroma green (#00b140) |
| `compose` | Character + background → composited image |
| `batch` | Character + N backgrounds → N composites |
| `video` | Green-screen video + background → composited video |
| `video-layers` | Multi-layer video composite from JSON config |

## Key Algorithms

- **Alpha cleanup** (`clean_alpha`): 3-stage — threshold low-alpha, remove small islands (scipy connected components), morphological opening
- **Green defringe** (`defringe_green`): Hue rotation of green-spill pixels toward character's natural colors (circular mean of non-green hues). Avoids naive green channel reduction which produces brown edges.
- **Green detection** (`_is_green_background`): Corner-sampling heuristic, requires 3/4 corners green-dominant. Guards against images < 11px.
