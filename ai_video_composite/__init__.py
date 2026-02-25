"""ai-video-composite — Green-screen removal and compositing for AI-generated media.

AI image and video generators (FLUX, Gemini, Kling, etc.) produce character
images on green-screen backgrounds, but the green is never broadcast-standard
chroma green — it's pastel sage, lime, or uneven. Traditional chromakey breaks.

This package provides:
  1. AI background removal (rembg isnet-anime) with alpha cleanup and green
     spill defringing — handles the messy edges that rembg alone leaves behind.
  2. Green-screen standardization — converts any background to exact #00b140
     chroma green so FFmpeg chromakey works reliably.
  3. Image compositing — character onto background with positioning/scaling.
  4. Video compositing — single-layer and multi-layer FFmpeg chromakey.
"""

from .removal import (
    remove_background,
    clean_alpha,
    defringe_green,
    standardize_greenscreen,
    remove_background_file,
)
from .compositing import (
    composite_character,
    batch_composite,
)
from .video import (
    composite_video,
    composite_video_layers,
    VideoLayer,
    load_scene_config,
)

__all__ = [
    "remove_background",
    "clean_alpha",
    "defringe_green",
    "standardize_greenscreen",
    "remove_background_file",
    "composite_character",
    "batch_composite",
    "composite_video",
    "composite_video_layers",
    "VideoLayer",
    "load_scene_config",
]
