"""Example: multi-layer video compositing from a scene config JSON."""
from pathlib import Path

from ai_video_composite import load_scene_config, composite_video_layers, VideoLayer

# --- Option 1: Load from JSON config file ---
background, layers, audio_mode = load_scene_config(Path("scene_config.json"))
composite_video_layers(background, layers, Path("output.mp4"), audio_mode=audio_mode)

# --- Option 2: Build layers programmatically ---
layers = [
    VideoLayer(
        video="characters/dog_idle.mp4",
        position="left",
        scale=0.35,
        offset_x=30,
        offset_y=20,
        chromakey=True,
        similarity=0.15,    # Stricter key for small characters
        blend=0.08,
        audio=False,
    ),
    VideoLayer(
        video="characters/hero_talking.mp4",
        position="right",
        scale=0.65,
        offset_y=30,
        chromakey=True,
        similarity=0.15,
        blend=0.08,
        audio=True,          # Use this layer's audio
    ),
]

composite_video_layers(
    Path("backgrounds/tavern.png"),
    layers,
    Path("tavern_scene.mp4"),
    audio_mode="auto",       # "auto", "mix", or "none"
)
