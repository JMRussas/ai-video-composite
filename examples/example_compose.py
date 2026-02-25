"""Example: composite a character onto a background using the Python API."""
from pathlib import Path

from ai_video_composite import remove_background, composite_character
from PIL import Image

# --- Step 1: Remove background from a character image ---
char = Image.open("my_character.png")
transparent = remove_background(char)
transparent.save("character_transparent.png")

# --- Step 2: Composite onto a background ---
composite_character(
    Path("character_transparent.png"),
    Path("background.png"),
    Path("output.png"),
    position="right",   # "left", "center", or "right"
    scale=0.7,           # Character height = 70% of background height
    offset_y=20,         # 20px up from bottom edge
)
