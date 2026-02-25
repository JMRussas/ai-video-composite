"""Tests for ai_video_composite.compositing — image compositing and green detection."""

import gc
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from ai_video_composite.compositing import (
    _is_green_background,
    composite_character,
    _MIN_SAMPLE_SIZE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_rgb(w: int, h: int, r: int, g: int, b: int) -> Image.Image:
    """Create a solid-color RGB image."""
    arr = np.full((h, w, 3), [r, g, b], dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _solid_rgba(w: int, h: int, r: int, g: int, b: int, a: int) -> Image.Image:
    """Create a solid-color RGBA image."""
    arr = np.full((h, w, 4), [r, g, b, a], dtype=np.uint8)
    return Image.fromarray(arr, "RGBA")


def _save_temp_image(img: Image.Image, suffix: str = ".png") -> Path:
    """Save an image to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    path = Path(tmp.name)
    tmp.close()
    img.save(path)
    return path


def _cleanup(*paths: Path):
    """Best-effort temp file cleanup (handles Windows file locking)."""
    gc.collect()
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except PermissionError:
            pass  # Windows file locking — temp dir will clean up eventually


# ---------------------------------------------------------------------------
# _is_green_background tests
# ---------------------------------------------------------------------------

class TestIsGreenBackground:
    def test_chroma_green_detected(self):
        """Standard chroma green (#00b140) should be detected."""
        img = _solid_rgb(100, 100, 0, 177, 64)
        assert _is_green_background(img) is True

    def test_bright_green_detected(self):
        """Bright green background should be detected."""
        img = _solid_rgb(100, 100, 20, 200, 20)
        assert _is_green_background(img) is True

    def test_blue_not_detected(self):
        """Blue background should not be detected as green."""
        img = _solid_rgb(100, 100, 0, 0, 200)
        assert _is_green_background(img) is False

    def test_red_not_detected(self):
        """Red background should not be detected as green."""
        img = _solid_rgb(100, 100, 200, 0, 0)
        assert _is_green_background(img) is False

    def test_white_not_detected(self):
        """White background should not be detected (green not dominant)."""
        img = _solid_rgb(100, 100, 255, 255, 255)
        assert _is_green_background(img) is False

    def test_small_image_returns_false(self):
        """Images smaller than minimum sample size should return False."""
        img = _solid_rgb(5, 5, 0, 255, 0)
        assert _is_green_background(img) is False

    def test_boundary_size(self):
        """Image exactly at minimum sample size should work."""
        img = _solid_rgb(_MIN_SAMPLE_SIZE, _MIN_SAMPLE_SIZE, 0, 200, 0)
        assert _is_green_background(img) is True

    def test_rgba_input_works(self):
        """RGBA images should be handled (converted internally)."""
        img = _solid_rgba(100, 100, 0, 200, 0, 255)
        assert _is_green_background(img) is True

    def test_mixed_corners_threshold(self):
        """Need at least 3 of 4 corners green to return True."""
        arr = np.full((100, 100, 3), [0, 200, 0], dtype=np.uint8)
        # Make two corners red (should fail — only 2 green corners)
        arr[0:10, 0:10] = [200, 0, 0]       # top-left
        arr[0:10, 90:100] = [200, 0, 0]     # top-right
        img = Image.fromarray(arr, "RGB")
        assert _is_green_background(img) is False

        # Make only one corner red (3 green corners — should pass)
        arr2 = np.full((100, 100, 3), [0, 200, 0], dtype=np.uint8)
        arr2[0:10, 0:10] = [200, 0, 0]      # top-left only
        img2 = Image.fromarray(arr2, "RGB")
        assert _is_green_background(img2) is True


# ---------------------------------------------------------------------------
# composite_character tests
# ---------------------------------------------------------------------------

class TestCompositeCharacter:
    @patch("ai_video_composite.compositing.remove_background")
    def test_basic_composite(self, mock_remove_bg):
        """Character should be composited onto background at correct position."""
        # Create a small transparent character (red square on transparent)
        char_arr = np.zeros((50, 30, 4), dtype=np.uint8)
        char_arr[10:40, 5:25] = [255, 0, 0, 255]
        char_img = Image.fromarray(char_arr, "RGBA")

        # Background: blue
        bg_img = _solid_rgb(200, 200, 0, 0, 255)

        char_path = _save_temp_image(char_img)
        bg_path = _save_temp_image(bg_img)
        out_path = Path(tempfile.mktemp(suffix=".png"))

        try:
            result = composite_character(
                char_path, bg_path, out_path,
                position="center", scale=0.5,
            )
            assert result == out_path
            assert out_path.exists()

            # Output should be the background size
            with Image.open(out_path) as output_img:
                assert output_img.size == (200, 200)
        finally:
            _cleanup(char_path, bg_path, out_path)

        # remove_background should NOT be called (input is already RGBA with no green)
        mock_remove_bg.assert_not_called()

    @patch("ai_video_composite.compositing.remove_background")
    def test_green_background_triggers_removal(self, mock_remove_bg):
        """Green-screen character should trigger background removal."""
        # Create a green-screen character
        char_img = _solid_rgb(100, 100, 0, 200, 0)
        bg_img = _solid_rgb(200, 200, 0, 0, 255)

        # Mock remove_background to return a transparent image
        transparent = _solid_rgba(100, 100, 255, 0, 0, 255)
        mock_remove_bg.return_value = transparent

        char_path = _save_temp_image(char_img)
        bg_path = _save_temp_image(bg_img)
        out_path = Path(tempfile.mktemp(suffix=".png"))

        try:
            composite_character(char_path, bg_path, out_path)
            mock_remove_bg.assert_called_once()
        finally:
            _cleanup(char_path, bg_path, out_path)

    def test_position_left(self):
        """Left position should place character near left edge."""
        char_arr = np.zeros((50, 30, 4), dtype=np.uint8)
        char_arr[:, :] = [255, 0, 0, 255]
        char_img = Image.fromarray(char_arr, "RGBA")
        bg_img = _solid_rgb(400, 200, 0, 0, 255)

        char_path = _save_temp_image(char_img)
        bg_path = _save_temp_image(bg_img)
        out_path = Path(tempfile.mktemp(suffix=".png"))

        try:
            composite_character(
                char_path, bg_path, out_path,
                position="left", scale=0.5,
            )
            output_img = Image.open(out_path).convert("RGB")
            arr = np.array(output_img)
            # Character should be in the left 30% of the image
            # Check that there's red (character) in the left region
            left_region = arr[:, :120, :]
            has_red = np.any((left_region[:, :, 0] > 200) & (left_region[:, :, 2] < 50))
            assert has_red, "Character should be in the left region"
        finally:
            _cleanup(char_path, bg_path, out_path)

    def test_position_right(self):
        """Right position should place character near right edge."""
        char_arr = np.zeros((50, 30, 4), dtype=np.uint8)
        char_arr[:, :] = [255, 0, 0, 255]
        char_img = Image.fromarray(char_arr, "RGBA")
        bg_img = _solid_rgb(400, 200, 0, 0, 255)

        char_path = _save_temp_image(char_img)
        bg_path = _save_temp_image(bg_img)
        out_path = Path(tempfile.mktemp(suffix=".png"))

        try:
            composite_character(
                char_path, bg_path, out_path,
                position="right", scale=0.5,
            )
            output_img = Image.open(out_path).convert("RGB")
            arr = np.array(output_img)
            # Character should be in the right 30% of the image
            right_region = arr[:, 280:, :]
            has_red = np.any((right_region[:, :, 0] > 200) & (right_region[:, :, 2] < 50))
            assert has_red, "Character should be in the right region"
        finally:
            _cleanup(char_path, bg_path, out_path)
