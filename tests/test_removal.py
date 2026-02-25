"""Tests for ai_video_composite.removal — alpha cleanup and green defringing."""

import numpy as np
import pytest
from PIL import Image

from ai_video_composite.removal import clean_alpha, defringe_green


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_rgba(w: int, h: int, r: int, g: int, b: int, a: int) -> Image.Image:
    """Create a solid-color RGBA image."""
    arr = np.full((h, w, 4), [r, g, b, a], dtype=np.uint8)
    return Image.fromarray(arr, "RGBA")


def _image_with_low_alpha_border(size: int = 100) -> Image.Image:
    """Opaque red center with a semi-transparent (alpha=5) green border."""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    # Fill with low-alpha green border
    arr[:, :] = [0, 255, 0, 5]
    # Opaque red center
    margin = 20
    arr[margin:-margin, margin:-margin] = [255, 0, 0, 255]
    return Image.fromarray(arr, "RGBA")


# ---------------------------------------------------------------------------
# clean_alpha tests
# ---------------------------------------------------------------------------

class TestCleanAlpha:
    def test_removes_low_alpha_pixels(self):
        """Pixels below alpha_floor should become fully transparent."""
        img = _image_with_low_alpha_border()
        result = clean_alpha(img, alpha_floor=10)
        arr = np.array(result)
        # Border pixels (alpha=5) should now be 0
        assert arr[0, 0, 3] == 0
        assert arr[0, -1, 3] == 0
        # Center pixels should be untouched
        assert arr[50, 50, 3] == 255

    def test_preserves_opaque_pixels(self):
        """Fully opaque pixels should not be modified."""
        img = _solid_rgba(50, 50, 128, 64, 32, 255)
        result = clean_alpha(img)
        arr = np.array(result)
        assert arr[25, 25, 0] == 128
        assert arr[25, 25, 1] == 64
        assert arr[25, 25, 2] == 32
        assert arr[25, 25, 3] == 255

    def test_fully_transparent_image_unchanged(self):
        """An already-transparent image should pass through cleanly."""
        img = _solid_rgba(50, 50, 0, 0, 0, 0)
        result = clean_alpha(img)
        arr = np.array(result)
        assert np.all(arr[:, :, 3] == 0)

    def test_custom_alpha_floor(self):
        """Alpha floor parameter should control the threshold."""
        # Create image with alpha=50
        arr = np.full((30, 30, 4), [100, 100, 100, 50], dtype=np.uint8)
        img = Image.fromarray(arr, "RGBA")

        # alpha_floor=60 should remove these pixels
        result = clean_alpha(img, alpha_floor=60)
        assert np.all(np.array(result)[:, :, 3] == 0)

        # alpha_floor=40 should keep them
        result = clean_alpha(img, alpha_floor=40)
        assert np.all(np.array(result)[:, :, 3] == 50)

    def test_island_removal_with_scipy(self):
        """Small isolated pixel clusters should be removed (if scipy available)."""
        pytest.importorskip("scipy")

        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        # Large opaque block (should survive)
        arr[10:60, 10:60] = [255, 0, 0, 255]
        # Tiny 3x3 island (should be removed with min_island_pixels=50)
        arr[80:83, 80:83] = [0, 255, 0, 255]
        img = Image.fromarray(arr, "RGBA")

        result = clean_alpha(img, min_island_pixels=50)
        result_arr = np.array(result)

        # Large block survives
        assert result_arr[30, 30, 3] == 255
        # Tiny island removed
        assert result_arr[81, 81, 3] == 0


# ---------------------------------------------------------------------------
# defringe_green tests
# ---------------------------------------------------------------------------

class TestDefringeGreen:
    def test_no_green_pixels_unchanged(self):
        """Image with no green spill should pass through unchanged."""
        img = _solid_rgba(50, 50, 200, 100, 100, 255)
        result = defringe_green(img)
        # Should be identical (no green to fix)
        assert np.array_equal(np.array(img), np.array(result))

    def test_fully_transparent_unchanged(self):
        """Fully transparent image should pass through unchanged."""
        img = _solid_rgba(50, 50, 0, 255, 0, 0)
        result = defringe_green(img)
        assert np.array_equal(np.array(img), np.array(result))

    def test_green_edge_pixels_shifted(self):
        """Semi-transparent green edge pixels should have their hue rotated."""
        arr = np.zeros((50, 50, 4), dtype=np.uint8)
        # Opaque red center (the "character")
        arr[10:40, 10:40] = [200, 80, 80, 255]
        # Green fringe border (semi-transparent, chroma-green hue)
        arr[8:10, 10:40] = [30, 200, 30, 128]   # top edge
        arr[40:42, 10:40] = [30, 200, 30, 128]   # bottom edge
        arr[10:40, 8:10] = [30, 200, 30, 128]    # left edge
        arr[10:40, 40:42] = [30, 200, 30, 128]   # right edge
        img = Image.fromarray(arr, "RGBA")

        result = defringe_green(img)
        result_arr = np.array(result)

        # The green fringe pixels should have reduced green dominance
        edge_g = result_arr[9, 20, 1]  # green channel of a fringe pixel
        edge_r = result_arr[9, 20, 0]
        # Green should no longer dominate as strongly
        original_excess = 200 - (30 + 30) / 2  # 170
        new_excess = float(edge_g) - (float(edge_r) + float(result_arr[9, 20, 2])) / 2
        assert new_excess < original_excess

    def test_strength_zero_no_change(self):
        """strength=0.0 should produce no color change."""
        arr = np.zeros((50, 50, 4), dtype=np.uint8)
        arr[10:40, 10:40] = [200, 80, 80, 255]
        arr[8:10, 10:40] = [30, 200, 30, 128]
        img = Image.fromarray(arr, "RGBA")

        result = defringe_green(img, strength=0.0)
        # With zero strength, blend is zero, so pixels should be effectively unchanged
        # (floating point rounding may cause tiny differences)
        orig = np.array(img, dtype=np.float32)
        res = np.array(result, dtype=np.float32)
        assert np.allclose(orig, res, atol=2.0)
