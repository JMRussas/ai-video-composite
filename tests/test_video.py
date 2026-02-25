"""Tests for ai_video_composite.video — position computation, scene config loading."""

import json
import tempfile
from pathlib import Path

import pytest

from ai_video_composite.video import (
    _compute_position,
    load_scene_config,
    VideoLayer,
)


# ---------------------------------------------------------------------------
# _compute_position tests
# ---------------------------------------------------------------------------

class TestComputePosition:
    """Test position calculation for video layers."""

    def test_center_position(self):
        """Center should horizontally center the layer, anchored to bottom."""
        x, y = _compute_position("center", bg_w=800, bg_h=600, layer_w=200, layer_h=400)
        assert x == 300   # (800 - 200) / 2
        assert y == 200   # 600 - 400

    def test_left_position(self):
        """Left should place layer at 15% of background width."""
        x, y = _compute_position("left", bg_w=1000, bg_h=600, layer_w=200, layer_h=400)
        assert x == 50    # int(1000 * 0.15) - 200 // 2 = 150 - 100
        assert y == 200   # 600 - 400

    def test_right_position(self):
        """Right should place layer at 85% of background width."""
        x, y = _compute_position("right", bg_w=1000, bg_h=600, layer_w=200, layer_h=400)
        assert x == 750   # int(1000 * 0.85) - 200 // 2 = 850 - 100
        assert y == 200   # 600 - 400

    def test_pixel_position(self):
        """Explicit 'x,y' format should set exact coordinates."""
        x, y = _compute_position("100, 50", bg_w=800, bg_h=600, layer_w=200, layer_h=400)
        assert x == 100
        assert y == 50

    def test_offset_x_positive(self):
        """Positive offset_x should shift right."""
        x1, _ = _compute_position("center", bg_w=800, bg_h=600, layer_w=200, layer_h=400)
        x2, _ = _compute_position("center", bg_w=800, bg_h=600, layer_w=200, layer_h=400, offset_x=30)
        assert x2 == x1 + 30

    def test_offset_y_positive(self):
        """Positive offset_y should shift upward (lower y value)."""
        _, y1 = _compute_position("center", bg_w=800, bg_h=600, layer_w=200, layer_h=400)
        _, y2 = _compute_position("center", bg_w=800, bg_h=600, layer_w=200, layer_h=400, offset_y=20)
        assert y2 == y1 - 20

    def test_combined_offsets(self):
        """Both offsets applied together."""
        x, y = _compute_position(
            "center", bg_w=800, bg_h=600, layer_w=200, layer_h=400,
            offset_x=10, offset_y=25,
        )
        assert x == 310   # 300 + 10
        assert y == 175    # 200 - 25


# ---------------------------------------------------------------------------
# VideoLayer tests
# ---------------------------------------------------------------------------

class TestVideoLayer:
    """Test VideoLayer dataclass defaults."""

    def test_defaults(self):
        layer = VideoLayer(video="test.mp4")
        assert layer.position == "center"
        assert layer.scale == 0.75
        assert layer.offset_x == 0
        assert layer.offset_y == 0
        assert layer.chromakey is True
        assert layer.chroma_color == "0x00b140"
        assert layer.similarity == 0.3
        assert layer.blend == 0.1
        assert layer.audio is False

    def test_custom_values(self):
        layer = VideoLayer(
            video="char.mp4",
            position="left",
            scale=0.5,
            offset_x=10,
            offset_y=20,
            chromakey=False,
            chroma_color="0x00ff00",
            similarity=0.15,
            blend=0.05,
            audio=True,
        )
        assert layer.video == "char.mp4"
        assert layer.position == "left"
        assert layer.scale == 0.5
        assert layer.chromakey is False
        assert layer.audio is True


# ---------------------------------------------------------------------------
# load_scene_config tests
# ---------------------------------------------------------------------------

class TestLoadSceneConfig:
    """Test JSON scene config loading."""

    def _write_config(self, config: dict) -> Path:
        """Write a config dict to a temp JSON file and return its path."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=tempfile.gettempdir()
        )
        json.dump(config, tmp)
        tmp.close()
        return Path(tmp.name)

    def test_basic_config(self):
        """Should parse a valid scene config into background, layers, audio."""
        config = {
            "name": "test_scene",
            "background": "bg.png",
            "audio": "mix",
            "layers": [
                {
                    "name": "hero",
                    "video": "hero.mp4",
                    "position": "right",
                    "scale": 0.6,
                    "audio": True,
                },
            ],
        }
        path = self._write_config(config)
        try:
            bg, layers, audio_mode = load_scene_config(path)

            assert bg == path.parent / "bg.png"
            assert audio_mode == "mix"
            assert len(layers) == 1
            assert layers[0].position == "right"
            assert layers[0].scale == 0.6
            assert layers[0].audio is True
        finally:
            path.unlink(missing_ok=True)

    def test_defaults_applied(self):
        """Missing optional fields should use VideoLayer defaults."""
        config = {
            "background": "bg.png",
            "layers": [{"video": "char.mp4"}],
        }
        path = self._write_config(config)
        try:
            bg, layers, audio_mode = load_scene_config(path)

            assert audio_mode == "auto"
            assert layers[0].position == "center"
            assert layers[0].scale == 0.75
            assert layers[0].chromakey is True
        finally:
            path.unlink(missing_ok=True)

    def test_multiple_layers(self):
        """Should correctly parse multiple layers."""
        config = {
            "background": "bg.png",
            "layers": [
                {"video": "a.mp4", "position": "left", "scale": 0.4},
                {"video": "b.mp4", "position": "right", "scale": 0.7, "audio": True},
            ],
        }
        path = self._write_config(config)
        try:
            _, layers, _ = load_scene_config(path)

            assert len(layers) == 2
            assert layers[0].position == "left"
            assert layers[0].scale == 0.4
            assert layers[1].position == "right"
            assert layers[1].audio is True
        finally:
            path.unlink(missing_ok=True)

    def test_paths_relative_to_config(self):
        """All paths should be resolved relative to the JSON file's directory."""
        config = {
            "background": "scenes/bg.png",
            "layers": [{"video": "chars/hero.mp4"}],
        }
        path = self._write_config(config)
        try:
            bg, layers, _ = load_scene_config(path)

            assert bg == path.parent / "scenes" / "bg.png"
            assert layers[0].video == str(path.parent / "chars" / "hero.mp4")
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_json_raises_valueerror(self):
        """Malformed JSON should raise ValueError with helpful message."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=tempfile.gettempdir()
        )
        tmp.write("{invalid json")
        tmp.close()
        path = Path(tmp.name)

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                load_scene_config(path)
        finally:
            path.unlink(missing_ok=True)

    def test_chromakey_settings_passed(self):
        """Per-layer chromakey settings should be loaded."""
        config = {
            "background": "bg.png",
            "layers": [{
                "video": "char.mp4",
                "chromakey": False,
                "chroma_color": "0x00ff00",
                "similarity": 0.15,
                "blend": 0.05,
            }],
        }
        path = self._write_config(config)
        try:
            _, layers, _ = load_scene_config(path)

            assert layers[0].chromakey is False
            assert layers[0].chroma_color == "0x00ff00"
            assert layers[0].similarity == 0.15
            assert layers[0].blend == 0.05
        finally:
            path.unlink(missing_ok=True)
