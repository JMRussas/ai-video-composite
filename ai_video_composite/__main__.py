#  ai-video-composite - CLI Entry Point
#
#  Command-line interface for background removal, green-screen standardization,
#  image compositing, and video compositing.
#
#  Depends on: removal.py, compositing.py, video.py
#  Used by:    (invoked directly via python -m ai_video_composite)

import argparse
import logging
from pathlib import Path

from .removal import remove_background_file, standardize_greenscreen
from .compositing import composite_character, batch_composite
from .video import composite_video, composite_video_layers, load_scene_config


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="AI Video Composite — green-screen removal and compositing for AI-generated media",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # remove-bg
    p_rembg = subparsers.add_parser("remove-bg", help="Remove background from character image")
    p_rembg.add_argument("input", help="Input character image")
    p_rembg.add_argument("-o", "--output", help="Output path (default: input_transparent.png)")

    # standardize
    p_std = subparsers.add_parser(
        "standardize",
        help="Convert any-background character to standard chroma green (#00b140)",
    )
    p_std.add_argument("input", help="Input character image (any background)")
    p_std.add_argument("-o", "--output", help="Output path (default: input_green.png)")

    # compose
    p_compose = subparsers.add_parser("compose", help="Composite character onto background")
    p_compose.add_argument("character", help="Character image (green screen or transparent)")
    p_compose.add_argument("background", help="Background/scene image")
    p_compose.add_argument("-o", "--output", required=True, help="Output path")
    p_compose.add_argument("--position", choices=["left", "center", "right"], default="center")
    p_compose.add_argument("--scale", type=float, default=0.75, help="Character height as fraction of bg (0-1)")
    p_compose.add_argument("--offset-x", type=int, default=0, help="Horizontal offset in pixels")
    p_compose.add_argument("--offset-y", type=int, default=0, help="Vertical offset from bottom in pixels")

    # batch
    p_batch = subparsers.add_parser("batch", help="Composite character onto all backgrounds in a directory")
    p_batch.add_argument("character", help="Character image")
    p_batch.add_argument("backgrounds", help="Directory of background images")
    p_batch.add_argument("-o", "--output", required=True, help="Output directory")
    p_batch.add_argument("--position", choices=["left", "center", "right"], default="center")
    p_batch.add_argument("--scale", type=float, default=0.75)
    p_batch.add_argument("--offset-x", type=int, default=0)
    p_batch.add_argument("--offset-y", type=int, default=0)

    # video
    p_video = subparsers.add_parser("video", help="Composite green-screen video onto background")
    p_video.add_argument("video", help="Green-screen video (.mp4)")
    p_video.add_argument("background", help="Background image")
    p_video.add_argument("-o", "--output", required=True, help="Output video path")
    p_video.add_argument("--similarity", type=float, default=0.3, help="Chromakey similarity (0.01-1.0)")
    p_video.add_argument("--blend", type=float, default=0.1, help="Edge blend (0-1)")
    p_video.add_argument("--chroma-color", default="0x00b140", help="Green-screen color (FFmpeg hex, default: 0x00b140)")

    # video-layers
    p_vlayers = subparsers.add_parser(
        "video-layers",
        help="Multi-layer video composite from scene config JSON",
    )
    p_vlayers.add_argument("scene", help="Scene config JSON file")
    p_vlayers.add_argument("-o", "--output", required=True, help="Output video path")
    p_vlayers.add_argument(
        "--audio", choices=["auto", "mix", "none"], default=None,
        help="Audio mode (overrides scene config): auto, mix, none",
    )

    args = parser.parse_args()

    if args.command == "remove-bg":
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else input_path.with_stem(f"{input_path.stem}_transparent")
        remove_background_file(input_path, output_path)

    elif args.command == "standardize":
        input_path = Path(args.input)
        output_path = (
            Path(args.output) if args.output
            else input_path.with_stem(f"{input_path.stem}_green")
        )
        standardize_greenscreen(input_path, output_path)

    elif args.command == "compose":
        composite_character(
            Path(args.character), Path(args.background), Path(args.output),
            position=args.position, scale=args.scale,
            offset_x=args.offset_x, offset_y=args.offset_y,
        )

    elif args.command == "batch":
        batch_composite(
            Path(args.character), Path(args.backgrounds), Path(args.output),
            position=args.position, scale=args.scale,
            offset_x=args.offset_x, offset_y=args.offset_y,
        )

    elif args.command == "video":
        composite_video(
            Path(args.video), Path(args.background), Path(args.output),
            similarity=args.similarity, blend=args.blend,
            chroma_color=args.chroma_color,
        )

    elif args.command == "video-layers":
        background, layers, scene_audio = load_scene_config(Path(args.scene))
        audio_mode = args.audio if args.audio else scene_audio
        composite_video_layers(background, layers, Path(args.output), audio_mode=audio_mode)


if __name__ == "__main__":
    main()
