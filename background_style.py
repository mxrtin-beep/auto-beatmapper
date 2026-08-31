#!/usr/bin/env python3
"""
Background image support: attaching one to a beatmap, and deriving its
combo colors from the image itself instead of the pipeline's own fixed
orange/blue/yellow default (see beatmap_utils.default_metadata).

Usage (also exercised by main.py/gui.py automatically when --background
is passed):
    python3 background_style.py cover.jpg out/Song\ \[Insane\].osu
"""

from __future__ import annotations

import argparse
import os

from beatmap_utils import Beatmap, read_osu, write_osu

BACKGROUND_EVENTS_MARKER = "//Background and Video events"


def extract_combo_colors(image_path: str, num_colors: int = 4) -> list[tuple[int, int, int]]:
    """The `num_colors` most common colors in the image, most common first.

    Downscaled first (color counting doesn't need full resolution, and a
    huge cover image would otherwise make this slow) and quantized with
    Pillow's own median-cut palette, which is what actually groups close-
    but-not-identical pixels (JPEG noise, gradients) into one shared
    color instead of the true dominant colors getting outvoted by
    thousands of near-duplicates -- a plain histogram over raw pixels
    would badly under-count exactly the flat, saturated color a combo
    color should be pulled from.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img.thumbnail((200, 200))
    paletted = img.quantize(colors=max(1, num_colors), method=Image.Quantize.MEDIANCUT)
    palette = paletted.getpalette()
    counts = sorted(paletted.getcolors(), key=lambda c: c[0], reverse=True)

    colors: list[tuple[int, int, int]] = []
    for _count, idx in counts[:num_colors]:
        r, g, b = palette[idx * 3:idx * 3 + 3]
        colors.append((r, g, b))
    return colors


def apply_background(osu_path: str, background_filename: str,
                      colours: list[tuple[int, int, int]] | None = None) -> None:
    """Mutate an already-written .osu file in place: declare
    `background_filename` (expected to sit right next to the .osu, same as
    the audio file) as its background image, and, if given, replace its
    combo colors with `colours`.

    `background_filename` should be a bare filename, not a path -- it's
    written into the beatmap exactly as osu! expects to find it alongside
    the .osu/audio inside the same folder or .osz.
    """
    bm = read_osu(osu_path)
    _set_background(bm, background_filename)
    if colours:
        bm.colours = [f"Combo{i + 1} : {r},{g},{b}" for i, (r, g, b) in enumerate(colours)]
    write_osu(bm, osu_path)


def _set_background(bm: Beatmap, background_filename: str) -> None:
    bg_line = f'0,0,"{background_filename}",0,0'
    # Replace rather than duplicate, in case this is ever applied twice to
    # the same file (e.g. a restyle-only rerun).
    bm.events = [line for line in bm.events
                 if not (line.startswith("0,0,\"") or line.startswith("0,0,'"))]
    if BACKGROUND_EVENTS_MARKER in bm.events:
        idx = bm.events.index(BACKGROUND_EVENTS_MARKER) + 1
        bm.events.insert(idx, bg_line)
    else:
        bm.events.insert(0, bg_line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach a background image (and its combo colors) to a beatmap.")
    parser.add_argument("image", help="Path to the background image.")
    parser.add_argument("osu_files", nargs="+", help="One or more .osu files to attach it to.")
    parser.add_argument("--num-colors", type=int, default=4, help="How many combo colors to derive (default 4).")
    args = parser.parse_args()

    colours = extract_combo_colors(args.image, args.num_colors)
    background_filename = os.path.basename(args.image)
    for osu_path in args.osu_files:
        apply_background(osu_path, background_filename, colours)
        print(f"Applied background + combo colors to {osu_path}")


if __name__ == "__main__":
    main()
