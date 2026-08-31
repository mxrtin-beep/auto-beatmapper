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
import colorsys
import os

from beatmap_utils import Beatmap, read_osu, write_osu

BACKGROUND_EVENTS_MARKER = "//Background and Video events"

# Combo colors sit on top of gameplay, over whatever the background happens
# to be behind them -- a color pulled straight from the image can be a
# near-black shadow or a near-white highlight, either of which reads as
# barely-there against a busy background (worse yet, a near-black one is
# also just hard to see against osu!'s own default near-black playfield
# dim). Never used as-is; see _visible_variant.
_LUMINANCE_BLACK_MAX = 60
_LUMINANCE_WHITE_MIN = 210
_MIN_LIGHTNESS = 0.32
_MAX_LIGHTNESS = 0.72
_MIN_SATURATION = 0.45

MIN_COMBO_COLORS = 3
MAX_COMBO_COLORS = 4


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _is_black_or_white(rgb: tuple[int, int, int]) -> bool:
    lum = _luminance(rgb)
    return lum <= _LUMINANCE_BLACK_MAX or lum >= _LUMINANCE_WHITE_MIN


def _hue(rgb: tuple[int, int, int]) -> float:
    r, g, b = (c / 255.0 for c in rgb)
    h, _l, _s = colorsys.rgb_to_hls(r, g, b)
    return h


def _visible_variant(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """`rgb`'s own hue, pulled into a lightness/saturation band that reads
    clearly as a combo color -- a muddy, dim brown keeps being recognizably
    brown, it's just no longer *too* dark or *too* washed out to see."""
    r, g, b = (c / 255.0 for c in rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(max(l, _MIN_LIGHTNESS), _MAX_LIGHTNESS)
    s = max(s, _MIN_SATURATION)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return tuple(int(round(c * 255)) for c in (r2, g2, b2))


def _hue_from_hls(h: float, l: float, s: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
    return tuple(int(round(c * 255)) for c in (r, g, b))


def _hue_distance(a: float, b: float) -> float:
    """Circular distance between two [0, 1) hues (hue 0.98 and 0.02 are
    close, not far, since hue wraps around)."""
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def extract_combo_colors(image_path: str, num_colors: int = MAX_COMBO_COLORS) -> list[tuple[int, int, int]]:
    """3-4 combo colors (never fewer than `MIN_COMBO_COLORS`, never more
    than `min(num_colors, MAX_COMBO_COLORS)`) derived from the image's own
    most common colors, most common first.

    Never black or white (see _is_black_or_white) and never dark/washed-out
    either -- every color is pulled into a visibly bright, saturated
    version of its own hue (_visible_variant) before it's used, and hues
    too close to an already-picked one are skipped so the set doesn't come
    out as several near-identical shades. If the image doesn't offer
    enough distinct, non-black/white colors this way (a near-monochrome
    photo, or one that's mostly black/white to begin with), the remaining
    slots are filled with complementary colors -- opposite (and, for a
    4th, a 120-degree split) on the color wheel from whatever *was* found,
    the same "brown + yellow found in the image -> add a blue/purple that
    actually contrasts with both" pairing a human picking accent colors
    would reach for.
    """
    from PIL import Image

    target = max(MIN_COMBO_COLORS, min(num_colors, MAX_COMBO_COLORS))

    img = Image.open(image_path).convert("RGB")
    img.thumbnail((200, 200))
    # Quantized to well more than `target` colors -- most of that palette
    # gets thrown away below (near-black/white, or too close a hue to one
    # already picked), so asking for only `target` up front would often
    # leave nothing left to fall back on besides synthetic complementaries,
    # even when the image genuinely has enough real distinct colors in it.
    palette_size = max(4 * target, 16)
    paletted = img.quantize(colors=palette_size, method=Image.Quantize.MEDIANCUT)
    palette = paletted.getpalette()
    counts = sorted(paletted.getcolors(), key=lambda c: c[0], reverse=True)

    picked: list[tuple[int, int, int]] = []
    picked_hues: list[float] = []
    for _count, idx in counts:
        rgb = tuple(palette[idx * 3:idx * 3 + 3])
        if _is_black_or_white(rgb):
            continue
        hue = _hue(rgb)
        if any(_hue_distance(hue, h) < 0.035 for h in picked_hues):
            continue  # too similar to a hue already picked -- skip, don't just re-shade it
        picked.append(_visible_variant(rgb))
        picked_hues.append(hue)
        if len(picked) >= target:
            break

    colors = list(picked)

    # Not enough distinct colors survived filtering -- fill out with
    # complementary hues instead of returning fewer than MIN_COMBO_COLORS
    # (or, with nothing at all, a fixed vivid fallback set rather than an
    # empty list).
    if not colors:
        base_hues = [0.11, 0.55]  # a warm amber and a contrasting blue
    else:
        base_hues = picked_hues
    complement_offsets = [0.5, 1 / 3, 2 / 3]  # opposite, then a 120-degree split either way
    offset_i = 0
    attempts = 0
    while len(colors) < target and attempts < 50:
        attempts += 1
        base_hue = base_hues[(len(colors) - len(picked)) % len(base_hues)] if base_hues else 0.11
        new_hue = (base_hue + complement_offsets[offset_i % len(complement_offsets)]) % 1.0
        offset_i += 1
        if any(_hue_distance(new_hue, _hue(c)) < 0.06 for c in colors):
            continue
        colors.append(_hue_from_hls(new_hue, 0.5, 0.65))
    while len(colors) < target:  # exhausted every offset combination -- pad with a fixed vivid color
        colors.append(_hue_from_hls(0.11, 0.5, 0.65))

    return colors[:target]


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
