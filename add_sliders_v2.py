#!/usr/bin/env python3
"""
V2 slider merger — Stage 2 for the Base Map v2 pathway.

Takes a Base Map v2 .osu (see generate_base_beatmap_v2.py — plain circles
only, an independent pathway not fed by/into the main add_variety.py/
apply_style.py pipeline) and merges some adjacent circles into sliders,
governed by exactly two knobs:

  * --slider-length-bias (0-1): which chain length (2, 3, or 4 nodes) a
    merge tends to pick — 0 skews toward short/choppy sliders, 1 toward
    long ones, 0.5 the same balanced default add_variety.py's own
    --slider-length-bias uses (chain_len_weights, reused here directly —
    every length stays possible at every setting, only the balance
    shifts).
  * --curviness (0-1): forwarded straight through to apply_style.py,
    which does the actual positioning.

Only ever merges genuinely adjacent circles with no more than
--max-gap-beats between any pair in the chain (default 1.0 beat) — a
chain reaching across a real silent gap would read as a slider dragging
through empty space, not a phrase. A fixed 80% chance any otherwise-
eligible run actually becomes a slider (not exposed as a setting) keeps
some plain circles around for rhythmic contrast, the same reasoning
add_variety.py's own --chain-probability exists for.

Positioning — distance-snap spacing between objects, playfield bounds, and
avoiding overlap — is handled entirely by re-running apply_style.py
against the merged result: it already solves exactly that (the same
machinery the main pipeline relies on), so there's no reason to duplicate
any of it here. This script's only job is deciding *which* circles become
one slider; apply_style.py decides where everything actually goes.

Two intermediate stages, mirroring the main pipeline's own Base/Variety/
Styled naming:

  * Circles  — generate_base_beatmap_v2.py's own output, untouched.
  * Sliders  — the merge this module does, on its own: circles combined
               into sliders, but still sitting at generate_base_beatmap_v2
               .py's placeholder positions (--merged-output, if given —
               it's an internal working file otherwise, deleted once
               apply_style.py is done reading it).
  * Styled   — this module's actual --output: the same objects, positioned
               for real by apply_style.py.

Usage:
    python3 add_sliders_v2.py base_v2.osu song.mp3 --output out/song_v2_styled.osu \
        --slider-length-bias 0.5 --curviness 0.5
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import apply_style
from add_variety import chain_len_weights, make_slider_chain
from beatmap_utils import HitObject, read_osu, write_osu

CHAIN_PROBABILITY = 0.8  # fixed, not exposed as a setting -- keeps some plain circles for contrast


def merge_into_sliders(circles: list[HitObject], beat_length_ms: float, slider_multiplier: float,
                        rng: random.Random, slider_length_bias: float,
                        max_gap_beats: float = 1.0) -> list[HitObject]:
    """Walk `circles` (sorted by time) and replace some adjacent runs of
    2-4 with a single chain slider spanning them, via add_variety.py's own
    make_slider_chain — identical mechanics to add_variety.py's normal-
    section merging, just without its energy-category gating (Base Map v2
    has no such categories; any adjacent, close-enough run is eligible).
    A chain can only include circles whose *own* consecutive gaps are all
    within `max_gap_beats` of each other — extending a chain across
    whatever real silent gap Base Map v2 already left between two circles
    would turn a deliberate silence into a slider dragging through it.
    """
    weights = chain_len_weights(slider_length_bias)
    max_gap_ms = beat_length_ms * max_gap_beats + 1.0
    result: list[HitObject] = []
    i, n = 0, len(circles)
    while i < n:
        cur = circles[i]
        max_chain = 1
        while (i + max_chain < n and max_chain < 4
               and circles[i + max_chain].time - circles[i + max_chain - 1].time <= max_gap_ms):
            max_chain += 1
        can_chain = max_chain >= 2 and rng.random() < CHAIN_PROBABILITY
        if can_chain:
            chain_len = rng.choices([2, 3, 4][:max_chain - 1], weights=weights[:max_chain - 1])[0]
            nodes = circles[i:i + chain_len]
            result.append(make_slider_chain(nodes, beat_length_ms, slider_multiplier))
            i += chain_len
        else:
            result.append(cur)
            i += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge circles into sliders for a Base Map v2 beatmap, "
                                                   "then position everything via apply_style.py.")
    parser.add_argument("beatmap", help="Path to the Base Map v2 .osu file (from generate_base_beatmap_v2.py).")
    parser.add_argument("audio", help="Path to the same song's audio file (for apply_style.py's energy-aware patterns).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", default="Auto Base v2 (Styled)", help="Difficulty/version name to write into the final, styled map.")
    parser.add_argument("--slider-length-bias", type=float, default=0.5,
                         help="Which chain length a merge tends to pick, 0-1 (default 0.5). 0 skews "
                              "toward more/shorter sliders, 1 toward fewer/longer ones.")
    parser.add_argument("--curviness", type=float, default=0.5,
                         help="Forwarded to apply_style.py's --curviness (0-1, how curvy sliders feel).")
    parser.add_argument("--spacing", type=float, default=1.8,
                         help="Forwarded to apply_style.py's --spacing (jump/spacing distance multiplier).")
    parser.add_argument("--max-gap-beats", type=float, default=1.0,
                         help="Circles more than this many beats apart are never merged into the same "
                              "chain (default 1.0) -- keeps a slider from dragging across a real silent gap.")
    parser.add_argument("--merged-output", default=None,
                         help="Also write the merged-but-unstyled intermediate .osu here (circles "
                              "already combined into sliders, but still at generate_base_beatmap_v2.py's "
                              "placeholder positions -- apply_style.py hasn't run yet). Omit to skip it; "
                              "it's an internal working file either way, used as apply_style.py's own "
                              "input.")
    parser.add_argument("--merged-version", default="Auto Base v2 (Sliders)",
                         help="Difficulty/version name written into --merged-output, if given.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed. Omit for a different result every run; pass a fixed value "
                              "(printed on every run) to reproduce it later.")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.SystemRandom().randrange(2**32)
    rng = random.Random(args.seed)
    print(f"Using seed: {args.seed}")

    bm = read_osu(args.beatmap)
    beat_length_ms = bm.beat_length
    slider_multiplier = bm.slider_multiplier

    circles = sorted([h for h in bm.hit_objects if not h.is_slider], key=lambda h: h.time)
    if len(circles) < 2:
        raise RuntimeError("Base Map v2 beatmap needs at least two circles to merge into sliders.")

    merged = merge_into_sliders(circles, beat_length_ms, slider_multiplier, rng,
                                 args.slider_length_bias, max_gap_beats=args.max_gap_beats)
    merged.sort(key=lambda h: h.time)
    n_sliders = sum(1 for o in merged if o.is_slider)
    print(f"{len(circles)} circles -> {len(merged)} objects ({n_sliders} sliders)")

    bm.hit_objects = merged
    # Written under --merged-output's own name (and kept) if given -- an
    # intermediate someone might genuinely want to inspect, to see the
    # structural merge on its own before apply_style.py's positioning is
    # even in the picture (still at generate_base_beatmap_v2.py's own
    # placeholder positions here, not real ones yet). Otherwise it's
    # purely an internal working file, written under a hidden name next
    # to --output and deleted once apply_style.py is done reading it.
    keep_merged = args.merged_output is not None
    merged_path = args.merged_output if keep_merged else args.output + ".merged.osu"
    bm.metadata["Version"] = args.merged_version if keep_merged else bm.metadata.get("Version", "")
    os.makedirs(os.path.dirname(os.path.abspath(merged_path)) or ".", exist_ok=True)
    write_osu(bm, merged_path)
    if keep_merged:
        print(f"Wrote {merged_path}")

    # apply_style.py does the actual positioning -- distance-snap spacing
    # between every pair of objects, playfield-bounds clamping, and (for
    # any leftover fast circle runs the merge above didn't happen to
    # absorb into a slider) its own stream/stack handling. All of that is
    # already solved there; re-running it against the merged result is
    # simpler and more robust than reimplementing any part of it here.
    old_argv = sys.argv
    try:
        sys.argv = ["apply_style.py", merged_path, "--output", args.output, "--audio", args.audio,
                    "--version", args.version, "--seed", str(args.seed),
                    "--curviness", str(args.curviness), "--spacing", str(args.spacing)]
        apply_style.main()
    finally:
        sys.argv = old_argv
        if not keep_merged:
            os.remove(merged_path)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
