#!/usr/bin/env python3
"""
V2 slider merger — Stage 2 for the Base Map v2 pathway.

Takes a Base Map v2 .osu (see generate_base_beatmap_v2.py — plain circles
only, an independent pathway not fed by/into the main add_variety.py/
apply_style.py pipeline) and merges some adjacent circles into sliders,
governed by:

  * --chain-probability (0-1): how often an otherwise-eligible run of
    adjacent circles actually becomes a slider at all, versus staying
    plain circles — 0 never merges anything, 1 merges every eligible run.
    Default 0.8, the same balance add_variety.py's own --chain-probability
    defaults to.
  * --slider-length-bias (0-1): of whichever runs --chain-probability
    already decided to merge, which chain length (2, 3, or 4 nodes) a
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
through empty space, not a phrase.

Positioning — distance-snap spacing between objects (via apply_style.py's
own --spacing, forwarded straight through here, the same knob that
already produces accurately-spaced, error-free output for the main
pipeline), playfield bounds, and avoiding overlap — is handled entirely by
re-running apply_style.py against the merged result: it already solves
exactly that (the same machinery the main pipeline relies on), so there's
no reason to duplicate any of it here. This script's only job is deciding
*which* circles become one slider; apply_style.py decides where everything
actually goes.

Hitsounds are assigned here too (add_variety.py's own assign_hitsounds,
reused directly) — without this, every object stays on the default
"normal" sample, which reads as a broken/incomplete map to any checker.

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

import numpy as np

import apply_style
import make_easy
from add_variety import (assign_hitsounds, chain_len_weights, compute_energy_curve, find_repeating_measure_map,
                          is_on_downbeat, make_bounce_slider, make_energy_lookup, make_slider_chain)
from beatmap_utils import HitObject, read_osu, write_osu
from make_easy import recompute_combos

# How much --spacing is scaled down per tier below Insane -- osu!'s own
# distance-snap formula is CircleSize-agnostic, but a lower tier's much
# bigger circles (see make_easy.TIER_TARGET) need to travel *less* far to
# still read as a comfortable, correctly-spaced jump; reusing Insane's own
# jump distance on Easy's big circles is what was reading as "too spacey"
# and, combined with the bigger circles eating into the playfield margin
# a curve's bow bulges into, made an occasional off-screen slider likelier
# too. Insane keeps the user's own --spacing exactly; each tier down
# scales it further, matching how much smaller a jump reads as comfortable
# once the circles themselves are bigger.
TIER_SPACING_SCALE: dict[str, float] = {"insane": 1.0, "hard": 0.85, "normal": 0.7, "easy": 0.55}

# Of an otherwise-eligible, evenly-spaced 3-4 circle run that's becoming a
# slider, how often it becomes a repeating/"bounce" slider (one path back
# and forth) instead of a waypoint chain visiting each point in turn — see
# merge_into_sliders. Not exposed on the CLI: it's a stylistic mix within
# "becomes a slider at all" (--chain-probability's own job), not a
# separate knob worth surfacing.
BOUNCE_PROBABILITY = 0.3


def merge_into_sliders(circles: list[HitObject], beat_length_ms: float, slider_multiplier: float,
                        rng: random.Random, slider_length_bias: float, chain_probability: float = 0.8,
                        max_gap_beats: float = 1.0, offset_ms: float | None = None,
                        measure_repeat_map: dict[int, int] | None = None) -> list[HitObject]:
    """Walk `circles` (sorted by time) and replace some adjacent runs of
    2-4 with a single chain slider spanning them, via add_variety.py's own
    make_slider_chain — identical mechanics to add_variety.py's normal-
    section merging, just without its energy-category gating (Base Map v2
    has no such categories; any adjacent, close-enough run is eligible).
    A chain can only include circles whose *own* consecutive gaps are all
    within `max_gap_beats` of each other — extending a chain across
    whatever real silent gap Base Map v2 already left between two circles
    would turn a deliberate silence into a slider dragging through it.

    A run of 3+ circles that are *evenly* spaced (equal consecutive gaps —
    exactly what generate_base_beatmap_v2.py's own quarter/eighth-beat
    subdivisions and its embellishment chains both produce) sometimes
    becomes a repeating/"bounce" slider instead (add_variety.py's own
    make_bounce_slider): one back-and-forth path between the run's first
    two points, repeated enough times to cover the same span, rather than
    a waypoint chain visiting every point in turn. A merely-adjacent but
    unevenly-spaced run never qualifies — a bounce slider's repeats are
    all the same duration by construction, so it can only stand in for a
    run that's genuinely on one rhythmic grid already.

    `measure_repeat_map` (see add_variety.py's find_repeating_measure_map;
    pass `offset_ms` alongside it) makes a repeated measure's layout copy
    an earlier occurrence's own choices — circle vs. chain, chain length,
    plain chain vs. bounce — the same way assign_hitsounds already copies
    a repeated measure's accent pattern, rather than every occurrence
    rolling independently. Replay only ever reuses a *decision*, never
    forces one that doesn't fit this occurrence's own actual data (not
    enough eligible circles here to reach the same chain length, or this
    occurrence isn't evenly spaced the way the original was for a bounce)
    — a decision that doesn't fit is decided fresh instead, same as if no
    repeat map were given at all. Pass `measure_repeat_map=None` (the
    default) for the original, fully independent-per-run behavior.
    """
    weights = chain_len_weights(slider_length_bias)
    max_gap_ms = beat_length_ms * max_gap_beats + 1.0
    result: list[HitObject] = []
    replayed_decisions: dict[tuple[int, int], tuple] = {}
    i, n = 0, len(circles)
    while i < n:
        cur = circles[i]
        max_chain = 1
        while (i + max_chain < n and max_chain < 4
               and circles[i + max_chain].time - circles[i + max_chain - 1].time <= max_gap_ms):
            max_chain += 1

        replay_key = None
        replay = None
        if measure_repeat_map is not None and offset_ms is not None:
            # Eighth-beat (32nd-note) resolution -- the finest grid a
            # circle can actually start on (see generate_base_beatmap_v2
            # .py's own climax tier) -- not whole-beat: a chain can start
            # on any quarter/eighth-beat offset within its measure, and
            # rounding to the nearest whole beat would collide several
            # different starting circles from the same busy beat onto one
            # key, corrupting replay for all of them.
            eighth_idx = round((cur.time - offset_ms) / beat_length_ms * 8.0)
            measure_idx, slot_in_measure = divmod(eighth_idx, 32)
            canonical_measure = measure_repeat_map.get(measure_idx, measure_idx)
            if canonical_measure != measure_idx:
                replay = replayed_decisions.get((canonical_measure, slot_in_measure))
            replay_key = (measure_idx, slot_in_measure)

        if replay is not None and replay[0] == "chain" and replay[1] <= max_chain:
            chain_len, want_bounce = replay[1], replay[2]
            can_chain = True
        elif replay is not None and replay[0] == "circle":
            chain_len, want_bounce = None, False
            can_chain = False
        else:
            can_chain = max_chain >= 2 and rng.random() < chain_probability
            chain_len = (rng.choices([2, 3, 4][:max_chain - 1], weights=weights[:max_chain - 1])[0]
                         if can_chain else None)
            want_bounce = can_chain and rng.random() < BOUNCE_PROBABILITY

        if can_chain:
            nodes = circles[i:i + chain_len]
            gaps = [nodes[k + 1].time - nodes[k].time for k in range(len(nodes) - 1)]
            evenly_spaced = chain_len >= 3 and (max(gaps) - min(gaps)) < 1.0
            if want_bounce and evenly_spaced:
                result.append(make_bounce_slider(nodes[0], nodes[1], beat_length_ms, slider_multiplier,
                                                   num_bounces=chain_len - 1, one_way_ms=gaps[0]))
            else:
                result.append(make_slider_chain(nodes, beat_length_ms, slider_multiplier))
            if replay_key is not None:
                replayed_decisions[replay_key] = ("chain", chain_len, want_bounce and evenly_spaced)
            i += chain_len
        else:
            result.append(cur)
            if replay_key is not None:
                replayed_decisions[replay_key] = ("circle",)
            i += 1
    return result


def thin_for_tier(objects: list[HitObject], tier: str, beat_length_ms: float, offset_ms: float,
                   measure_length_ms: float, energy_at, rng: random.Random) -> list[HitObject]:
    """Delete objects for an easier tier, using the exact same category-
    based rule make_easy.py's own thin_by_deletion applies to the main
    pipeline's spread (TIER_DELETE_PROBABILITY: quarter/eighth-beat
    circles thinned hardest, half-beat less, whole-beat rarely, sliders
    their own rate) plus its deterministic enforce_min_gap pass -- just
    without either one's cascading re-snap machinery, since here every
    tier gets its own fresh apply_style.py pass afterward (see main()),
    which repositions every survivor from scratch anyway. "insane" always
    returns `objects` untouched (TIER_DELETE_PROBABILITY["insane"] is
    None), the same as make_easy.py's own convention.
    """
    thinning = make_easy.TIER_DELETE_PROBABILITY[tier]
    if thinning is None:
        return list(objects)

    measure_buckets = apply_style.compute_measure_energy_buckets(energy_at, offset_ms, measure_length_ms,
                                                                   objects[-1].time)
    eligible = (make_easy.find_repetitive_measures(measure_buckets)
                if thinning["scope"] == "repetitive" else None)

    def measure_of(t: float) -> int:
        return int((t - offset_ms) // measure_length_ms)

    def category_of(o: HitObject) -> str:
        return "slider" if o.is_slider else make_easy.classify_beat_position(o.time, offset_ms, beat_length_ms)

    n = len(objects)
    kept: list[HitObject] = []
    for i, obj in enumerate(objects):
        can_delete = (0 < i < n - 1 and (eligible is None or measure_of(obj.time) in eligible)
                      and not is_on_downbeat(obj.time, offset_ms, measure_length_ms))
        if can_delete and rng.random() < thinning[category_of(obj)]:
            continue
        kept.append(obj)

    # Deterministic minimum-gap pass, same idea as make_easy.py's own
    # enforce_min_gap -- the probabilities above can, by chance, still
    # leave two survivors closer than this tier should ever allow.
    min_gap_ms = beat_length_ms * thinning["min_gap_beats"] - 1.0
    m = len(kept)
    final: list[HitObject] = []
    for i, obj in enumerate(kept):
        if (final and 0 < i < m - 1 and (obj.time - final[-1].time) < min_gap_ms
                and not is_on_downbeat(obj.time, offset_ms, measure_length_ms)):
            continue
        final.append(obj)
    return final


def build_tier(tier: str, objects: list[HitObject], bm, args, rng: random.Random, energy_at,
               output_path: str, version: str, circle_size: float, approach_rate: float,
               hp_drain: float | None, overall_difficulty: float | None, spacing: float) -> None:
    """Thin `objects` for `tier` (a no-op for "insane"), then position the
    survivors with their own fresh apply_style.py pass -- rather than
    deriving from an already-styled Insane map the way make_easy.py's own
    thin-and-re-snap approach does -- so every tier's jump distances and
    curve shapes are apply_style.py's own real output for that tier's own
    (scaled-down, see TIER_SPACING_SCALE) --spacing, not a rescaled copy
    of Insane's. That also means a lower tier's much bigger circles never
    inherit Insane's tighter spacing, which was reading as "too spacey" on
    Easy, and gives every tier the exact same off-screen-safe positioning
    apply_style.py already guarantees for Insane, instead of only Insane
    actually running it.
    """
    beat_length_ms = bm.beat_length
    slider_multiplier = bm.slider_multiplier
    offset_ms = bm.offset
    measure_length_ms = beat_length_ms * 4.0

    thinned = thin_for_tier(objects, tier, beat_length_ms, offset_ms, measure_length_ms, energy_at, rng)
    if len(thinned) < 2:
        raise RuntimeError(f"Tier '{tier}' thinned down to fewer than two objects -- lower its deletion odds.")
    recompute_combos(thinned, offset_ms, measure_length_ms)

    obj_energy = np.array([energy_at(o.time) for o in thinned])
    q_high = float(np.quantile(obj_energy, 0.75))
    q_climax = float(np.quantile(obj_energy, 0.92))
    # measure_repeat_map lets a verse/chorus's second pass copy its first
    # pass's own accent pattern (see add_variety.py's find_repeating_
    # measure_map/assign_hitsounds) rather than re-deriving independently
    # -- computed fresh per tier since thinning changes which objects (and
    # so which beats) actually exist to copy from.
    measure_buckets = apply_style.compute_measure_energy_buckets(energy_at, offset_ms, measure_length_ms,
                                                                   thinned[-1].time)
    measure_repeat_map = find_repeating_measure_map(measure_buckets)
    assign_hitsounds(thinned, energy_at, offset_ms, measure_length_ms, q_high, q_climax,
                      measure_repeat_map=measure_repeat_map)

    tier_bm = read_osu(args.beatmap)
    tier_bm.hit_objects = thinned
    tier_bm.metadata["Version"] = version
    tier_merged_path = output_path + ".merged.osu"
    os.makedirs(os.path.dirname(os.path.abspath(tier_merged_path)) or ".", exist_ok=True)
    write_osu(tier_bm, tier_merged_path)

    old_argv = sys.argv
    try:
        sys.argv = ["apply_style.py", tier_merged_path, "--output", output_path, "--audio", args.audio,
                    "--version", version, "--seed", str(rng.randrange(2**32)),
                    "--curviness", str(args.curviness), "--spacing", str(spacing),
                    # apply_style.py's own default (0.1) treats most fast
                    # runs as ordinary flow, only occasionally reading as a
                    # deliberate stream -- appropriate for the main
                    # pipeline, where a fast run can show up incidentally.
                    # Here a fast (quarter-beat-or-closer) run of 4+ circles
                    # is never incidental: generate_base_beatmap_v2.py only
                    # ever produces one via its own climax/intense tiers or
                    # a deliberately sparse embellishment chain (see
                    # add_embellishment_chains) -- always deliberate, so it
                    # should always read as one gesture (and, combined with
                    # --stack-probability's own 1.0 default, always stack
                    # on the exact same spot).
                    "--stream-frequency", "1.0"]
        apply_style.main()
    finally:
        sys.argv = old_argv
        os.remove(tier_merged_path)

    styled_bm = read_osu(output_path)
    styled_bm.difficulty["CircleSize"] = f"{circle_size:.1f}"
    styled_bm.difficulty["ApproachRate"] = f"{approach_rate:.1f}"
    if hp_drain is not None:
        styled_bm.difficulty["HPDrainRate"] = f"{hp_drain:.1f}"
    if overall_difficulty is not None:
        styled_bm.difficulty["OverallDifficulty"] = f"{overall_difficulty:.1f}"
    write_osu(styled_bm, output_path)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge circles into sliders for a Base Map v2 beatmap, "
                                                   "then position everything via apply_style.py.")
    parser.add_argument("beatmap", help="Path to the Base Map v2 .osu file (from generate_base_beatmap_v2.py).")
    parser.add_argument("audio", help="Path to the same song's audio file (for apply_style.py's energy-aware patterns).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", default="Auto Base v2 (Styled)", help="Difficulty/version name to write into the final, styled map.")
    parser.add_argument("--chain-probability", type=float, default=0.8,
                         help="How often an otherwise-eligible run of adjacent circles actually "
                              "becomes a slider at all, 0-1 (default 0.8). 0 = always plain circles, "
                              "1 = every eligible run becomes a slider.")
    parser.add_argument("--slider-length-bias", type=float, default=0.5,
                         help="Of whichever runs --chain-probability already decided to merge: which "
                              "chain length tends to get picked, 0-1 (default 0.5). 0 skews toward "
                              "more/shorter sliders, 1 toward fewer/longer ones.")
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
    parser.add_argument("--circle-size", type=float, default=4.5,
                         help="CircleSize written into the final output (default 4.5).")
    parser.add_argument("--approach-rate", type=float, default=8.4,
                         help="ApproachRate written into the final output (default 8.4).")
    parser.add_argument("--hard-output", default=None,
                         help="Also generate a Hard tier here -- thinned from the same merge and given "
                              "its own apply_style.py pass (its own scaled-down --spacing and Difficulty "
                              "settings; see TIER_SPACING_SCALE/make_easy.TIER_TARGET), not derived from "
                              "--output the way make_easy.py's own spread works. Omit to skip it.")
    parser.add_argument("--normal-output", default=None, help="Same as --hard-output, for Normal.")
    parser.add_argument("--easy-output", default=None, help="Same as --hard-output, for Easy.")
    parser.add_argument("--reuse-layout", dest="reuse_layout", action="store_true", default=True,
                         help="When a measure repeats an earlier one (the same windowed measure-loudness "
                              "pattern recurring, see add_variety.py's find_repeating_measure_map), copy "
                              "that earlier measure's own circle/chain/bounce layout decisions instead of "
                              "rolling independently -- on by default. A decision that doesn't fit this "
                              "occurrence's own actual data (not enough eligible circles here, not evenly "
                              "spaced enough for a bounce) is still decided fresh either way.")
    parser.add_argument("--no-reuse-layout", dest="reuse_layout", action="store_false",
                         help="Revert to the original behavior: every run's circle/chain/bounce choice is "
                              "rolled independently, with no attempt to reuse an earlier repeated "
                              "measure's own layout.")
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

    # Energy analysis moved up here (rather than only just before hitsounds,
    # its original use) so merge_into_sliders can also key its own
    # --reuse-layout decisions off the same measure_repeat_map assign_
    # hitsounds uses below -- both want "does this measure repeat an
    # earlier one", computed the same way, so it's derived once and shared.
    print("Analyzing song energy...")
    times_ms, energy = compute_energy_curve(args.audio)
    energy_at = make_energy_lookup(times_ms, energy)
    measure_length_ms = beat_length_ms * 4.0
    measure_buckets = apply_style.compute_measure_energy_buckets(energy_at, bm.offset, measure_length_ms,
                                                                   circles[-1].time)
    measure_repeat_map = find_repeating_measure_map(measure_buckets) if args.reuse_layout else None

    merged = merge_into_sliders(circles, beat_length_ms, slider_multiplier, rng, args.slider_length_bias,
                                 chain_probability=args.chain_probability, max_gap_beats=args.max_gap_beats,
                                 offset_ms=bm.offset, measure_repeat_map=measure_repeat_map)
    merged.sort(key=lambda h: h.time)
    n_sliders = sum(1 for o in merged if o.is_slider)
    print(f"{len(circles)} circles -> {len(merged)} objects ({n_sliders} sliders)")

    # Combos: generate_base_beatmap_v2.py stamps is_new_combo by array
    # position (every 8th of its own *circles*), which merging silently
    # invalidates -- several circles collapsing into one slider object
    # shifts every position after them, so those flags no longer land
    # anywhere near a real 8-object boundary and a combo can run well past
    # 8. Recomputed here from scratch instead (the same downbeat-aligned,
    # 8-object-capped logic make_easy.py's own tier derivation uses).
    measure_length_ms = beat_length_ms * 4.0
    recompute_combos(merged, bm.offset, measure_length_ms)

    # Hitsounds: reuses add_variety.py's own accenting logic (bigger
    # accents on louder/downbeat moments, a forced minimum so no long
    # stretch reads as "no hitsounds" to a checker) rather than leaving
    # every object on the default plain sample generate_base_beatmap_v2.py
    # itself never assigns anything past. Hitsound repeat-copying is
    # always on regardless of --reuse-layout (that flag is about the
    # circle/chain/bounce layout choice specifically) -- recomputed against
    # `merged`'s own true end time rather than reusing the layout pass's
    # measure_buckets (based on the pre-merge circles), since merging can
    # shift exactly where a slider's own span ends.
    obj_energy = np.array([energy_at(o.time) for o in merged])
    q_high = float(np.quantile(obj_energy, 0.75))
    q_climax = float(np.quantile(obj_energy, 0.92))
    measure_buckets = apply_style.compute_measure_energy_buckets(energy_at, bm.offset, measure_length_ms,
                                                                   merged[-1].time)
    measure_repeat_map = find_repeating_measure_map(measure_buckets)
    assign_hitsounds(merged, energy_at, bm.offset, measure_length_ms, q_high, q_climax,
                      measure_repeat_map=measure_repeat_map)

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

    if not keep_merged:
        os.remove(merged_path)

    # Each tier -- Insane (--output, always) plus whichever of Hard/Normal/
    # Easy were asked for -- gets its own thin-then-apply_style.py pass
    # (see build_tier): every tier's positions, jump distances and curve
    # shapes are apply_style.py's own real output for that tier's own
    # (scaled-down below Insane) --spacing, rather than Insane's own
    # already-styled positions rescaled/resnapped the way make_easy.py's
    # spread works for the main pipeline. That gives every tier the exact
    # same off-screen-safe positioning apply_style.py already guarantees,
    # and never leaves a bigger-circled lower tier stuck with Insane's own
    # (comparatively tight) jump distance.
    tier_outputs = [("insane", args.output, args.version, args.circle_size, args.approach_rate, None, None)]
    for tier, tier_output in (("hard", args.hard_output), ("normal", args.normal_output), ("easy", args.easy_output)):
        if tier_output is None:
            continue
        target = make_easy.TIER_TARGET[tier]
        tier_outputs.append((tier, tier_output, tier.capitalize(), target["CircleSize"], target["ApproachRate"],
                              target["HPDrainRate"], target["OverallDifficulty"]))

    for tier, tier_output, version, circle_size, approach_rate, hp_drain, overall_difficulty in tier_outputs:
        spacing = args.spacing * TIER_SPACING_SCALE[tier]
        build_tier(tier, merged, bm, args, rng, energy_at, tier_output, version, circle_size, approach_rate,
                   hp_drain, overall_difficulty, spacing)


if __name__ == "__main__":
    main()
