#!/usr/bin/env python3
"""
Optional Stage 4 — Derive the rest of the difficulty spread.

Takes the final Styled beatmap (positions already set by apply_style.py,
which this treats as the spread's hardest difficulty, Insane) and derives
three easier ones from it — Hard, Normal, Easy — each thinning the
previous tier's stream density a bit further and clamping Difficulty
settings to that tier's own osu! ranking-criteria range (Difficulty-
specific > Easy/Normal/Hard/Insane). Thinning works two ways: merging an
adjacent close-together pair into a short slider (using the exact
positions apply_style.py already chose, so it still looks like the same
map, just calmer) or dropping a note outright. Which measures are even
eligible for thinning gets wider going down the spread:

  * Insane — no thinning at all; only Difficulty settings are touched.
  * Hard   — only the song's *repetitive* measures (a verse/chorus that
             recurs) are thinned, lightly. Non-repetitive sections (a
             bridge, an intro/outro) are left exactly as Insane had them
             — simplifying material the player only sees once doesn't
             help them learn anything.
  * Normal, Easy — thinning applies everywhere, more aggressively at
             each step down, matching those tiers' own lower overall
             note-density guidelines.

"Repetitive" is decided the same way apply_style.py's motif regularity is:
each measure gets an energy bucket (its own average loudness, quantized),
and a bucket is "repetitive" if measures with that bucket occur in more
than one separate stretch of the song — i.e. the same kind of section
comes back more than once.

No object's *timing* is ever touched — only which objects exist (some
circle pairs become sliders, some notes are dropped) and the Difficulty
section's numbers. One of those numbers, SliderMultiplier, doubles as
every slider's velocity — lowering it (Easy/Normal only, to respect the
"avoid slider velocity above 1.3" guideline) is the one case that requires
touching hit objects at all, and even then only to rescale each slider's
declared `length` so its *duration* is exactly preserved.

Usage:
    python3 make_easy.py song_insane.osu --audio song.mp3 --tier hard --output out/song_hard.osu
"""

from __future__ import annotations

import argparse
import math
import os

import random

from beatmap_utils import (HitObject, PLAYFIELD_H, PLAYFIELD_W, clamp_to_playfield, read_osu,
                            slider_length_for_gap, write_osu)
from apply_style import compute_energy_lookup, compute_measure_energy_buckets
from add_variety import is_on_downbeat

# Hitsound bit flags, matching add_variety.py.
HS_NORMAL = 0
HS_WHISTLE = 2
HS_FINISH = 4

# Difficulty-setting ranges per tier, straight from osu!'s ranking criteria
# (Difficulty-specific > Easy/Normal/Hard/Insane > Difficulty setting
# guidelines). SliderMultiplier is left alone at every tier: the "avoid
# slider velocity above 1.3" guideline turned out to actively contradict
# what a real, hand-mapped Easy difficulty looks like — the reference
# example set in example/keha_backstabber/ uses a SliderMultiplier of
# 3.54 on its Easy difficulty, because a *slow-reading* Easy leans on long
# sliders that cover real distance over several beats (a smooth, gentle
# path to track) rather than a cramped one clicking through the same
# rhythm as circles — capping the multiplier down actively fights that.
TIER_SETTINGS: dict[str, dict[str, tuple[float, float] | None]] = {
    "easy": {
        "HPDrainRate": (1.0, 3.0), "CircleSize": (2.0, 4.0),
        "OverallDifficulty": (1.0, 3.0), "ApproachRate": (2.0, 5.0),
        "SliderMultiplier": None,
    },
    "normal": {
        "HPDrainRate": (3.0, 5.0), "CircleSize": (2.0, 5.0),
        "OverallDifficulty": (3.0, 5.0), "ApproachRate": (4.0, 6.0),
        "SliderMultiplier": None,
    },
    "hard": {
        "HPDrainRate": (4.0, 6.0), "CircleSize": (2.0, 6.0),
        "OverallDifficulty": (5.0, 7.0), "ApproachRate": (6.0, 8.0),
        "SliderMultiplier": None,
    },
    "insane": {
        "HPDrainRate": (5.0, 8.0), "CircleSize": (2.0, 7.0),
        "OverallDifficulty": (7.0, 9.0), "ApproachRate": (7.0, 9.3),
        "SliderMultiplier": None,
    },
}

# How aggressively (and where) each tier thins the Insane/Styled density.
# scope="repetitive" only thins measures find_repetitive_measures() flags;
# scope="everywhere" thins every eligible measure; scope=None skips
# thinning entirely (Insane is exactly the Styled difficulty).
#
# merge_gap_beats is the real lever for how the map *feels*, not just how
# often it thins: a real Easy difficulty (see example/keha_backstabber/'s
# [Wanpachi's Easy]) never has two objects less than a full beat apart,
# and well over half its objects are sliders — a quarter-beat-only merge
# window (what this used to be fixed to) can only ever touch the same
# rapid stream notes Hard already barely touches, nowhere near enough to
# get there. Widening the merge window tier by tier is what actually
# closes that gap: Hard only chains genuine rapid streams; Easy chains
# anything up to a full beat apart, turning most of the map's *rhythm*
# — not just its stray fast bursts — into long, slow-reading slider
# chains. merge_probability gates merging too, not just dropping — a
# repetitive dance/pop track can have *most* of its measures flagged
# repetitive, so a plain "merge every eligible pair" left Hard
# (repetitive-only) thinning nearly as much of the song as Normal
# (everywhere), converging both toward a similar difficulty instead of a
# real spread.
TIER_THINNING: dict[str, dict | None] = {
    "insane": None,
    "hard": {"scope": "repetitive", "merge_gap_beats": 0.25, "merge_probability": 0.35, "drop_probability": 0.05},
    "normal": {"scope": "everywhere", "merge_gap_beats": 0.5, "merge_probability": 0.7, "drop_probability": 0.15},
    "easy": {"scope": "everywhere", "merge_gap_beats": 1.0, "merge_probability": 0.9, "drop_probability": 0.25},
}


def find_repetitive_measures(measure_buckets: dict[int, int], window: int = 4) -> set[int]:
    """Which measure indices are part of a several-measure pattern that recurs elsewhere.

    A single measure's energy bucket alone is too coarse a signal — with
    only a handful of bucket levels, almost every bucket value shows up
    somewhere else in a song purely by chance, which would mark nearly the
    whole track "repetitive." Instead this looks at short *sequences* of
    consecutive measures (a `window`-measure "shingle" of bucket values):
    if the same sequence of bucket values shows up starting at two
    well-separated measures, that's much stronger evidence of the song
    actually repeating a section (a verse or chorus recurring) than a
    single measure's loudness matching by coincidence.
    """
    n = max(measure_buckets) + 1 if measure_buckets else 0
    if n < window * 2:
        return set(measure_buckets.keys())  # too short a track to meaningfully judge repetition

    signature_starts: dict[tuple[int, ...], list[int]] = {}
    for start in range(n - window + 1):
        sig = tuple(measure_buckets.get(start + k, 0) for k in range(window))
        signature_starts.setdefault(sig, []).append(start)

    repetitive: set[int] = set()
    for sig, starts in signature_starts.items():
        # Only count starts as separate occurrences if they're far enough
        # apart that they aren't just the same overlapping window sliding
        # by one measure.
        distinct_starts = []
        for s in starts:
            if not distinct_starts or s - distinct_starts[-1] >= window:
                distinct_starts.append(s)
        if len(distinct_starts) >= 2:
            for s in distinct_starts:
                repetitive.update(range(s, s + window))
    return repetitive


def clamp_difficulty_to_tier(difficulty: dict[str, str], tier: str) -> dict[str, str]:
    """Clamp HP/CS/OD/AR (and, for Easy/Normal, SliderMultiplier) straight
    to `tier`'s own osu! ranking-criteria range (see TIER_SETTINGS) — not a
    relative shift from Insane/Styled's own (Hard/Insane-range) settings,
    which could still land outside a lower tier's actual range.

    Slider velocity is the one exception to "never touches timing": every
    slider's `length` was chosen for a specific *duration* at the original
    multiplier, and duration is proportional to length/multiplier — so the
    caller must rescale every slider's `length` by (new multiplier / old
    multiplier) alongside this change when SliderMultiplier is clamped,
    which keeps every duration exactly what it was (see main()).
    """
    ranges = TIER_SETTINGS[tier]

    def clamp(key: str, lo: float, hi: float) -> str:
        try:
            value = float(difficulty.get(key, (lo + hi) / 2))
        except ValueError:
            value = (lo + hi) / 2
        return f"{max(lo, min(hi, value)):.1f}"

    result = dict(difficulty)
    for key in ("HPDrainRate", "CircleSize", "OverallDifficulty", "ApproachRate"):
        result[key] = clamp(key, *ranges[key])
    if ranges["SliderMultiplier"] is not None:
        result["SliderMultiplier"] = clamp("SliderMultiplier", *ranges["SliderMultiplier"])
    return result


def recompute_combos(objects: list[HitObject], offset_ms: float, measure_length_ms: float,
                      max_combo_length: int = 8) -> None:
    """Re-derive new-combo flags after thinning, the same way add_variety.py does.

    Merging a circle pair into a slider keeps only the *first* object's
    is_new_combo flag — if the second one had it (it was meant to start a
    new combo), that combo break silently disappears, and the combo before
    it can run well past the usual 8-object cap. Recomputing from scratch
    (aligned to real downbeats, with the same 8-object hard cap) is more
    robust than trying to carry the old flags through a shape change.
    Mutates `objects` in place.
    """
    last_combo_time = None
    combo_count = 0
    for obj in objects:
        on_downbeat = is_on_downbeat(obj.time, offset_ms, measure_length_ms)
        overdue_time = last_combo_time is not None and (obj.time - last_combo_time) > measure_length_ms * 2.5
        overdue_count = combo_count >= max_combo_length
        obj.is_new_combo = on_downbeat or overdue_time or overdue_count
        if obj.is_new_combo:
            last_combo_time = obj.time
            combo_count = 1
        else:
            combo_count += 1


def _safe_translation_fraction(points: list[tuple[float, float]], dx: float, dy: float,
                                margin: int = 20) -> float:
    """The largest f in [0, 1] such that every point in `points`, shifted by
    (f*dx, f*dy), still lands within the playfield margin on both axes.

    Used to translate a whole slider (head + curve points) as one rigid
    unit: scaling the *same* delta down for every point can't distort the
    shape (unlike clamping each point independently), so whatever curve the
    points describe is guaranteed to stay exactly as originally shaped,
    just shifted less far.
    """
    lo_x, hi_x = margin, PLAYFIELD_W - margin
    lo_y, hi_y = margin, PLAYFIELD_H - margin
    fraction = 1.0
    for px, py in points:
        for p, d, lo, hi in ((px, dx, lo_x, hi_x), (py, dy, lo_y, hi_y)):
            if d == 0:
                continue
            # p + f*d must stay in [lo, hi] -> f in [(lo-p)/d, (hi-p)/d] (order depends on sign of d)
            f_lo, f_hi = sorted([(lo - p) / d, (hi - p) / d])
            fraction = min(fraction, max(0.0, f_hi))
    return max(0.0, fraction)


def merge_chain(chain: list[HitObject], beat_length_ms: float, slider_multiplier: float) -> HitObject:
    """Combine 2+ consecutive circles into a single multi-anchor slider,
    exactly like add_variety.py's own make_slider_chain — usable as a
    general style tool wherever a run of circles is a candidate to read as
    one held slider instead of several separate clicks, not just for
    thinning: the same "some stream runs become held slider chains instead
    of stacks" variety apply_style.py's stack/line choice already gives,
    generalized to any run of eligible circles.

    Length is derived from `slider_length_for_gap` (start to the *last*
    node), so the merged slider's declared duration reconstructs to
    exactly that same on-disk gap — no drift, no overlap with whatever
    comes after it, the same guarantee a plain pairwise merge has.
    """
    start, rest = chain[0], chain[1:]
    length = slider_length_for_gap(start.time, chain[-1].time, beat_length_ms, slider_multiplier)
    return HitObject(x=start.x, y=start.y, time=start.time, is_new_combo=start.is_new_combo,
                      hitsound=start.hitsound, is_slider=True, curve_type="L",
                      points=[(o.x, o.y) for o in rest], slides=1, length=length)


def thin_repetitive_streams(objects: list[HitObject], beat_length_ms: float, slider_multiplier: float,
                             offset_ms: float, measure_length_ms: float,
                             eligible_measures: set[int] | None, rng: random.Random,
                             drop_probability: float = 0.12, merge_probability: float = 1.0,
                             merge_gap_beats: float = 0.25, max_chain_len: int = 4) -> list[HitObject]:
    """Thin objects in `eligible_measures` (or every measure, if None —
    Normal/Easy thin everywhere; Hard passes just the repetitive ones), two
    ways: merge a run of adjacent circles into one slider chain, or drop a
    note outright. Nothing here ever computes a new object's time or
    reasons about a fixed subdivision length — it only ever reuses the
    real, already-legal timestamps and positions already sitting in
    `objects`, so nothing it does can introduce a timing overlap, an
    unsnapped slider, or an off-screen object on its own:

      * A chain of up to `max_chain_len` consecutive circles, each no more
        than `merge_gap_beats` beats from the last, merges into one slider
        via merge_chain() — `slider_length_for_gap` measures the gap
        between the chain's own (already valid, already on whatever grid
        apply_style.py left them on) start/end timestamps the same way the
        .osu file rounds them, so the merged slider's declared duration
        reconstructs to exactly that gap, landing its end precisely on the
        last node's original timestamp; and every point is an already-
        validated position apply_style.py placed, so the chain is
        guaranteed to stay on-screen. A run stops extending the moment the
        next candidate would form a "stack" (identical x, y — see
        apply_style.py's build_stream_runs) with the chain's current end:
        an "L" segment between two identical points is a real zero-length
        segment, which apply_style.py's own "slider shape consistency"
        rule exists to prevent.
      * Dropping a note removes an object; it can never overlap anything,
        since there's nothing left there to overlap. But since apply_style.py
        positioned every object via distance-snap against the object *before*
        it, removing one changes the time gap on both sides of the hole
        without updating anyone's position — left alone, the very next kept
        object would sit exactly where it did before, now visually too close
        for the larger time gap it's actually separated by. Whenever a note
        was just dropped, the next object kept is re-snapped to the correct
        distance from its new predecessor, along the same direction
        apply_style.py originally sent it in (measured against the dropped
        note it followed), so distance-snap still holds across the hole.

    `merge_gap_beats` is what actually controls how far this reaches: at
    the default quarter beat it only touches the same rapid stream notes
    add_variety.py calls a "stream"; widened toward a full beat (Easy), it
    reaches ordinary on-beat rhythm too, turning much more of the map into
    long, slow-reading slider chains — closer to how a real hand-mapped
    Easy difficulty reads than thinning stream bursts alone ever could.
    """
    merge_gap_ms = beat_length_ms * merge_gap_beats + 1.0
    px_per_beat = slider_multiplier * 100.0

    def measure_of(time_ms: float) -> int:
        return int((time_ms - offset_ms) // measure_length_ms)

    def is_eligible(time_ms: float) -> bool:
        return eligible_measures is None or measure_of(time_ms) in eligible_measures

    result: list[HitObject] = []
    i = 0
    n = len(objects)
    dropped_since_last_keep = False
    while i < n:
        obj = objects[i]

        if not obj.is_slider and is_eligible(obj.time):
            # Greedily extend a chain of eligible, non-stacked circles.
            j = i
            while (j + 1 < n and (j - i + 1) < max_chain_len
                   and not objects[j + 1].is_slider
                   and is_eligible(objects[j + 1].time)
                   and (objects[j + 1].time - objects[j].time) <= merge_gap_ms
                   and (objects[j + 1].x, objects[j + 1].y) != (objects[j].x, objects[j].y)):
                j += 1
            chain_len = j - i + 1

            if chain_len >= 2 and rng.random() < merge_probability:
                result.append(merge_chain(objects[i:j + 1], beat_length_ms, slider_multiplier))
                dropped_since_last_keep = False
                i = j + 1
                continue

            can_drop = 0 < i < n - 1 and result and not result[-1].is_slider
            # A run of 2+ eligible circles missed its merge roll: still a
            # tight, sub-merge_gap_beats pair. For a tier reaching beyond
            # genuine rapid streams (Normal/Easy, merge_gap_beats >= half a
            # beat), that's exactly what merge_gap_beats says shouldn't
            # survive — rather than leaving it to another coin flip that can
            # just as easily fail too (and does, often enough to leave stray
            # tight gaps in an otherwise sparse map), drop it outright
            # whenever dropping is possible at all. Hard (quarter-beat-only)
            # keeps the plain probabilistic drop below instead — those really
            # are the rapid stream bursts Hard is meant to barely touch, not
            # a density target to enforce.
            # A circle within merge_gap_beats of whatever comes right after
            # it (chain_len >= 2) never merged into it — either because that
            # neighbor is already a slider (can't be folded into a plain
            # chain without discarding its own shape) or the merge roll
            # missed. Same reasoning either way: force it, don't leave a
            # coin flip that can just as easily fail to another tight gap
            # a sparse tier's own merge_gap_beats says shouldn't survive.
            has_next = i + 1 < n
            next_too_close = has_next and (objects[i + 1].time - obj.time) <= merge_gap_ms
            if (chain_len >= 2 or next_too_close) and can_drop and merge_gap_beats >= 0.5:
                dropped_since_last_keep = True
                i += 1
                continue
            # A lone eligible circle with no close neighbor either side:
            # drop is still only ever probabilistic — nothing here says a
            # single beat on its own needs thinning.
            if can_drop and rng.random() < drop_probability:
                dropped_since_last_keep = True
                i += 1
                continue

        if dropped_since_last_keep and result and i > 0:
            # Re-snap against the new predecessor, keeping the direction
            # this object was already sent in from the note that got
            # dropped (objects[i - 1]) so the flow's shape doesn't change,
            # only how far this one hop travels. If this object is itself a
            # slider, it's translated as one rigid unit — head and every
            # curve point shifted by the *same* offset, scaled down (never
            # per-point clamped) just enough that all of them stay in
            # bounds. Clamping each point independently would distort the
            # shape (points moving by different amounts), and a distorted
            # "P" (perfect-circle) or "B" (Bezier) curve's actual rendered
            # arc can bulge well outside its own anchor points even when
            # every anchor is individually in bounds — the same failure
            # mode apply_style.py's own curve-shape comments warn about.
            # A uniformly-scaled rigid shift can't distort anything, so
            # this can't happen here.
            prev = result[-1]
            dropped_pred = objects[i - 1]
            angle = math.atan2(obj.y - dropped_pred.y, obj.x - dropped_pred.x)
            new_gap_ms = max(1.0, obj.time - prev.time)
            spacing = px_per_beat * (new_gap_ms / beat_length_ms)
            target_x = prev.x + spacing * math.cos(angle)
            target_y = prev.y + spacing * math.sin(angle)
            delta_x, delta_y = target_x - obj.x, target_y - obj.y

            all_points = [(obj.x, obj.y)] + list(obj.points)
            fraction = _safe_translation_fraction(all_points, delta_x, delta_y)
            new_x, new_y = clamp_to_playfield(obj.x + fraction * delta_x, obj.y + fraction * delta_y)
            new_points = [clamp_to_playfield(px + fraction * delta_x, py + fraction * delta_y)
                          for px, py in obj.points]
            obj = HitObject(x=new_x, y=new_y, time=obj.time, is_new_combo=obj.is_new_combo,
                             hitsound=obj.hitsound, is_slider=obj.is_slider, curve_type=obj.curve_type,
                             points=new_points, slides=obj.slides, length=obj.length,
                             edge_hitsounds=obj.edge_hitsounds, edge_samplesets=obj.edge_samplesets)

        dropped_since_last_keep = False
        result.append(obj)
        i += 1
    return result


def regularize_hitsounds(objects: list[HitObject], offset_ms: float, measure_length_ms: float,
                          eligible_measures: set[int] | None) -> None:
    """In eligible measures (or every measure, if None), guarantee a
    predictable downbeat accent — easier to anticipate. Mutates in place;
    this only ever *adds* a finish accent on downbeats that don't already
    have one, never removes an existing accent elsewhere in the measure:
    forcing every non-downbeat hit down to a plain HS_NORMAL (an earlier
    version of this function did) can silence long, otherwise fine
    stretches of the styled map's own accenting whenever eligible_measures
    covers most or all of the song (Normal/Easy's "everywhere" scope),
    which is exactly the "long periods without hitsounds" a checker (or a
    player) would flag. Non-eligible measures (and slider repeats) are
    left exactly as they were either way."""

    def is_downbeat(time_ms: float) -> bool:
        rel = (time_ms - offset_ms) % measure_length_ms
        return min(rel, measure_length_ms - rel) < 1.0

    def measure_of(time_ms: float) -> int:
        return int((time_ms - offset_ms) // measure_length_ms)

    for obj in objects:
        if eligible_measures is not None and measure_of(obj.time) not in eligible_measures:
            continue
        if not is_downbeat(obj.time):
            continue
        obj.hitsound = HS_FINISH
        if obj.is_slider:
            tail = obj.edge_hitsounds[-1] if obj.edge_hitsounds else HS_NORMAL
            obj.edge_hitsounds = [HS_FINISH] + [tail] * obj.slides


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive one difficulty tier from the Styled/Insane beatmap.")
    parser.add_argument("beatmap", help="Path to the Styled .osu file (from apply_style.py) — treated as Insane.")
    parser.add_argument("--audio", required=True, help="Path to the same song's MP3 (used to find repetitive sections).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tier", choices=["easy", "normal", "hard", "insane"], default="easy",
                         help="Which difficulty tier to derive (see TIER_SETTINGS/TIER_THINNING). "
                              "Determines both the Difficulty-setting range and how much thinning "
                              "is applied and where.")
    parser.add_argument("--version", default=None,
                         help="Difficulty/version name to write into the map. Defaults to the tier "
                              "name, capitalized (Easy/Normal/Hard/Insane).")
    parser.add_argument("--drop-probability", type=float, default=None,
                         help="Chance an unmerged eligible stream note is dropped entirely for "
                              "extra thinning (0-1). Defaults to the tier's own value.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for which notes get dropped. Omit for a different "
                              "result every run; pass a fixed value (printed on every run) to "
                              "reproduce it later.")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.SystemRandom().randrange(2**32)
    rng = random.Random(args.seed)
    print(f"Using seed: {args.seed}")

    bm = read_osu(args.beatmap)
    bm.metadata["Version"] = args.version or args.tier.capitalize()
    beat_length_ms = bm.beat_length
    slider_multiplier = bm.slider_multiplier
    offset_ms = bm.offset
    measure_length_ms = beat_length_ms * 4.0

    objects = sorted(bm.hit_objects, key=lambda h: h.time)
    if not objects:
        raise RuntimeError("Beatmap has no hit objects to derive a difficulty from.")
    original_start = objects[0].time
    # Use *end* time, not .time (start), for drain purposes — chain-merging
    # can now (with a wide enough merge_gap_beats) fold the last few original
    # objects into one slider whose .time is the chain's *start*, well before
    # the original last object's own time; the slider's end (which the merge
    # deliberately lands exactly on the last folded object's original time —
    # see merge_chain/slider_length_for_gap) is what actually still matches.
    original_end = objects[-1].end_time(beat_length_ms, slider_multiplier)

    thinning = TIER_THINNING[args.tier]
    if thinning is not None:
        print("Analyzing song structure...")
        energy_at = compute_energy_lookup(args.audio)
        measure_buckets = compute_measure_energy_buckets(energy_at, offset_ms, measure_length_ms, objects[-1].time)
        if thinning["scope"] == "repetitive":
            eligible_measures = find_repetitive_measures(measure_buckets)
            print(f"  {len(eligible_measures)}/{len(measure_buckets)} measures are in a repeating section")
        else:
            eligible_measures = None  # thin everywhere

        drop_probability = args.drop_probability if args.drop_probability is not None else thinning["drop_probability"]
        merge_probability = thinning["merge_probability"]
        merge_gap_beats = thinning["merge_gap_beats"]
        before = len(objects)
        objects = thin_repetitive_streams(objects, beat_length_ms, slider_multiplier, offset_ms,
                                           measure_length_ms, eligible_measures, rng,
                                           drop_probability=drop_probability, merge_probability=merge_probability,
                                           merge_gap_beats=merge_gap_beats)
        print(f"Thinned {before - len(objects)} object(s) by merging stream pairs into sliders or dropping them")

        regularize_hitsounds(objects, offset_ms, measure_length_ms, eligible_measures)
        recompute_combos(objects, offset_ms, measure_length_ms)

    bm.difficulty = clamp_difficulty_to_tier(bm.difficulty, args.tier)
    new_slider_multiplier = bm.slider_multiplier

    # Lowering SliderMultiplier (Easy/Normal only, to respect the "avoid
    # slider velocity above 1.3" guideline) would otherwise silently change
    # every slider's duration, since duration is proportional to
    # length/multiplier — rescale every slider's declared length by the
    # same ratio so the *duration* every slider already has is exactly
    # preserved. This is the one case where a Difficulty-section change
    # requires touching hit objects to keep the "never touches timing"
    # promise. A no-op for Hard/Insane, whose SliderMultiplier is untouched.
    if new_slider_multiplier != slider_multiplier:
        ratio = new_slider_multiplier / slider_multiplier
        for obj in objects:
            if obj.is_slider:
                obj.length *= ratio

    bm.hit_objects = objects

    # Sanity check: thinning must never introduce an overlap, judged the same
    # way the .osu file itself will be read back (see add_variety.py's own
    # version of this check for why raw floats aren't the right comparison).
    # Uses the *new* multiplier, matching what bm.difficulty (and so the
    # written file) now actually holds.
    for a, b in zip(objects, objects[1:]):
        a_end = round(a.time) + a.duration_ms(beat_length_ms, new_slider_multiplier)
        if round(b.time) < a_end - 1e-6:
            raise AssertionError(f"Overlap introduced at {a.time:.1f}ms while deriving the {args.tier} difficulty.")

    # Ranking criteria requires every difficulty in a set to have essentially
    # the same drain time — thin_repetitive_streams merges/drops only ever
    # touch objects *between* the first and last object (dropping requires
    # both a predecessor already in `result` and a successor still ahead in
    # `objects`, which structurally excludes index 0 and the very last
    # index; merging always lands its slider's end exactly on the last
    # folded object's own original time, via slider_length_for_gap), so
    # this should never fire — kept as an explicit guarantee rather than an
    # assumption. Compares *end* time (not .time/start) since the final
    # object can now be a merged slider whose start is well before the
    # original last object's own time.
    final_end = objects[-1].end_time(beat_length_ms, new_slider_multiplier)
    if objects[0].time != original_start or abs(final_end - original_end) > 1e-6:
        raise AssertionError(f"The {args.tier} difficulty's drain time no longer matches Insane's.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_osu(bm, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
