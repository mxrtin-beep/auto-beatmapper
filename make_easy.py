#!/usr/bin/env python3
"""
Optional Stage 4 — Derive the rest of the difficulty spread.

Takes the final Styled beatmap (positions already set by apply_style.py,
which this treats as the spread's hardest difficulty, Insane) and derives
three easier ones from it — Hard, Normal, Easy — by deleting objects,
nothing else: no merging, no reshaping. An object either survives exactly
as apply_style.py placed it, or it's gone entirely — never partially
edited into something new (which is exactly what was producing spacing
complaints: a merged slider's on-screen geometry doesn't automatically
read as correctly "spaced" for a checker the way two untouched, already-
validated objects farther apart in the surviving sequence do). Circles
sitting on a quarter- or eighth-beat subdivision are deleted first and
most often — the fast subdivisions a lower tier has the least business
keeping — then half-beat circles, then (rarely) whole-beat circles or
whole sliders. Difficulty settings are also clamped to each tier's own
osu! ranking-criteria range (Difficulty-specific > Easy/Normal/Hard/
Insane), and downbeats are never deleted, which keeps regularize_hitsounds
below able to guarantee no measure goes without an accent. Which measures
are even eligible for deletion gets wider going down the spread:

  * Insane — no deletion at all; only Difficulty settings are touched.
  * Hard   — only the song's *repetitive* measures (a verse/chorus that
             recurs) are thinned, lightly. Non-repetitive sections (a
             bridge, an intro/outro) are left exactly as Insane had them
             — simplifying material the player only sees once doesn't
             help them learn anything.
  * Normal, Easy — deletion applies everywhere, more aggressively at
             each step down, matching those tiers' own lower overall
             note-density guidelines.

"Repetitive" is decided the same way apply_style.py's motif regularity is:
each measure gets an energy bucket (its own average loudness, quantized),
and a bucket is "repetitive" if measures with that bucket occur in more
than one separate stretch of the song — i.e. the same kind of section
comes back more than once.

No object's *timing* is ever touched, and no object is ever partially
edited — only which objects exist, and the Difficulty section's numbers.

Usage:
    python3 make_easy.py song_insane.osu --audio song.mp3 --tier hard --output out/song_hard.osu
"""

from __future__ import annotations

import argparse
import math
import os

import random

from beatmap_utils import HitObject, PLAYFIELD_H, PLAYFIELD_W, clamp_to_playfield, read_osu, write_osu
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

# How aggressively (and where) each tier deletes objects, by category.
# scope="repetitive" only thins measures find_repetitive_measures() flags;
# scope="everywhere" thins every eligible measure; scope=None skips
# thinning entirely (Insane is exactly the Styled difficulty). Deletion
# probability is highest for circles on a quarter/eighth-beat subdivision
# (the fast, stream-like density a lower tier has the least business
# keeping), lower for half-beat circles, lower still for whole-beat
# circles (the backbone of the rhythm — rarely touched even in Easy), and
# separately tunable for whole sliders (never split or reshaped, only ever
# kept whole or deleted whole).
TIER_DELETE_PROBABILITY: dict[str, dict | None] = {
    "insane": None,
    "hard": {"scope": "repetitive", "min_gap_beats": 0.0,
             "quarter_eighth": 0.35, "half_beat": 0.10, "whole_beat": 0.0, "slider": 0.05},
    "normal": {"scope": "everywhere", "min_gap_beats": 0.5,
               "quarter_eighth": 0.75, "half_beat": 0.35, "whole_beat": 0.05, "slider": 0.15},
    "easy": {"scope": "everywhere", "min_gap_beats": 1.0,
             "quarter_eighth": 0.97, "half_beat": 0.65, "whole_beat": 0.12, "slider": 0.30},
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


def classify_beat_position(time_ms: float, offset_ms: float, beat_length_ms: float,
                            tolerance_ms: float = 2.0) -> str:
    """Which subdivision of the beat grid `time_ms` sits on: "whole_beat",
    "half_beat", or "quarter_eighth" (anything finer, including an eighth
    beat or anything add_variety.py's own subdivision-fitting placed off
    the clean quarter grid — all equally "fast" from a lower tier's
    perspective, so all treated the same for deletion priority)."""

    def near_multiple(period_ms: float) -> bool:
        rel = (time_ms - offset_ms) % period_ms
        return min(rel, period_ms - rel) < tolerance_ms

    if near_multiple(beat_length_ms):
        return "whole_beat"
    if near_multiple(beat_length_ms / 2.0):
        return "half_beat"
    return "quarter_eighth"


def estimate_spacing_scale(objects: list[HitObject], beat_length_ms: float, slider_multiplier: float) -> float:
    """The map's own actual on-screen-distance-per-beat, relative to the
    bare distance-snap formula (slider_multiplier * 100 px/beat).

    apply_style.py scales that bare formula by its own --spacing (default
    1.3, plus a few percent of seeded wobble) before placing anything, so
    re-snapping an object after a deletion using the *bare* formula (scale
    1.0) makes its gap visually inconsistent with every untouched gap
    around it — exactly what a spacing checker comparing "expected vs.
    actual" px against neighboring objects flags, both when the untouched
    gaps read larger (bare 1.0 vs. apply_style's 1.3) and when a checker's
    own expectation is instead anchored to those larger neighbors. Sampling
    the ratio the existing (untouched) map already used, rather than
    assuming any particular --spacing value, keeps a re-snapped gap
    consistent with its neighbors regardless of what --spacing the Insane
    beatmap was actually styled with.
    """
    ratios = []
    for a, b in zip(objects, objects[1:]):
        if a.is_slider:
            continue
        gap_ms = b.time - a.time
        if gap_ms <= 0:
            continue
        expected_1x = slider_multiplier * 100.0 * (gap_ms / beat_length_ms)
        if expected_1x < 1.0:
            continue
        dist = math.hypot(b.x - a.x, b.y - a.y)
        if dist < 1.0:
            continue  # a deliberate stack — not governed by distance-snap at all
        ratios.append(dist / expected_1x)
    if not ratios:
        return 1.0
    ratios.sort()
    return ratios[len(ratios) // 2]  # median — robust to the odd jump/stream outlier


def _resnap_after_drop(obj: HitObject, prev: HitObject, dropped_pred: HitObject,
                        beat_length_ms: float, px_per_beat: float) -> HitObject:
    """`obj` is the next kept object after one or more deletions; re-snap it
    to the correct distance-snap spacing from its new predecessor `prev`,
    along the same direction apply_style.py originally sent it in (measured
    against `dropped_pred`, whichever object it immediately followed before
    any deletions) — so distance-snap still holds across the hole, instead
    of `obj` sitting exactly where it was for a since-deleted predecessor,
    now visually too close for the larger time gap it's really separated
    by. `px_per_beat` should be `slider_multiplier * 100 * estimate_spacing_scale(...)`,
    not the bare formula — see estimate_spacing_scale for why. If `obj` is
    itself a slider, it's translated as one rigid unit — head and every
    curve point shifted by the *same* offset, scaled down (never per-point
    clamped) just enough that all of them stay in bounds. Clamping each
    point independently would distort the shape (points moving by
    different amounts), and a distorted "P" (perfect-circle) or "B"
    (Bezier) curve's actual rendered arc can bulge well outside its own
    anchor points even when every anchor is individually in bounds — the
    same failure mode apply_style.py's own curve-shape comments warn
    about. A uniformly-scaled rigid shift can't distort anything.
    """
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
    return HitObject(x=new_x, y=new_y, time=obj.time, is_new_combo=obj.is_new_combo,
                      hitsound=obj.hitsound, is_slider=obj.is_slider, curve_type=obj.curve_type,
                      points=new_points, slides=obj.slides, length=obj.length,
                      edge_hitsounds=obj.edge_hitsounds, edge_samplesets=obj.edge_samplesets)


def thin_by_deletion(objects: list[HitObject], beat_length_ms: float, slider_multiplier: float,
                      offset_ms: float, measure_length_ms: float,
                      eligible_measures: set[int] | None, rng: random.Random,
                      delete_probability: dict[str, float]) -> list[HitObject]:
    """Thin objects in `eligible_measures` (or every measure, if None —
    Normal/Easy thin everywhere; Hard passes just the repetitive ones) by
    deleting some of them outright — nothing is ever merged, split, or
    reshaped, only kept exactly as apply_style.py placed it or removed
    entirely. Each object's category (see classify_beat_position; sliders
    are their own category regardless of what beat they start on) picks
    its deletion probability from `delete_probability`.

    Never deletes the very first or last object (drain time), or an object
    sitting on a downbeat (so regularize_hitsounds below can always find
    something to accent every measure — no gap without a hitsound).
    Deleting an object can never introduce a timing overlap, since nothing
    is left there to overlap — but it does leave the *next* kept object
    sitting exactly where apply_style.py put it for a since-deleted
    predecessor, no longer matching the (now larger) time gap it's really
    separated by. That object is re-snapped to the correct distance from
    its new predecessor, in the same direction apply_style.py originally
    sent it (measured against whichever object it immediately followed
    before any deletions), so distance-snap always holds across a deleted
    run regardless of how many objects in a row got removed or what type
    they were.
    """

    def measure_of(time_ms: float) -> int:
        return int((time_ms - offset_ms) // measure_length_ms)

    def is_eligible(time_ms: float) -> bool:
        return eligible_measures is None or measure_of(time_ms) in eligible_measures

    def category_of(obj: HitObject) -> str:
        return "slider" if obj.is_slider else classify_beat_position(obj.time, offset_ms, beat_length_ms)

    px_per_beat = slider_multiplier * 100.0 * estimate_spacing_scale(objects, beat_length_ms, slider_multiplier)

    n = len(objects)
    result: list[HitObject] = []
    dropped_since_last_keep = False
    for i, obj in enumerate(objects):
        can_delete = (0 < i < n - 1 and is_eligible(obj.time)
                      and not is_on_downbeat(obj.time, offset_ms, measure_length_ms))
        if can_delete and rng.random() < delete_probability[category_of(obj)]:
            dropped_since_last_keep = True
            continue

        if dropped_since_last_keep and result:
            obj = _resnap_after_drop(obj, result[-1], objects[i - 1], beat_length_ms, px_per_beat)

        dropped_since_last_keep = False
        result.append(obj)
    return result


def enforce_min_gap(objects: list[HitObject], beat_length_ms: float, slider_multiplier: float,
                     offset_ms: float, measure_length_ms: float, min_gap_beats: float) -> list[HitObject]:
    """Deterministically delete whatever's needed so no two consecutive
    objects are closer than `min_gap_beats` — thin_by_deletion's per-
    category probabilities can, by chance, leave a pair that close (a coin
    flip failing is still a coin flip), and a gap that tight reads as a
    real visual overlap at a lower tier's larger circle size, exactly what
    a beatmap checker flags as bad spacing. Downbeats are still never
    deleted (same reasoning as thin_by_deletion — regularize_hitsounds
    needs one every measure), so a genuine downbeat-to-downbeat collision
    at an extremely fast tempo could still leave a tight pair; that's
    always been a real limit of the source material, not something
    deleting more objects could fix anyway.

    A single left-to-right pass is enough: each kept object is compared
    only against the nearest object *already kept* (not the original
    sequence), so a deletion earlier in the pass is what setting the bar
    for the next comparison is already based on — no separate pass needed
    to catch a gap that only appears once its own predecessor was removed.
    """
    min_gap_ms = beat_length_ms * min_gap_beats - 1.0
    px_per_beat = slider_multiplier * 100.0 * estimate_spacing_scale(objects, beat_length_ms, slider_multiplier)
    n = len(objects)
    result: list[HitObject] = []
    dropped_since_last_keep = False
    for i, obj in enumerate(objects):
        if (0 < i < n - 1 and result and (obj.time - result[-1].time) < min_gap_ms
                and not is_on_downbeat(obj.time, offset_ms, measure_length_ms)):
            dropped_since_last_keep = True
            continue

        if dropped_since_last_keep and result:
            obj = _resnap_after_drop(obj, result[-1], objects[i - 1], beat_length_ms, px_per_beat)

        dropped_since_last_keep = False
        result.append(obj)
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
                         help="Which difficulty tier to derive (see TIER_SETTINGS/TIER_DELETE_PROBABILITY). "
                              "Determines both the Difficulty-setting range and how much deletion "
                              "is applied and where.")
    parser.add_argument("--version", default=None,
                         help="Difficulty/version name to write into the map. Defaults to the tier "
                              "name, capitalized (Easy/Normal/Hard/Insane).")
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
    # end_time, not .time (start): a superset of the simpler "first/last
    # object's .time is unchanged" invariant that deletion alone already
    # guarantees (the first/last object is never deleted — see
    # thin_by_deletion), but costs nothing to check the more general way.
    original_end = objects[-1].end_time(beat_length_ms, slider_multiplier)

    thinning = TIER_DELETE_PROBABILITY[args.tier]
    if thinning is not None:
        print("Analyzing song structure...")
        energy_at = compute_energy_lookup(args.audio)
        measure_buckets = compute_measure_energy_buckets(energy_at, offset_ms, measure_length_ms, objects[-1].time)
        if thinning["scope"] == "repetitive":
            eligible_measures = find_repetitive_measures(measure_buckets)
            print(f"  {len(eligible_measures)}/{len(measure_buckets)} measures are in a repeating section")
        else:
            eligible_measures = None  # thin everywhere

        delete_probability = {k: v for k, v in thinning.items() if k not in ("scope", "min_gap_beats")}
        before = len(objects)
        objects = thin_by_deletion(objects, beat_length_ms, slider_multiplier, offset_ms,
                                    measure_length_ms, eligible_measures, rng,
                                    delete_probability=delete_probability)
        print(f"Deleted {before - len(objects)} object(s)")

        min_gap_beats = thinning["min_gap_beats"]
        if min_gap_beats > 0:
            before_gap = len(objects)
            objects = enforce_min_gap(objects, beat_length_ms, slider_multiplier, offset_ms,
                                       measure_length_ms, min_gap_beats)
            if len(objects) != before_gap:
                print(f"Deleted {before_gap - len(objects)} more object(s) to clear "
                      f"sub-{min_gap_beats}-beat gaps (checkers flag these as bad spacing)")

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
    # the same drain time — thin_by_deletion never deletes the first or last
    # object, so this should never fire — kept as an explicit guarantee
    # rather than an assumption.
    final_end = objects[-1].end_time(beat_length_ms, new_slider_multiplier)
    if objects[0].time != original_start or abs(final_end - original_end) > 1e-6:
        raise AssertionError(f"The {args.tier} difficulty's drain time no longer matches Insane's.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_osu(bm, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
