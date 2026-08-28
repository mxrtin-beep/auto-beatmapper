#!/usr/bin/env python3
"""
Optional Stage 4 — Make an easier difficulty.

Takes the final Styled beatmap (positions already set by apply_style.py)
and derives a second, easier difficulty from it: lower Difficulty settings,
and fewer clicks in the song's *repetitive* sections (a verse or chorus
that recurs) — a run of closely-spaced circles there gets thinned, either
by merging adjacent pairs into short sliders (using the exact positions
apply_style.py already chose, so it still looks like the same map, just
calmer) or by dropping some of them outright, so it still looks like the
same map, just calmer. Non-repetitive sections (a bridge, an intro/outro,
anything that only happens once) are left untouched — simplifying material
the player only ever sees once doesn't help them learn anything.

"Repetitive" is decided the same way apply_style.py's motif regularity is:
each measure gets an energy bucket (its own average loudness, quantized),
and a bucket is "repetitive" if measures with that bucket occur in more
than one separate stretch of the song — i.e. the same kind of section
comes back more than once.

No object's *timing* is ever touched — only which objects exist (some
circle pairs become sliders, some notes are dropped) and the Difficulty
section's numbers, which are clamped to osu!'s own "Easy" difficulty
guidelines (Ranking Criteria, Difficulty-specific > Easy). One of those
numbers, SliderMultiplier, doubles as every slider's velocity — lowering
it to respect the "avoid slider velocity above 1.3" guideline is the one
case that requires touching hit objects at all, and even then only to
rescale each slider's declared `length` so its *duration* is exactly
preserved.

Usage:
    python3 make_easy.py song_styled.osu --audio song.mp3 --output out/song_easy.osu
"""

from __future__ import annotations

import argparse
import math
import os

import random

from beatmap_utils import HitObject, clamp_to_playfield, read_osu, slider_length_for_gap, write_osu
from apply_style import compute_energy_lookup, compute_measure_energy_buckets
from add_variety import is_on_downbeat

# Hitsound bit flags, matching add_variety.py.
HS_NORMAL = 0
HS_WHISTLE = 2
HS_FINISH = 4


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


def easier_difficulty(difficulty: dict[str, str]) -> dict[str, str]:
    """Lower HP/CS/OD/AR/SliderMultiplier to osu!'s own "Easy" difficulty
    setting guidelines (Ranking Criteria, Difficulty-specific > Easy):
    AR <= 5, OD/HP between 1 and 3, CS <= 4, slider velocity (SliderMultiplier)
    not above 1.3. The Styled difficulty's own settings sit in Hard/Insane
    range, so a small shift-then-clamp (as a previous version of this
    function did) isn't enough to land in Easy's actual range — this clamps
    straight to it.

    Slider velocity is the one exception to "never touches timing": every
    slider's `length` was chosen for a specific *duration* at the original
    multiplier, and duration is proportional to length/multiplier — so the
    caller must rescale every slider's `length` by (new multiplier / old
    multiplier) alongside this change, which keeps every duration exactly
    what it was (see main()). Difficulty is not touched.
    """

    def clamp(key: str, lo: float, hi: float) -> str:
        try:
            value = float(difficulty.get(key, (lo + hi) / 2))
        except ValueError:
            value = (lo + hi) / 2
        return f"{max(lo, min(hi, value)):.1f}"

    easier = dict(difficulty)
    easier["HPDrainRate"] = clamp("HPDrainRate", 1.0, 3.0)
    easier["CircleSize"] = clamp("CircleSize", 2.0, 4.0)
    easier["OverallDifficulty"] = clamp("OverallDifficulty", 1.0, 3.0)
    easier["ApproachRate"] = clamp("ApproachRate", 2.0, 5.0)
    easier["SliderMultiplier"] = clamp("SliderMultiplier", 0.8, 1.3)
    return easier


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


def thin_repetitive_streams(objects: list[HitObject], beat_length_ms: float, slider_multiplier: float,
                             offset_ms: float, measure_length_ms: float,
                             repetitive_measures: set[int], rng: random.Random,
                             drop_probability: float = 0.12) -> list[HitObject]:
    """Thin closely-spaced circles in repetitive measures, two ways: merge an
    adjacent pair into a slider, or drop a note outright. Nothing here ever
    computes a new object's time or reasons about a fixed subdivision length
    — it only ever reuses the two real, already-legal timestamps that are
    already sitting in `objects`, so nothing it does can introduce a timing
    overlap or an unsnapped slider on its own:

      * Merging obj/nxt into one slider uses `slider_length_for_gap`, which
        measures the gap between their *own* (already valid, already on
        whatever grid apply_style.py left them on) timestamps the same way
        the .osu file rounds them — so the merged slider's declared
        duration reconstructs to exactly that same gap, landing its end
        precisely on nxt's original timestamp. There is nothing left to
        drift off-grid or overlap the object after it, because "the grid"
        here just means "where these two objects already, legally, were".
        A pair apply_style.py placed in a "stack" (identical x, y — see
        apply_style.py's build_stream_runs) is skipped instead of merged:
        a straight "L" slider between two identical points is a real
        zero-length slider — its declared `length` (from the time gap)
        would no longer match its actual geometric path length of zero,
        which apply_style.py's own "slider shape consistency" rule exists
        to prevent.
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

    A quarter-beat-or-less gap between two circles is exactly what
    add_variety.py calls a stream note — that's the density this targets.
    Both operations are restricted to repetitive measures, so the player
    still learns the section once at full density elsewhere in the map.
    """
    quarter_beat_ms = beat_length_ms / 4.0
    threshold = quarter_beat_ms + 1.0
    px_per_beat = slider_multiplier * 100.0

    def measure_of(time_ms: float) -> int:
        return int((time_ms - offset_ms) // measure_length_ms)

    result: list[HitObject] = []
    i = 0
    n = len(objects)
    dropped_since_last_keep = False
    while i < n:
        obj = objects[i]
        has_next = i + 1 < n
        gap_ms = (objects[i + 1].time - obj.time) if has_next else None
        is_dense_stream_note = not obj.is_slider and has_next and gap_ms <= threshold
        in_repetitive_section = is_dense_stream_note and measure_of(obj.time) in repetitive_measures
        is_stacked_pair = has_next and (objects[i + 1].x, objects[i + 1].y) == (obj.x, obj.y)

        if in_repetitive_section and not objects[i + 1].is_slider and not is_stacked_pair:
            nxt = objects[i + 1]
            length = slider_length_for_gap(obj.time, nxt.time, beat_length_ms, slider_multiplier)
            result.append(HitObject(
                x=obj.x, y=obj.y, time=obj.time, is_new_combo=obj.is_new_combo, hitsound=obj.hitsound,
                is_slider=True, curve_type="L", points=[(nxt.x, nxt.y)], slides=1, length=length,
            ))
            dropped_since_last_keep = False
            i += 2
            continue

        if (in_repetitive_section and result and not result[-1].is_slider
                and rng.random() < drop_probability):
            dropped_since_last_keep = True
            i += 1  # drop this note entirely
            continue

        if dropped_since_last_keep and result and i > 0:
            # Re-snap against the new predecessor, keeping the direction
            # this object was already sent in from the note that got
            # dropped (objects[i - 1]) so the flow's shape doesn't change,
            # only how far this one hop travels. If this object is itself a
            # slider, its curve points (absolute coordinates, not relative
            # to its head) are translated by the same offset as its head so
            # the curve stays attached to it instead of being left behind.
            prev = result[-1]
            dropped_pred = objects[i - 1]
            angle = math.atan2(obj.y - dropped_pred.y, obj.x - dropped_pred.x)
            new_gap_ms = max(1.0, obj.time - prev.time)
            spacing = px_per_beat * (new_gap_ms / beat_length_ms)
            new_x, new_y = clamp_to_playfield(prev.x + spacing * math.cos(angle),
                                               prev.y + spacing * math.sin(angle))
            delta_x, delta_y = new_x - obj.x, new_y - obj.y
            new_points = [clamp_to_playfield(px + delta_x, py + delta_y) for px, py in obj.points]
            obj = HitObject(x=new_x, y=new_y, time=obj.time, is_new_combo=obj.is_new_combo,
                             hitsound=obj.hitsound, is_slider=obj.is_slider, curve_type=obj.curve_type,
                             points=new_points, slides=obj.slides, length=obj.length,
                             edge_hitsounds=obj.edge_hitsounds, edge_samplesets=obj.edge_samplesets)

        dropped_since_last_keep = False
        result.append(obj)
        i += 1
    return result


def regularize_hitsounds(objects: list[HitObject], offset_ms: float, measure_length_ms: float,
                          repetitive_measures: set[int]) -> None:
    """In repetitive sections, settle on one predictable accent (downbeat only) instead of
    the styled map's fuller finish/clap/whistle variety — easier to anticipate. Mutates in place;
    non-repetitive sections (and slider repeats) are left exactly as they were."""

    def is_downbeat(time_ms: float) -> bool:
        rel = (time_ms - offset_ms) % measure_length_ms
        return min(rel, measure_length_ms - rel) < 1.0

    def measure_of(time_ms: float) -> int:
        return int((time_ms - offset_ms) // measure_length_ms)

    for obj in objects:
        if measure_of(obj.time) not in repetitive_measures:
            continue
        hs = HS_FINISH if is_downbeat(obj.time) else HS_NORMAL
        obj.hitsound = hs
        if obj.is_slider:
            obj.edge_hitsounds = [hs] + [HS_NORMAL] * obj.slides


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive an easier difficulty from a Styled beatmap.")
    parser.add_argument("beatmap", help="Path to the Styled .osu file (from apply_style.py).")
    parser.add_argument("--audio", required=True, help="Path to the same song's MP3 (used to find repetitive sections).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", default="Auto Easy", help="Difficulty/version name to write into the map.")
    parser.add_argument("--drop-probability", type=float, default=0.12,
                         help="Chance an unmerged repetitive-section stream note is dropped "
                              "entirely for extra thinning (0-1).")
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
    bm.metadata["Version"] = args.version
    beat_length_ms = bm.beat_length
    slider_multiplier = bm.slider_multiplier
    offset_ms = bm.offset
    measure_length_ms = beat_length_ms * 4.0

    objects = sorted(bm.hit_objects, key=lambda h: h.time)
    if not objects:
        raise RuntimeError("Beatmap has no hit objects to simplify.")
    original_start, original_end = objects[0].time, objects[-1].time

    print("Analyzing song structure...")
    energy_at = compute_energy_lookup(args.audio)
    measure_buckets = compute_measure_energy_buckets(energy_at, offset_ms, measure_length_ms, objects[-1].time)
    repetitive_measures = find_repetitive_measures(measure_buckets)
    print(f"  {len(repetitive_measures)}/{len(measure_buckets)} measures are in a repeating section")

    before = len(objects)
    objects = thin_repetitive_streams(objects, beat_length_ms, slider_multiplier, offset_ms,
                                       measure_length_ms, repetitive_measures, rng,
                                       drop_probability=args.drop_probability)
    print(f"Thinned {before - len(objects)} object(s) by merging stream pairs into sliders or dropping them")

    regularize_hitsounds(objects, offset_ms, measure_length_ms, repetitive_measures)
    recompute_combos(objects, offset_ms, measure_length_ms)

    bm.difficulty = easier_difficulty(bm.difficulty)
    new_slider_multiplier = bm.slider_multiplier

    # Lowering SliderMultiplier (to respect the Easy-tier "avoid slider
    # velocity above 1.3" guideline) would otherwise silently change every
    # slider's duration, since duration is proportional to length/multiplier
    # — rescale every slider's declared length by the same ratio so the
    # *duration* every slider already has is exactly preserved. This is the
    # one case where a Difficulty-section change requires touching hit
    # objects to keep the "never touches timing" promise.
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
            raise AssertionError(f"Overlap introduced at {a.time:.1f}ms while thinning for Easy mode.")

    # Ranking criteria requires every difficulty in a set to have essentially
    # the same drain time — thin_repetitive_streams merges/drops only ever
    # touch stream notes *between* the first and last object (dropping
    # requires both a predecessor already in `result` and a successor still
    # ahead in `objects`, which structurally excludes index 0 and the very
    # last index; merging always keeps the merged pair's own start/end
    # timestamps), so this should never fire — kept as an explicit guarantee
    # rather than an assumption.
    if objects[0].time != original_start or objects[-1].time != original_end:
        raise AssertionError("Easy mode's drain time no longer matches the Styled difficulty's.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_osu(bm, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
