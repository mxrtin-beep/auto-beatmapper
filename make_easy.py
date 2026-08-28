#!/usr/bin/env python3
"""
Optional Stage 4 — Make an easier difficulty.

Takes the final Styled beatmap (positions already set by apply_style.py)
and derives a second, easier difficulty from it: lower Difficulty settings,
and fewer clicks in the song's *repetitive* sections (a verse or chorus
that recurs) — a run of closely-spaced circles there gets thinned by
merging adjacent pairs into short sliders, using the exact positions
apply_style.py already chose, so it still looks like the same map, just
calmer. Non-repetitive sections (a bridge, an intro/outro, anything that
only happens once) are left untouched — simplifying material the player
only ever sees once doesn't help them learn anything.

"Repetitive" is decided the same way apply_style.py's motif regularity is:
each measure gets an energy bucket (its own average loudness, quantized),
and a bucket is "repetitive" if measures with that bucket occur in more
than one separate stretch of the song — i.e. the same kind of section
comes back more than once.

Timing is never touched — only which objects exist (some circle pairs
become sliders) and the Difficulty section's numbers.

Usage:
    python3 make_easy.py song_styled.osu --audio song.mp3 --output out/song_easy.osu
"""

from __future__ import annotations

import argparse
import os

from beatmap_utils import HitObject, read_osu, write_osu
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
    """Lower HP/CS/OD/AR (larger circles, more reaction time, more forgiving timing/health)."""

    def shift(key: str, delta: float, lo: float, hi: float) -> str:
        try:
            value = float(difficulty.get(key, (lo + hi) / 2))
        except ValueError:
            value = (lo + hi) / 2
        return f"{max(lo, min(hi, value + delta)):.1f}"

    easier = dict(difficulty)
    easier["HPDrainRate"] = shift("HPDrainRate", -2.0, 1.0, 7.0)
    easier["CircleSize"] = shift("CircleSize", -1.5, 2.0, 5.0)
    easier["OverallDifficulty"] = shift("OverallDifficulty", -2.0, 1.0, 6.0)
    easier["ApproachRate"] = shift("ApproachRate", -2.0, 3.0, 7.0)
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
                             repetitive_measures: set[int]) -> list[HitObject]:
    """Merge adjacent close-together circle pairs into sliders, only in repetitive measures.

    A quarter-beat-or-less gap between two circles is exactly what
    add_variety.py calls a stream note. Merging a pair into a slider keeps
    the same two positions (so it still looks like the same map) but cuts
    the click count for that pair in half.
    """
    quarter_beat_ms = beat_length_ms / 4.0
    threshold = quarter_beat_ms + 1.0

    def measure_of(time_ms: float) -> int:
        return int((time_ms - offset_ms) // measure_length_ms)

    result: list[HitObject] = []
    i = 0
    n = len(objects)
    while i < n:
        obj = objects[i]
        has_next = i + 1 < n
        can_merge = (has_next and not obj.is_slider and not objects[i + 1].is_slider
                     and (objects[i + 1].time - obj.time) <= threshold
                     and measure_of(obj.time) in repetitive_measures
                     and measure_of(objects[i + 1].time) in repetitive_measures)
        if can_merge:
            nxt = objects[i + 1]
            duration_ms = nxt.time - obj.time
            length = slider_multiplier * 100.0 * (duration_ms / beat_length_ms)
            result.append(HitObject(
                x=obj.x, y=obj.y, time=obj.time, is_new_combo=obj.is_new_combo, hitsound=obj.hitsound,
                is_slider=True, curve_type="L", points=[(nxt.x, nxt.y)], slides=1, length=length,
            ))
            i += 2
        else:
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
    args = parser.parse_args()

    bm = read_osu(args.beatmap)
    bm.metadata["Version"] = args.version
    beat_length_ms = bm.beat_length
    slider_multiplier = bm.slider_multiplier
    offset_ms = bm.offset
    measure_length_ms = beat_length_ms * 4.0

    objects = sorted(bm.hit_objects, key=lambda h: h.time)
    if not objects:
        raise RuntimeError("Beatmap has no hit objects to simplify.")

    print("Analyzing song structure...")
    energy_at = compute_energy_lookup(args.audio)
    measure_buckets = compute_measure_energy_buckets(energy_at, offset_ms, measure_length_ms, objects[-1].time)
    repetitive_measures = find_repetitive_measures(measure_buckets)
    print(f"  {len(repetitive_measures)}/{len(measure_buckets)} measures are in a repeating section")

    before = len(objects)
    objects = thin_repetitive_streams(objects, beat_length_ms, slider_multiplier, offset_ms,
                                       measure_length_ms, repetitive_measures)
    print(f"Thinned {before - len(objects)} object(s) by merging stream pairs into sliders")

    regularize_hitsounds(objects, offset_ms, measure_length_ms, repetitive_measures)
    recompute_combos(objects, offset_ms, measure_length_ms)

    bm.hit_objects = objects
    bm.difficulty = easier_difficulty(bm.difficulty)

    # Sanity check: thinning must never introduce an overlap.
    for a, b in zip(objects, objects[1:]):
        a_end = a.end_time(beat_length_ms, slider_multiplier)
        if b.time < a_end - 1e-6:
            raise AssertionError(f"Overlap introduced at {a.time:.1f}ms while thinning for Easy mode.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_osu(bm, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
