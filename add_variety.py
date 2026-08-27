#!/usr/bin/env python3
"""
Stage 2 — Add variety.

Takes the plain half-beat circle skeleton from generate_base_beatmap.py and
reshapes it based on the song's loudness (RMS energy) over time:

  * Quiet sections  -> thinned out (down to one object per full beat).
  * Normal sections -> runs of adjacent circles are combined into short
                        slider chains (2-4 notes per slider), so normal
                        sections read mostly as sliders rather than a wall
                        of individually-stacked circles. A few circles are
                        left standalone for rhythmic variety.
  * Intense sections -> extra circles are inserted on quarter- and
                        eighth-beat subdivisions.

The one hard rule throughout: no two hittable objects may occupy overlapping
time. A slider "occupies" the timeline for its full duration, so nothing is
ever placed while a slider is still being held. Subdivision timestamps are
computed by dividing each interval into an exact whole number of equal
steps (rather than repeatedly adding a fixed subdivision length), so
floating-point drift can never leave two objects a fraction of a
millisecond apart — which would otherwise round to the *same* millisecond
when written out and become unplayable simultaneous notes.

Usage:
    python3 add_variety.py base.osu song.mp3 --output out/song_variety.osu
"""

from __future__ import annotations

import argparse
import os

import librosa
import numpy as np

from beatmap_utils import HitObject, read_osu, write_osu


# --- energy analysis --------------------------------------------------------

def compute_energy_curve(audio_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_ms, normalized_rms) for the whole track."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length) * 1000.0
    rms = rms / (rms.max() + 1e-9)
    return times, rms


def make_energy_lookup(times_ms: np.ndarray, energy: np.ndarray):
    def energy_at(t_ms: float) -> float:
        return float(np.interp(t_ms, times_ms, energy))
    return energy_at


def classify(energy_value: float, q_low: float, q_high: float) -> str:
    if energy_value < q_low:
        return "quiet"
    if energy_value > q_high:
        return "intense"
    return "normal"


# --- slider construction -----------------------------------------------------

def make_slider_chain(nodes: list[HitObject], beat_length_ms: float, slider_multiplier: float) -> HitObject:
    """Combine consecutive circles into a single multi-anchor ("chain") slider.

    `nodes` is 2+ circles in time order; the slider starts at the first and
    passes through every subsequent one as a straight-line waypoint, held
    for exactly the time span from the first to the last node.
    """
    start, rest = nodes[0], nodes[1:]
    duration_ms = nodes[-1].time - start.time
    px_per_beat = slider_multiplier * 100.0
    length = px_per_beat * (duration_ms / beat_length_ms)
    return HitObject(
        x=start.x, y=start.y, time=start.time, is_new_combo=start.is_new_combo,
        is_slider=True, curve_type="L", points=[(n.x, n.y) for n in rest], slides=1, length=length,
    )


def interpolate_point(a: HitObject, b: HitObject, t: float) -> tuple[int, int]:
    """Point at fraction t (0..1) of the way from a to b — used for inserted subdivisions."""
    x = a.x + (b.x - a.x) * t
    y = a.y + (b.y - a.y) * t
    return int(round(x)), int(round(y))


def main() -> None:
    parser = argparse.ArgumentParser(description="Add sliders/density variation to a base beatmap based on song energy.")
    parser.add_argument("beatmap", help="Path to the base .osu file (from generate_base_beatmap.py).")
    parser.add_argument("audio", help="Path to the same song's MP3 (used for energy analysis).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", default="Auto Variety", help="Difficulty/version name to write into the map.")
    parser.add_argument("--quiet-quantile", type=float, default=0.35)
    parser.add_argument("--intense-quantile", type=float, default=0.75)
    parser.add_argument("--climax-quantile", type=float, default=0.92)
    args = parser.parse_args()

    bm = read_osu(args.beatmap)
    bm.metadata["Version"] = args.version
    beat_length_ms = bm.beat_length
    slider_multiplier = bm.slider_multiplier
    half_beat_ms = beat_length_ms / 2.0
    quarter_beat_ms = beat_length_ms / 4.0
    eighth_beat_ms = beat_length_ms / 8.0

    circles = sorted([h for h in bm.hit_objects if not h.is_slider], key=lambda h: h.time)
    if len(circles) < 2:
        raise RuntimeError("Base beatmap needs at least two circles to add variety to.")

    print("Analyzing song energy...")
    times_ms, energy = compute_energy_curve(args.audio)
    energy_at = make_energy_lookup(times_ms, energy)

    slot_energy = np.array([energy_at(c.time) for c in circles])
    q_low = float(np.quantile(slot_energy, args.quiet_quantile))
    q_high = float(np.quantile(slot_energy, args.intense_quantile))
    q_climax = float(np.quantile(slot_energy, args.climax_quantile))
    print(f"  energy quantiles -> quiet<{q_low:.3f}  intense>{q_high:.3f}  climax>{q_climax:.3f}")

    categories = [classify(e, q_low, q_high) for e in slot_energy]

    new_objects: list[HitObject] = []
    i = 0
    n = len(circles)
    chain_counter = 0  # varies chain length / occasionally skips merging, for variety

    while i < n:
        cat = categories[i]
        cur = circles[i]
        has_next = i + 1 < n

        if cat == "quiet":
            # Thin quiet sections down to one object per full beat: keep this
            # circle, and drop the following half-beat circle if there is one.
            new_objects.append(cur)
            i += 2 if has_next else 1
            continue

        if cat == "normal":
            # Pair this circle with the next one into a short (single-segment)
            # slider whenever possible. Doing this for most eligible pairs in
            # a row is what produces a visible *chain* of short sliders back
            # to back, rather than a wall of individually-stacked circles.
            # Every so often a pair is left as plain circles so the section
            # still breathes and doesn't turn into an unbroken slider train.
            chain_counter += 1
            can_chain = has_next and categories[i + 1] != "intense" and (chain_counter % 4 != 0)
            if can_chain:
                nxt = circles[i + 1]
                new_objects.append(make_slider_chain([cur, nxt], beat_length_ms, slider_multiplier))
                i += 2
            else:
                new_objects.append(cur)
                i += 1
            continue

        # cat == "intense": keep the circle and pack in subdivisions up to the
        # next existing object, never overlapping it. The interval is split
        # into an exact whole number of equal steps so no inserted timestamp
        # can land a fraction of a millisecond from the next object (which
        # would round to the same millisecond on disk and be unplayable).
        new_objects.append(cur)
        next_time = circles[i + 1].time if has_next else cur.time + half_beat_ms
        next_obj = circles[i + 1] if has_next else cur

        subdivision = eighth_beat_ms if slot_energy[i] > q_climax else quarter_beat_ms
        interval = next_time - cur.time
        num_steps = max(1, round(interval / subdivision))
        step = interval / num_steps
        for k in range(1, num_steps):
            t = cur.time + k * step
            frac = k / num_steps
            x, y = interpolate_point(cur, next_obj, frac)
            new_objects.append(HitObject(x=x, y=y, time=t, is_new_combo=False))
        i += 1

    # Recompute new-combo flags cleanly (every 8 objects) now that the object
    # list has been reshaped.
    new_objects.sort(key=lambda h: h.time)
    for idx, obj in enumerate(new_objects):
        obj.is_new_combo = (idx % 8 == 0)

    # Sanity check: nothing should overlap in time.
    for a, b in zip(new_objects, new_objects[1:]):
        a_end = a.end_time(beat_length_ms, slider_multiplier)
        if b.time < a_end - 1e-6:
            raise AssertionError(f"Overlap detected: object at {a.time:.1f}ms ends at {a_end:.1f}ms, "
                                  f"but next object starts at {b.time:.1f}ms")

    bm.hit_objects = new_objects
    print(f"{len(circles)} base circles -> {len(new_objects)} objects "
          f"({sum(1 for o in new_objects if o.is_slider)} sliders)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_osu(bm, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
