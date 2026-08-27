#!/usr/bin/env python3
"""
Stage 2 — Add variety.

Takes the plain half-beat circle skeleton from generate_base_beatmap.py and
reshapes it based on the song's loudness (RMS energy) over time:

  * Quiet sections   -> thinned out (down to one object per full beat).
  * Normal sections  -> runs of adjacent circles are combined into short
                         slider chains, so normal sections read mostly as
                         sliders rather than a wall of individually-stacked
                         circles. A few circles are left standalone for
                         rhythmic variety.
  * Intense sections -> a short burst (1-2 consecutive intense half-beats,
                         the "triplet" feel) stays as plain circles with
                         inserted subdivisions, but a longer run of intense
                         beats (3+ in a row) becomes a single slider that
                         bounces back and forth across the whole run instead
                         of a long wall of individually-clicked circles —
                         the same visual intensity with far less click
                         density.

A small fraction of otherwise-eligible objects are dropped entirely as
short rests, so sections get a breath instead of being wall-to-wall notes.

New combos are aligned to the song's actual downbeats (every 4 beats from
the detected offset) rather than a fixed object count, so combo colors
don't drift onto arbitrary off-beats as objects get merged/dropped/added.

Hitsounds are assigned from local energy and downbeat position (bigger
accents — finishes/claps — line up with strong hits), and PreviewTime is
set to the loudest sustained stretch of the track.

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
import random

import librosa
import numpy as np

from beatmap_utils import HitObject, read_osu, write_osu

# Hitsound bit flags (osu! HitObject hitSound field / slider edgeHitsounds).
HS_NORMAL = 0
HS_WHISTLE = 2
HS_FINISH = 4
HS_CLAP = 8


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


def smooth_slot_energy(values: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smoothing over consecutive beat slots.

    Raw per-slot energy is noisy enough that it flickers between tiers
    almost every other half-beat, even in the middle of an objectively
    loud or quiet section — which fragments what should be one coherent
    run of, say, intense beats into dozens of isolated singles. Smoothing
    over a couple of beats first makes the resulting categories track the
    song's actual sections (verse/chorus/breakdown) instead of that noise.
    """
    if window <= 1 or len(values) < 2:
        return values
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def classify(energy_value: float, q_low: float, q_high: float) -> str:
    if energy_value < q_low:
        return "quiet"
    if energy_value > q_high:
        return "intense"
    return "normal"


def compute_preview_time_ms(times_ms: np.ndarray, energy: np.ndarray, window_ms: float = 5000.0) -> float:
    """Pick the start of the loudest sustained stretch of the track, for PreviewTime."""
    if len(times_ms) < 2:
        return 0.0
    dt_ms = float(np.median(np.diff(times_ms)))
    window_frames = max(1, int(round(window_ms / dt_ms)))
    kernel = np.ones(window_frames) / window_frames
    smoothed = np.convolve(energy, kernel, mode="same")
    peak_idx = int(np.argmax(smoothed))
    return float(times_ms[peak_idx])


# --- downbeat / combo helpers ------------------------------------------------

def is_near_multiple(time_ms: float, offset_ms: float, period_ms: float, tolerance_ms: float = 1.0) -> bool:
    """Whether time_ms lands on a multiple of period_ms from offset_ms."""
    rel = (time_ms - offset_ms) % period_ms
    return min(rel, period_ms - rel) < tolerance_ms


def is_on_downbeat(time_ms: float, offset_ms: float, measure_length_ms: float, tolerance_ms: float = 1.0) -> bool:
    """Whether time_ms lands on a measure boundary (assumes 4/4 time)."""
    return is_near_multiple(time_ms, offset_ms, measure_length_ms, tolerance_ms)


def find_track_end_ms(times_ms: np.ndarray, energy: np.ndarray, floor: float = 0.08) -> float:
    """The last moment the track is actually audible, for trimming a trailing fade-out.

    Looks only at the very end of the track: the last index where energy
    still exceeds `floor` (a small fraction of the track's peak loudness,
    since energy is normalized 0-1 by the max). Everything after that is
    presumed to be a fade-out or trailing silence, which shouldn't have
    hittable objects sitting on screen with nothing audible happening.
    """
    above = np.where(energy > floor)[0]
    if len(above) == 0:
        return float(times_ms[-1])
    return float(times_ms[above[-1]])


def hitsound_for(energy_value: float, is_downbeat: bool, q_high: float, q_climax: float) -> int:
    """Pick a hitsound accent from local loudness and beat position."""
    if is_downbeat and energy_value > q_high:
        return HS_FINISH
    if energy_value > q_climax:
        return HS_CLAP
    if energy_value > q_high:
        return HS_WHISTLE
    return HS_NORMAL


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


def make_bounce_slider(start: HitObject, end: HitObject, beat_length_ms: float, slider_multiplier: float,
                        num_bounces: int, total_duration_ms: float, end_buffer_ms: float = 0.0) -> HitObject:
    """A slider that repeats back and forth `num_bounces` times over total_duration_ms.

    This is the "chain of short sliders" idea taken further: instead of
    several separate slider objects, one slider with repeats reads as the
    same rapid back-and-forth motion but only needs a single click to
    start, dramatically cutting required inputs versus a wall of circles
    while keeping the same visual energy.

    `end_buffer_ms`, if given, shortens the slider so it finishes that much
    before total_duration_ms would otherwise put it — used when
    total_duration_ms was measured up to the very next object's start time,
    so the slider's end never lands exactly on top of it.
    """
    px_per_beat = slider_multiplier * 100.0
    playable_duration_ms = max(total_duration_ms - end_buffer_ms, total_duration_ms * 0.5)
    one_way_ms = playable_duration_ms / num_bounces
    length = px_per_beat * (one_way_ms / beat_length_ms)
    return HitObject(
        x=start.x, y=start.y, time=start.time, is_new_combo=start.is_new_combo,
        is_slider=True, curve_type="L", points=[(end.x, end.y)], slides=num_bounces, length=length,
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
    parser.add_argument("--chain-probability", type=float, default=0.75,
                         help="Chance an eligible normal-energy pair is merged into a slider (0-1).")
    parser.add_argument("--bounce-probability", type=float, default=0.6,
                         help="Chance a dense intense-section burst (4+ notes) becomes a single "
                              "back-and-forth slider instead of a run of circles (0-1).")
    parser.add_argument("--rest-probability", type=float, default=0.03,
                         help="Chance any non-quiet beat is dropped entirely as a short rest (0-1).")
    parser.add_argument("--category-smoothing-beats", type=float, default=2.0,
                         help="Smooth energy over this many beats before classifying quiet/normal/"
                              "intense, so categories track the song's actual sections instead of "
                              "flickering beat to beat (0 disables smoothing).")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed. Omit for a different map every run; pass a fixed "
                              "value (printed on every run) to reproduce the exact same map later.")
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
    half_beat_ms = beat_length_ms / 2.0
    quarter_beat_ms = beat_length_ms / 4.0
    eighth_beat_ms = beat_length_ms / 8.0
    measure_length_ms = beat_length_ms * 4.0  # assumes 4/4, matching the timing point's meter

    circles = sorted([h for h in bm.hit_objects if not h.is_slider], key=lambda h: h.time)
    if len(circles) < 2:
        raise RuntimeError("Base beatmap needs at least two circles to add variety to.")

    print("Analyzing song energy...")
    times_ms, energy = compute_energy_curve(args.audio)
    energy_at = make_energy_lookup(times_ms, energy)

    slot_energy = np.array([energy_at(c.time) for c in circles])
    # Categories are decided from smoothed energy (raw per-slot energy is
    # noisy enough to flicker between tiers almost every other half-beat,
    # which would fragment one coherent intense section into dozens of
    # isolated 1-2 note bursts); hitsound accents below still use the raw,
    # unsmoothed energy so individual loud hits are still picked out.
    smoothing_window = max(1, round(args.category_smoothing_beats * 2))  # beats -> half-beat slots
    smoothed_energy = smooth_slot_energy(slot_energy, smoothing_window)
    q_low = float(np.quantile(smoothed_energy, args.quiet_quantile))
    q_high = float(np.quantile(smoothed_energy, args.intense_quantile))
    q_climax = float(np.quantile(smoothed_energy, args.climax_quantile))
    print(f"  energy quantiles -> quiet<{q_low:.3f}  intense>{q_high:.3f}  climax>{q_climax:.3f}")

    categories = [classify(e, q_low, q_high) for e in smoothed_energy]

    new_objects: list[HitObject] = []
    i = 0
    n = len(circles)

    while i < n:
        cat = categories[i]
        cur = circles[i]
        has_next = i + 1 < n

        if cat == "quiet":
            # Thin quiet sections down to one object per full beat. This
            # keeps whichever half-beat slot actually lands ON a whole beat
            # (checked against the real beat grid), not just "every other
            # slot by array position" — a quiet section can start on either
            # an on-beat or off-beat half-beat slot, and picking by position
            # alone would sometimes keep the off-beat one instead, making
            # objects land consistently on the "and" of the beat (or even
            # the 3rd beat of the measure) instead of the beat itself.
            if is_near_multiple(cur.time, offset_ms, beat_length_ms):
                new_objects.append(cur)
            i += 1
            continue

        # A short, occasional rest: drop this beat entirely so busy sections
        # get a breath instead of being wall-to-wall notes. Never applied to
        # quiet sections, which are already thinned out above.
        if rng.random() < args.rest_probability:
            i += 1
            continue

        if cat == "normal":
            # Pair this circle with the next one into a short (single-segment)
            # slider whenever possible. Doing this for most eligible pairs in
            # a row is what produces a visible *chain* of short sliders back
            # to back, rather than a wall of individually-stacked circles.
            # Every so often a pair is left as plain circles so the section
            # still breathes and doesn't turn into an unbroken slider train.
            # The choice is randomized (seeded) so re-running the pipeline on
            # the same song doesn't always produce an identical map.
            can_chain = has_next and categories[i + 1] != "intense" and rng.random() < args.chain_probability
            if can_chain:
                nxt = circles[i + 1]
                new_objects.append(make_slider_chain([cur, nxt], beat_length_ms, slider_multiplier))
                i += 2
            else:
                new_objects.append(cur)
                i += 1
            continue

        # cat == "intense": a whole *run* of consecutive intense half-beat
        # slots is considered together, not slot by slot — two adjacent
        # base-grid circles are only half a beat apart, which on its own
        # never produces more than a 2-3 note "triplet" burst no matter how
        # intense the section is. A long intense passage is a run of many
        # such slots back to back; it's walked in short chunks, each
        # independently rolling whether it becomes a bouncing slider or
        # stays as circles/triplets, so a long run comes out *interspersed*
        # — bounce, circles, bounce, triplet — instead of one style for the
        # whole run (or one giant slider).
        run_end = i
        while run_end + 1 < n and categories[run_end + 1] == "intense":
            run_end += 1

        chunk_slots = 4  # half a measure — short enough that a bounce slider doesn't overstay
        pos = i
        while pos <= run_end:
            if rng.random() < args.rest_probability:
                pos += 1
                continue

            chunk_end = min(pos + chunk_slots - 1, run_end)
            chunk_len = chunk_end - pos + 1
            after_chunk = chunk_end + 1
            chunk_end_time = circles[after_chunk].time if after_chunk < n else circles[chunk_end].time + half_beat_ms

            if chunk_len >= 3 and rng.random() < args.bounce_probability:
                # Two bounces per half-beat slot reproduces the same rapid
                # back-and-forth density a wall of quarter/eighth-note
                # circles would have, in a slider that only needs one click.
                total_duration = chunk_end_time - circles[pos].time
                num_bounces = chunk_len * 2
                # A small buffer before the next object's start time so the
                # slider's end is never sitting exactly on top of it — only
                # applied when there *is* a following object (an
                # end-of-track chunk has nothing to leave room for).
                end_buffer = min(60.0, eighth_beat_ms) if after_chunk < n else 0.0
                new_objects.append(make_bounce_slider(circles[pos], circles[chunk_end], beat_length_ms,
                                                        slider_multiplier, num_bounces, total_duration,
                                                        end_buffer_ms=end_buffer))
                pos = after_chunk
                continue

            # This chunk stays as individually-clicked circles/triplets:
            # pack subdivisions into the gap up to the next existing object,
            # never overlapping it. The interval is split into an exact
            # whole number of equal steps so no inserted timestamp can land
            # a fraction of a millisecond from the next object (which would
            # round to the same millisecond on disk and become an
            # unplayable simultaneous note).
            for j in range(pos, chunk_end + 1):
                if rng.random() < args.rest_probability:
                    continue
                cur_j = circles[j]
                has_next_j = j + 1 < n
                next_time_j = circles[j + 1].time if has_next_j else cur_j.time + half_beat_ms
                next_obj_j = circles[j + 1] if has_next_j else cur_j

                subdivision = eighth_beat_ms if slot_energy[j] > q_climax else quarter_beat_ms
                interval = next_time_j - cur_j.time
                num_steps = max(1, round(interval / subdivision))

                new_objects.append(cur_j)
                step = interval / num_steps
                for k in range(1, num_steps):
                    t = cur_j.time + k * step
                    frac = k / num_steps
                    x, y = interpolate_point(cur_j, next_obj_j, frac)
                    new_objects.append(HitObject(x=x, y=y, time=t, is_new_combo=False))
            pos = chunk_end + 1

        i = run_end + 1

    new_objects.sort(key=lambda h: h.time)

    # Drop anything sitting in a trailing fade-out/silence: there's nothing
    # audible left to map to, so objects there would just be sitting on
    # screen with no beat behind them. A one-beat buffer after the last
    # audible moment keeps the very last real hit from being cut off.
    track_end_ms = find_track_end_ms(times_ms, energy) + beat_length_ms
    before_trim = len(new_objects)
    new_objects = [o for o in new_objects if o.time <= track_end_ms]
    trimmed = before_trim - len(new_objects)
    if trimmed:
        print(f"Trimmed {trimmed} object(s) in the trailing fade-out (after {track_end_ms:.0f}ms)")

    # New combos land on the song's actual downbeats (every 4 beats from the
    # detected offset), not a fixed object count — object count drifts as
    # things get merged/dropped/added, which would otherwise make combos (and
    # any combo-aligned patterning downstream) land on an arbitrary beat of
    # the measure instead of consistently the first. If a downbeat's own
    # object got swallowed as a slider waypoint (so nothing starts exactly on
    # it), a combo is forced at the next object instead of going a long
    # stretch with no combo break at all.
    last_combo_time = None
    for obj in new_objects:
        on_downbeat = is_on_downbeat(obj.time, offset_ms, measure_length_ms)
        overdue = last_combo_time is not None and (obj.time - last_combo_time) > measure_length_ms * 2.5
        obj.is_new_combo = on_downbeat or overdue
        if obj.is_new_combo:
            last_combo_time = obj.time

    # Hitsounds: bigger accents (finish/clap/whistle) line up with strong
    # downbeats and louder moments; quieter/off-beat hits stay a plain
    # normal sample. A bouncing slider only accents its head — repeating
    # the same clap/finish on every one of a dozen rapid reversals is
    # jarring rather than emphatic, so its repeats and tail stay a plain
    # normal sample instead.
    for obj in new_objects:
        e = energy_at(obj.time)
        on_downbeat = is_on_downbeat(obj.time, offset_ms, measure_length_ms)
        hs = hitsound_for(e, on_downbeat, q_high, q_climax)
        obj.hitsound = hs
        if obj.is_slider:
            if obj.slides > 1:
                obj.edge_hitsounds = [hs] + [HS_NORMAL] * obj.slides
            else:
                obj.edge_hitsounds = [hs] * (obj.slides + 1)

    # Sanity check: nothing should overlap in time.
    for a, b in zip(new_objects, new_objects[1:]):
        a_end = a.end_time(beat_length_ms, slider_multiplier)
        if b.time < a_end - 1e-6:
            raise AssertionError(f"Overlap detected: object at {a.time:.1f}ms ends at {a_end:.1f}ms, "
                                  f"but next object starts at {b.time:.1f}ms")

    bm.hit_objects = new_objects

    preview_ms = compute_preview_time_ms(times_ms, energy)
    bm.general["PreviewTime"] = str(int(round(preview_ms)))
    print(f"Preview time: {preview_ms:.0f} ms")

    bounces = sum(1 for o in new_objects if o.is_slider and o.slides > 1)
    print(f"{len(circles)} base circles -> {len(new_objects)} objects "
          f"({sum(1 for o in new_objects if o.is_slider)} sliders, {bounces} of them bouncing)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_osu(bm, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
