#!/usr/bin/env python3
"""
Stage 2 — Add variety.

Takes the plain half-beat circle skeleton from generate_base_beatmap.py and
reshapes it based on the song's loudness (RMS energy) over time:

  * Quiet sections   -> thinned out (down to one object per full beat).
  * Normal sections  -> runs of 2-4 adjacent circles (1, 1.5, or 2 beats'
                         worth) are combined into slider chains, so normal
                         sections read mostly as sliders of varying length
                         rather than a wall of individually-stacked circles
                         or all-identical 1-beat sliders. A few circles are
                         left standalone for rhythmic variety.
  * Intense sections -> a short burst (1-2 consecutive intense half-beats,
                         the "triplet" feel) always stays as plain circles
                         with inserted subdivisions. A longer run (3+ in a
                         row) is walked in short chunks, each independently
                         assigned one of three treatments — "stream"
                         (individually-clicked circles/triplets, capped at
                         8), "bounce" (one repeating slider), or "rest"
                         (dropped entirely, a deliberate intensity dip) —
                         with the same treatment never allowed to repeat
                         from one chunk to the next. Without that rule nothing
                         stops two or three consecutive stream chunks from
                         reading as one unbroken 16-24 note wall (each one
                         individually "capped" doesn't help if the next
                         chunk is right back to more circles), and several
                         bounce sliders in a row get just as repetitive.

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


def cap_stream_length(objects: list[HitObject], beat_length_ms: float, slider_multiplier: float,
                       quarter_beat_ms: float, max_len: int = 8) -> list[HitObject]:
    """Guarantee no run of quarter/eighth-beat circles (a "stream") is longer than max_len.

    A stream is a run of consecutive circles a quarter beat or less apart —
    a wider gap resets the count, since that's an ordinary paced circle,
    not a rapid subdivision. The (max_len + 1)'th circle in a row is
    replaced by a short slider instead, so a stream always resolves into a
    slider rather than continuing indefinitely.

    This is a backstop, not the primary mechanism (that's the chunk-type
    alternation in main()) — it should rarely fire, but when it does, the
    replacement slider spans the *exact* gap to the next object (which is
    itself already a clean subdivision, since both objects sit on the same
    grid) rather than some fraction of it. Shrinking it by an arbitrary
    fraction is exactly what previously caused two problems at once: an
    unsnapped slider end (it no longer landed on a clean beat fraction) and
    objects sometimes under 10ms apart (a small subdivision gap shrunk by
    even 10% can leave less than 10ms of clearance).
    """
    result: list[HitObject] = []
    consecutive = 0
    n = len(objects)
    for i, obj in enumerate(objects):
        if obj.is_slider:
            result.append(obj)
            consecutive = 0
            continue

        prev = result[-1] if result else None
        is_stream_note = (prev is not None and not prev.is_slider
                           and (obj.time - prev.time) <= quarter_beat_ms + 1.0)
        consecutive = consecutive + 1 if is_stream_note else 1

        if consecutive > max_len:
            has_next = i + 1 < n
            one_way_ms = (objects[i + 1].time - obj.time) if has_next else quarter_beat_ms
            # Both this slider's start time and the next object's time get
            # independently rounded to the nearest millisecond when written
            # out, while `length` (and so the reconstructed duration) is
            # derived from the exact, unrounded gap — after that independent
            # rounding on both ends, the reconstructed end can land a
            # fraction of a millisecond past the next object's now-rounded
            # start. A 1ms margin easily absorbs that without being a
            # perceptible or "unsnapped" amount.
            one_way_ms = max(1.0, one_way_ms - 1.0)
            px_per_beat = slider_multiplier * 100.0
            length = px_per_beat * (one_way_ms / beat_length_ms)
            result.append(HitObject(
                x=obj.x, y=obj.y, time=obj.time, is_new_combo=obj.is_new_combo,
                is_slider=True, curve_type="L", points=[(obj.x + 1, obj.y)], slides=1, length=length,
            ))
            consecutive = 0
        else:
            result.append(obj)
    return result


def build_break_periods(objects: list[HitObject], beat_length_ms: float, slider_multiplier: float,
                         min_gap_ms: float = 4000.0, edge_buffer_ms: float = 200.0) -> list[str]:
    """[Events] "Break Periods" lines for any long stretch with no hit objects.

    Without an explicit break, a long instrumental/silent stretch just
    looks like a mapper forgot to add hitsounds or objects there; declaring
    it as a real break tells both the game and any checker that the gap is
    intentional.
    """
    breaks = []
    for a, b in zip(objects, objects[1:]):
        end = a.end_time(beat_length_ms, slider_multiplier)
        gap = b.time - end
        if gap < min_gap_ms:
            continue
        start = end + edge_buffer_ms
        stop = b.time - edge_buffer_ms
        if stop - start >= 650.0:  # osu!'s own minimum break length
            breaks.append(f"2,{start:.0f},{stop:.0f}")
    return breaks


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
                        num_bounces: int, one_way_ms: float) -> HitObject:
    """A slider that repeats back and forth `num_bounces` times, each leg exactly one_way_ms long.

    This is the "chain of short sliders" idea taken further: instead of
    several separate slider objects, one slider with repeats reads as the
    same rapid back-and-forth motion but only needs a single click to
    start, dramatically cutting required inputs versus a wall of circles
    while keeping the same visual energy.

    `one_way_ms` must be an exact rhythmic subdivision (e.g. a quarter or
    eighth beat) — every repeat lands at start.time + k*one_way_ms, so
    each one is exactly on the beat grid. Computing the leg length from an
    arbitrary total duration instead (e.g. shortened by some fixed buffer
    to avoid touching the next object) would throw every repeat off-grid
    by that same fraction, which is exactly what triggers an editor's
    "unsnapped repeat" warning — the fix is to keep the leg length clean
    and drop a whole leg to make room for a gap instead (see call site).
    """
    px_per_beat = slider_multiplier * 100.0
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
            # Combine this circle with the next 1-3 into a slider whenever
            # possible — varying the chain length (2, 3, or 4 nodes: 1, 1.5,
            # or 2 beats) is what keeps sliders from all reading as the same
            # fixed length. Doing this for most eligible runs in a row is
            # what produces a visible *chain* of sliders back to back,
            # rather than a wall of individually-stacked circles. Every so
            # often a run is left as plain circles so the section still
            # breathes and doesn't turn into an unbroken slider train. The
            # choice is randomized (seeded) so re-running the pipeline on
            # the same song doesn't always produce an identical map.
            max_chain = 1
            while (i + max_chain < n and max_chain < 4
                   and categories[i + max_chain] != "intense"):
                max_chain += 1
            can_chain = max_chain >= 2 and rng.random() < args.chain_probability
            if can_chain:
                chain_len = rng.choices([2, 3, 4][:max_chain - 1], weights=[50, 30, 20][:max_chain - 1])[0]
                nodes = circles[i:i + chain_len]
                new_objects.append(make_slider_chain(nodes, beat_length_ms, slider_multiplier))
                i += chain_len
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
        # assigned one of three treatments — "stream" (individually-clicked
        # circles/triplets), "bounce" (one repeating slider), or "rest"
        # (dropped entirely, a deliberate breather) — with the same
        # treatment never allowed twice in a row. Without that rule, two or
        # three consecutive stream chunks read as one unbroken 16-24 note
        # wall regardless of each chunk individually being capped at 8, and
        # several bounce sliders back to back get just as repetitive.
        run_end = i
        while run_end + 1 < n and categories[run_end + 1] == "intense":
            run_end += 1

        chunk_slots = 4  # half a measure
        pos = i
        last_treatment = None
        while pos <= run_end:
            if rng.random() < args.rest_probability:
                pos += 1
                continue

            lookahead_end = min(pos + chunk_slots - 1, run_end)
            lookahead_len = lookahead_end - pos + 1

            # A short (1-2 slot) burst is always the "triplet" feel,
            # regardless of what came before — it's too short to read as a
            # repetitive wall on its own. Only chunks long enough to
            # meaningfully become a stream or a bounce slider (3+ slots)
            # are subject to the no-repeat-treatment rule.
            if lookahead_len < 3:
                treatment = "stream"
            else:
                options = [t for t in ("stream", "bounce", "rest") if t != last_treatment]
                weights = {"stream": 0.45, "bounce": 0.45, "rest": 0.10}
                treatment = rng.choices(options, weights=[weights[t] for t in options])[0]
            last_treatment = treatment

            chunk_end = lookahead_end
            if treatment == "stream":
                # A stream's own subdivision rate can pack up to 4 notes
                # into a single half-beat slot (eighth-note rate) — a full
                # 4-slot chunk at that rate is already 16 circles on its
                # own, well past the 8-circle cap, no matter how well the
                # chunk-level alternation above spaces treatments out. Bound
                # how many slots *this* stream actually covers by its own
                # rate up front, rather than letting cap_stream_length chop
                # an oversized stream in half with a bridging slider that
                # doesn't really break anything up.
                chunk_avg_energy = float(np.mean(slot_energy[pos:lookahead_end + 1]))
                steps_per_slot = 4 if chunk_avg_energy > q_climax else 2
                max_slots_for_cap = max(1, 8 // steps_per_slot)
                chunk_end = min(lookahead_end, pos + max_slots_for_cap - 1)

            chunk_len = chunk_end - pos + 1
            after_chunk = chunk_end + 1
            has_after_chunk = after_chunk < n

            if treatment == "rest":
                # Drop this whole chunk: a deliberate intensity dip instead
                # of forcing every intense half-beat to have something in it.
                pos = after_chunk
                continue

            if treatment == "bounce":
                # One subdivision (quarter or eighth beat, chosen once for
                # the whole chunk so the rate doesn't shift mid-slider) per
                # leg. Two legs per half-beat slot at quarter-beat rate, or
                # four at eighth-beat rate, span the chunk's half-beat slots
                # *exactly* — chunk_len consecutive base-grid slots are
                # always exactly chunk_len half-beats apart — so every
                # repeat lands precisely on the beat grid with no rounding.
                chunk_avg_energy = float(np.mean(slot_energy[pos:chunk_end + 1]))
                one_way_ms = eighth_beat_ms if chunk_avg_energy > q_climax else quarter_beat_ms
                legs_per_half_beat = 4 if one_way_ms == eighth_beat_ms else 2
                full_bounces = chunk_len * legs_per_half_beat

                # Rather than shrinking each leg to leave a gap before the
                # next object (which would throw every repeat off the beat
                # grid — exactly what triggers an "unsnapped repeat"
                # warning), drop the *last whole leg* instead: the gap this
                # leaves is itself one clean subdivision long.
                num_bounces = max(1, full_bounces - 1) if has_after_chunk else full_bounces
                new_objects.append(make_bounce_slider(circles[pos], circles[chunk_end], beat_length_ms,
                                                        slider_multiplier, num_bounces, one_way_ms))
                pos = after_chunk
                continue

            # treatment == "stream": individually-clicked circles/triplets,
            # all at one subdivision rate (decided once for the whole
            # chunk, the same way as the bounce branch above) rather than
            # switching between quarter- and eighth-notes slot to slot,
            # which would read as an inconsistent, hard-to-parse stream.
            # Subdivisions are packed into the gap up to the next existing
            # object, never overlapping it — the interval is split into an
            # exact whole number of equal steps so no inserted timestamp can
            # land a fraction of a millisecond from the next object (which
            # would round to the same millisecond on disk and become an
            # unplayable simultaneous note).
            chunk_avg_energy = float(np.mean(slot_energy[pos:chunk_end + 1]))
            subdivision = eighth_beat_ms if chunk_avg_energy > q_climax else quarter_beat_ms
            for j in range(pos, chunk_end + 1):
                if rng.random() < args.rest_probability:
                    continue
                cur_j = circles[j]
                has_next_j = j + 1 < n
                next_time_j = circles[j + 1].time if has_next_j else cur_j.time + half_beat_ms
                next_obj_j = circles[j + 1] if has_next_j else cur_j

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

    # Guarantee no run of quarter/eighth-beat circles ("stream") is longer
    # than 8 — the chunk-level bounce/circle decisions above already aim
    # for this, but this is the hard backstop regardless of how those rolls
    # landed: the (max_len+1)'th circle in a row always becomes a slider.
    before_cap = sum(1 for o in new_objects if not o.is_slider)
    new_objects = cap_stream_length(new_objects, beat_length_ms, slider_multiplier, quarter_beat_ms)
    after_cap = sum(1 for o in new_objects if not o.is_slider)
    if before_cap != after_cap:
        print(f"Capped long streams: converted {before_cap - after_cap} circle(s) into sliders")

    # New combos land on the song's actual downbeats (every 4 beats from the
    # detected offset), not a fixed object count — object count drifts as
    # things get merged/dropped/added, which would otherwise make combos (and
    # any combo-aligned patterning downstream) land on an arbitrary beat of
    # the measure instead of consistently the first. If a downbeat's own
    # object got swallowed as a slider waypoint (so nothing starts exactly on
    # it), a combo is forced at the next object instead of going a long
    # stretch with no combo break at all. Either way, a combo is also never
    # allowed to run past 8 objects — a section with extra subdivisions
    # inserted (a stream can have several notes per beat) could otherwise
    # rack up many more than 8 objects before the next downbeat arrives.
    MAX_COMBO_LENGTH = 8
    last_combo_time = None
    combo_count = 0
    for obj in new_objects:
        on_downbeat = is_on_downbeat(obj.time, offset_ms, measure_length_ms)
        overdue_time = last_combo_time is not None and (obj.time - last_combo_time) > measure_length_ms * 2.5
        overdue_count = combo_count >= MAX_COMBO_LENGTH
        obj.is_new_combo = on_downbeat or overdue_time or overdue_count
        if obj.is_new_combo:
            last_combo_time = obj.time
            combo_count = 1
        else:
            combo_count += 1

    # Hitsounds: bigger accents (finish/clap/whistle) line up with strong
    # downbeats and louder moments; quieter/off-beat hits stay a plain
    # normal sample. A bouncing slider only accents its head — repeating
    # the same clap/finish on every one of a dozen rapid reversals is
    # jarring rather than emphatic, so its repeats and tail stay a plain
    # normal sample instead.
    MAX_MS_WITHOUT_ACCENT = measure_length_ms
    last_accent_time = None
    for obj in new_objects:
        e = energy_at(obj.time)
        on_downbeat = is_on_downbeat(obj.time, offset_ms, measure_length_ms)
        hs = hitsound_for(e, on_downbeat, q_high, q_climax)
        # A long quiet/normal stretch can otherwise go many measures with
        # every hit landing on plain HS_NORMAL, which reads as "no
        # hitsounds" to any checker — force at least a soft whistle often
        # enough that never happens, even where the energy alone wouldn't
        # have earned one.
        if hs == HS_NORMAL and (last_accent_time is None
                                 or obj.time - last_accent_time > MAX_MS_WITHOUT_ACCENT):
            hs = HS_WHISTLE
        obj.hitsound = hs
        if hs != HS_NORMAL:
            last_accent_time = obj.time
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

    # A long stretch with no hit objects at all (an instrumental break, a
    # long fade before the trimmed-off outro, etc.) otherwise looks
    # unintentional — no hitsounds, nothing happening. Declaring it as a
    # real break period tells the game (and any checker) it's deliberate.
    breaks = build_break_periods(new_objects, beat_length_ms, slider_multiplier)
    if breaks:
        break_index = bm.events.index("//Break Periods") + 1
        bm.events[break_index:break_index] = breaks
        print(f"Added {len(breaks)} break period(s)")

    # PreviewTime is computed once, in generate_base_beatmap.py, and should
    # stay identical across every difficulty in the set (a mismatch reads
    # as a bug to any checker) — only fall back to computing one here if
    # the base beatmap somehow didn't already set it.
    if bm.general.get("PreviewTime", "-1") == "-1":
        preview_ms = compute_preview_time_ms(times_ms, energy)
        bm.general["PreviewTime"] = str(int(round(preview_ms)))
    print(f"Preview time: {bm.general['PreviewTime']} ms")

    bounces = sum(1 for o in new_objects if o.is_slider and o.slides > 1)
    print(f"{len(circles)} base circles -> {len(new_objects)} objects "
          f"({sum(1 for o in new_objects if o.is_slider)} sliders, {bounces} of them bouncing)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_osu(bm, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
