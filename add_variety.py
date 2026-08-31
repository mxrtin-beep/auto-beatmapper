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

from apply_style import compute_measure_energy_buckets
from beatmap_utils import HitObject, read_osu, slider_length_for_gap, write_osu
from pattern_uniformity import blended_choice, fuzzy_repeat_map

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



def assign_hitsounds(objects: list[HitObject], energy_at, offset_ms: float, measure_length_ms: float,
                      q_high: float, q_climax: float, measure_repeat_map: dict[int, int] | None = None) -> None:
    """Assign every object's hitsound (and, for sliders, edge_hitsounds) from
    local loudness and downbeat position, mutating `objects` in place.
    Shared between add_variety.py's own pipeline and add_sliders_v2.py (the
    Base Map v2 pathway) — a map with every object left on the default
    "normal" sample reads as broken/unfinished to any checker.

    Decided once per whole beat, not once per object: checking the
    reference set (example/keha_backstabber/) found every hitsound change
    lines up with a whole- or half-beat position, never switching between
    two objects that share the same whole beat, and a real accent (clap/
    finish) tends to land on one consistent beat of the bar (e.g. the
    backbeat) rather than flickering note to note the way sampling energy
    per-object could when it hovers right at a quantile threshold. All
    objects within the same beat share one hitsound, decided from that
    beat's own energy (sampled at its start) and whether it's a downbeat.

    `measure_repeat_map` (see find_repeating_measure_map) is optional but
    strongly recommended — without it, a verse's second pass gets its own
    hitsounds decided independently from its own (very similar, but not
    identical) energy, which drifts from the first pass's choices exactly
    where the reference set stays consistent. When given, a beat whose
    measure repeats an earlier one just copies that earlier measure's own
    corresponding beat, if it made one — a section's own accent pattern
    replaying, not a coincidence.

    A bouncing slider only accents its head — repeating the same clap/
    finish on every one of a dozen rapid reversals is jarring rather than
    emphatic, so its repeats and tail stay a plain normal sample instead.
    A long quiet/normal stretch can otherwise go many measures with every
    hit landing on plain HS_NORMAL, which itself reads as "no hitsounds"
    to a checker — at least a soft whistle is forced often enough that
    never happens, even where the energy alone wouldn't have earned one.
    """
    beat_length_ms = measure_length_ms / 4.0
    MAX_MS_WITHOUT_ACCENT = measure_length_ms
    last_accent_time = None
    beat_hitsound: dict[int, int] = {}
    for obj in objects:
        beat_idx = int(round((obj.time - offset_ms) / beat_length_ms))
        if beat_idx not in beat_hitsound:
            beat_time = offset_ms + beat_idx * beat_length_ms

            canonical_beat_idx = None
            if measure_repeat_map is not None:
                measure_idx, beat_in_measure = divmod(beat_idx, 4)
                canonical_measure = measure_repeat_map.get(measure_idx, measure_idx)
                if canonical_measure != measure_idx:
                    canonical_beat_idx = canonical_measure * 4 + beat_in_measure

            if canonical_beat_idx is not None and canonical_beat_idx in beat_hitsound:
                # This measure repeats an earlier one, and that earlier
                # measure's own corresponding beat already had an object
                # (and so a hitsound decided) -- reuse it verbatim, rather
                # than re-deriving independently from this pass's own
                # (similar but not identical) energy.
                hs = beat_hitsound[canonical_beat_idx]
            else:
                e = energy_at(beat_time)
                on_downbeat = is_on_downbeat(beat_time, offset_ms, measure_length_ms)
                hs = hitsound_for(e, on_downbeat, q_high, q_climax)
                if hs == HS_NORMAL and (last_accent_time is None
                                         or beat_time - last_accent_time > MAX_MS_WITHOUT_ACCENT):
                    hs = HS_WHISTLE
            beat_hitsound[beat_idx] = hs
            if hs != HS_NORMAL:
                last_accent_time = beat_time
        hs = beat_hitsound[beat_idx]
        obj.hitsound = hs
        if obj.is_slider:
            if obj.slides > 1:
                obj.edge_hitsounds = [hs] + [HS_NORMAL] * obj.slides
            else:
                obj.edge_hitsounds = [hs] * (obj.slides + 1)


def chain_len_weights(bias: float) -> tuple[float, float, float]:
    """Weights for chain lengths (2, 3, 4 nodes), interpolated so bias=0.5
    reproduces the original fixed (50, 30, 20) split exactly (keeping the
    default behavior identical), tilting toward (2) below that and (4)
    above it. Shared with add_sliders_v2.py, which uses the exact same
    length-bias idea for the Base Map v2 pathway's own slider merging."""
    SHORT, MID, LONG = (70.0, 22.0, 8.0), (50.0, 30.0, 20.0), (15.0, 30.0, 55.0)
    a, b, t = (SHORT, MID, bias / 0.5) if bias <= 0.5 else (MID, LONG, (bias - 0.5) / 0.5)
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def cap_stream_length(objects: list[HitObject], beat_length_ms: float, slider_multiplier: float,
                       quarter_beat_ms: float, max_len: int = 8) -> list[HitObject]:
    """Guarantee no run of quarter/eighth-beat circles (a "stream") is longer than max_len.

    A stream is a run of consecutive circles a quarter beat or less apart —
    a wider gap resets the count, since that's an ordinary paced circle,
    not a rapid subdivision. Once a run reaches max_len, the (max_len+1)'th
    and (max_len+2)'th circles are *combined* into one two-node slider —
    the same "merge circles into a slider" operation make_slider_chain
    does — rather than turning only the first of the pair into a slider
    that merely reaches the second one's timestamp while leaving that
    second circle in place as its own object: that used to leave a circle
    and a slider's tail both demanding input at the exact same instant,
    which is illegal regardless of how precisely the timing lines up.
    Combining both into one slider removes that second circle from the
    output entirely, so there is nothing left at that timestamp to
    conflict with the slider's end.

    This is a backstop, not the primary mechanism (that's the chunk-type
    alternation in main()) — it should rarely fire. If there's no next
    object to combine with (the run runs off the end of the track), or the
    next object is already a slider (which can't be folded into a simple
    two-node chain without discarding its own shape), the run is left
    over-length rather than forcing an unsafe merge — both are rare edge
    cases and neither risks an overlap.
    """
    result: list[HitObject] = []
    consecutive = 0
    i = 0
    n = len(objects)
    while i < n:
        obj = objects[i]
        if obj.is_slider:
            result.append(obj)
            consecutive = 0
            i += 1
            continue

        prev = result[-1] if result else None
        is_stream_note = (prev is not None and not prev.is_slider
                           and (obj.time - prev.time) <= quarter_beat_ms + 1.0)
        consecutive = consecutive + 1 if is_stream_note else 1

        has_next = i + 1 < n
        can_merge = has_next and not objects[i + 1].is_slider
        if consecutive > max_len and can_merge:
            nxt = objects[i + 1]
            result.append(make_slider_chain([obj, nxt], beat_length_ms, slider_multiplier))
            consecutive = 0
            i += 2  # nxt is now the slider's endpoint, not a separate object
            continue

        result.append(obj)
        i += 1
    return result


def carve_mid_breaks(objects: list[HitObject], energy_at, q_low: float, beat_length_ms: float,
                      min_run_ms: float = 16000.0, lead_ms: float | None = None) -> list[HitObject]:
    """Cut an actual break into the middle of any long, uninterrupted quiet
    stretch, instead of clicking through minutes of near-nothing.

    "Quiet" sections are already thinned to one object per beat, but that
    still leaves the player clicking through an extended low-energy stretch
    (a long intro, an ambient breakdown) with nothing much happening — it's
    a better rest for the player, and more true to how the song actually
    ebbs and flows, to drop the middle of a stretch like that entirely and
    let the surrounding lead-in/lead-out (kept, `lead_ms` each) frame a real
    break. build_break_periods() then declares the resulting gap official
    automatically, since it's just an ordinary long silence to that pass.

    Never touches anything shorter than `min_run_ms` — a normal-length
    quiet passage (a verse's held notes, a brief comedown) is left exactly
    as thinned; this only fires for stretches long enough that a real break
    reads as more musical than more of the same thinned-out clicking.
    """
    if lead_ms is None:
        lead_ms = beat_length_ms * 4.0  # one measure's lead-in/out
    result: list[HitObject] = []
    i = 0
    n = len(objects)
    while i < n:
        if energy_at(objects[i].time) >= q_low:
            result.append(objects[i])
            i += 1
            continue
        j = i
        while j < n and energy_at(objects[j].time) < q_low:
            j += 1
        run = objects[i:j]
        run_duration = run[-1].time - run[0].time
        if run_duration >= min_run_ms:
            run_start, run_end = run[0].time, run[-1].time
            result.extend(o for o in run if (o.time - run_start) <= lead_ms or (run_end - o.time) <= lead_ms)
        else:
            result.extend(run)
        i = j
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
    length = slider_length_for_gap(start.time, nodes[-1].time, beat_length_ms, slider_multiplier)
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
    each one is exactly on the beat grid. Rather than multiplying
    `one_way_ms` straight into a pixel length (which would drift off-grid
    by the same fraction-of-a-millisecond rounding described in
    `slider_length_for_gap`), the total span is measured with that helper
    against the *rounded* start/end times, so the slider's declared
    duration reconstructs to exactly the intended whole-millisecond span —
    matching whatever gap the caller already arranged to leave before the
    next real object (see call site) with no risk of eating into it.
    """
    length = slider_length_for_gap(start.time, start.time + num_bounces * one_way_ms,
                                    beat_length_ms, slider_multiplier, slides=num_bounces)
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
    parser.add_argument("--slider-length-bias", type=float, default=0.5,
                         help="Which chain length a merged slider in a normal-energy section tends "
                              "to pick, 0-1 (default 0.5). 2 nodes = 1 beat, 3 = 1.5 beats, 4 = 2 "
                              "beats. 0 skews toward the short 1-beat chain (more, choppier "
                              "sliders); 1 skews toward the long 2-beat chain (fewer, longer "
                              "sliders). Has no effect on --chain-probability (whether a run "
                              "becomes a slider at all) or on intense-section bounce sliders.")
    parser.add_argument("--stream-frequency", type=float, default=0.5,
                         help="How often an intense chunk becomes an actual stream (individually-"
                              "clicked circles on quarter/eighth-note subdivisions) versus a bounce "
                              "slider or a rest (0-1, default 0.5). Also scales the hard cap on how "
                              "many quarter/eighth-spaced circles may ever appear consecutively: 3 "
                              "at 0 (a stream, by definition, needs 4+ — so 0 means a run of "
                              "quarter/eighth-note circles is never long enough to read as a stream "
                              "at all) up to 8 at 1. That cap is enforced globally, at the very end, "
                              "regardless of the choices made per chunk here.")
    parser.add_argument("--category-smoothing-beats", type=float, default=2.0,
                         help="Smooth energy over this many beats before classifying quiet/normal/"
                              "intense, so categories track the song's actual sections instead of "
                              "flickering beat to beat (0 disables smoothing).")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed. Omit for a different map every run; pass a fixed "
                              "value (printed on every run) to reproduce the exact same map later.")
    parser.add_argument("--uniformity", type=float, default=0.0,
                         help="How strongly a returning section (a verse's second repeat, a "
                              "chorus that comes back later) reuses its earlier slider-vs-circle "
                              "choices and hitsounds instead of deciding independently, 0-1 "
                              "(default 0 -- a no-op, today's fully independent behavior plus "
                              "hitsound reuse on only *exact* repeats, unchanged). Above 0, "
                              "sections that only sound similar (not identical) increasingly "
                              "reuse an earlier section's choices too, at the same position in "
                              "the pattern -- not a guarantee, more a family resemblance the "
                              "closer this gets to 1, where even a loose resemblance is enough.")
    args = parser.parse_args()
    args.uniformity = max(0.0, min(1.0, args.uniformity))
    args.stream_frequency = max(0.0, min(1.0, args.stream_frequency))
    # 3 at frequency 0 (never long enough to be a stream, per the 4-note
    # definition), 8 at frequency 1 (matches the pre-existing hard cap).
    stream_max_len = round(3 + args.stream_frequency * 5)
    args.slider_length_bias = max(0.0, min(1.0, args.slider_length_bias))
    chain_weights = chain_len_weights(args.slider_length_bias)

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

    # A rough repeat map, computed early from the base grid's own energy (a
    # close enough stand-in for the final objects' -- energy buckets don't
    # depend on what ends up a circle vs. a slider) so the slider-vs-circle
    # decisions in the main loop below can already lean on it. Empty at
    # --uniformity 0's default: no repeat is trusted here beyond what a
    # canonical-measure lookup already misses (fuzzy_repeat_map's own
    # window=4 exact-match floor still applies once uniformity > 0).
    early_measure_buckets = compute_measure_energy_buckets(energy_at, offset_ms, measure_length_ms,
                                                            circles[-1].time)
    measure_repeat_map_early = (fuzzy_repeat_map(early_measure_buckets, args.uniformity, args.seed)
                                 if args.uniformity > 0.0 else {})

    def canonical_measure_for(time_ms: float) -> int | None:
        if not measure_repeat_map_early:
            return None
        measure_idx = int((time_ms - offset_ms) // measure_length_ms)
        return measure_repeat_map_early.get(measure_idx, measure_idx)

    def slot_for(time_ms: float) -> int:
        return int(round((time_ms - offset_ms) / half_beat_ms)) % 8

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
            def draw_chain(chain_rng: random.Random, max_chain: int = max_chain) -> tuple[bool, int]:
                can_chain = max_chain >= 2 and chain_rng.random() < args.chain_probability
                if not can_chain:
                    return False, 0
                chain_len = chain_rng.choices([2, 3, 4][:max_chain - 1], weights=chain_weights[:max_chain - 1])[0]
                return True, chain_len

            can_chain, chain_len = blended_choice(
                args.seed, "chain", canonical_measure_for(cur.time), slot_for(cur.time), args.uniformity,
                group_fn=draw_chain, indep_fn=lambda: draw_chain(rng))
            if can_chain:
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

            # How often a chunk becomes a genuine stream at all is
            # --stream-frequency's job; whatever's left over keeps the
            # original 0.45:0.10 bounce:rest ratio. A short (1-2 slot)
            # burst used to always force the "stream" treatment regardless
            # of this setting — that's exactly the "frequency does
            # nothing" bug: at frequency 0 it kept producing quarter/
            # eighth-note runs anyway. It's now subject to the same
            # frequency-weighted choice as everything else; only the
            # no-repeat-treatment rule is skipped for it (too short to
            # read as a repetitive wall either way).
            p_stream = 0.9 * args.stream_frequency
            p_rest_of = 1.0 - p_stream
            weights = {"stream": p_stream, "bounce": p_rest_of * 0.818, "rest": p_rest_of * 0.182}
            if lookahead_len < 3:
                options = ["stream", "bounce", "rest"]
            else:
                options = [t for t in ("stream", "bounce", "rest") if t != last_treatment]
            def draw_treatment(t_rng: random.Random, options: list[str] = options) -> str:
                return t_rng.choices(options, weights=[weights[t] for t in options])[0]

            pos_time = circles[pos].time
            treatment = blended_choice(
                args.seed, "treatment", canonical_measure_for(pos_time), slot_for(pos_time), args.uniformity,
                group_fn=draw_treatment, indep_fn=lambda: draw_treatment(rng))
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
                max_slots_for_cap = max(1, stream_max_len // steps_per_slot)
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
    # than stream_max_len (3-8, from --stream-frequency) — the chunk-level
    # bounce/circle decisions above already aim for this, but this is the
    # hard backstop regardless of how those rolls landed, and regardless of
    # a run built from several adjacent chunks: the (max_len+1)'th circle
    # in a row always becomes a slider.
    before_cap = sum(1 for o in new_objects if not o.is_slider)
    new_objects = cap_stream_length(new_objects, beat_length_ms, slider_multiplier, quarter_beat_ms,
                                     max_len=stream_max_len)
    after_cap = sum(1 for o in new_objects if not o.is_slider)
    if before_cap != after_cap:
        print(f"Capped long streams: converted {before_cap - after_cap} circle(s) into sliders")

    # Cut a real break into any long, uninterrupted quiet stretch (a long
    # intro, an ambient breakdown) instead of clicking through minutes of
    # near-nothing — build_break_periods() below declares the resulting gap
    # official automatically.
    before_breaks = len(new_objects)
    new_objects = carve_mid_breaks(new_objects, energy_at, q_low, beat_length_ms)
    if len(new_objects) != before_breaks:
        print(f"Carved a mid-track break out of {before_breaks - len(new_objects)} object(s) "
              f"in long quiet stretch(es)")

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
    # normal sample. measure_repeat_map lets a verse/chorus's second pass
    # copy its first pass's own accent pattern (see find_repeating_measure_
    # map and assign_hitsounds' own docstring) rather than re-deriving
    # independently from that pass's own, merely similar energy.
    measure_buckets = compute_measure_energy_buckets(energy_at, offset_ms, measure_length_ms,
                                                       new_objects[-1].time if new_objects else 0.0)
    measure_repeat_map = fuzzy_repeat_map(measure_buckets, args.uniformity, args.seed)
    assign_hitsounds(new_objects, energy_at, offset_ms, measure_length_ms, q_high, q_climax,
                      measure_repeat_map=measure_repeat_map)

    # Sanity check: nothing should overlap in time, judged the same way the
    # .osu file itself will be read back (every object's time rounded to a
    # whole millisecond) — not raw floats, where a slider whose length was
    # deliberately built from a *rounded* gap (see slider_length_for_gap)
    # can land a fraction of a millisecond past an unrounded next-object
    # time while still reconstructing to the exact same on-disk instant.
    for a, b in zip(new_objects, new_objects[1:]):
        a_end = round(a.time) + a.duration_ms(beat_length_ms, slider_multiplier)
        if round(b.time) < a_end - 1e-6:
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
