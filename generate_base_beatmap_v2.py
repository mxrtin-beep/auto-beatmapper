#!/usr/bin/env python3
"""
Stage 1, v2 — Intensity-adaptive base beatmap generator.

A second, independent take on Stage 1 (generate_base_beatmap.py), which
always places one plain circle every half beat regardless of the song. This
version instead adapts *how often* a circle lands to the song's own local
loudness (RMS energy, the same signal add_variety.py already uses for its
own quiet/normal/intense split) — the base skeleton itself is denser or
sparser before add_variety.py or apply_style.py ever touch it:

  * Silent/very quiet -> no circles at all (nothing to click to).
  * Quiet             -> one circle per whole beat.
  * Normal            -> one circle per half beat (matches v1's fixed rate).
  * Intense           -> one circle per quarter beat.
  * Climax            -> one circle per eighth beat.

Classification happens once per *whole beat*, not per finer subdivision —
every beat gets exactly one intensity tier from its own (smoothed) energy,
and every circle placed in that beat follows that tier's rate for the
beat's entire span. Deciding it any finer (e.g. per eighth-beat slot) let
one beat's density flicker between subdivisions mid-beat, which read as
haphazard rather than intentional — a circle starting on the weak 2nd or
4th beat about as often as the strong 1st or 3rd, since a slot's own
classification cared nothing about which beat it was actually part of.

A single `--intensity` knob (0-1) governs all four quiet/normal/intense/
climax thresholds at once: low means the map stays whole-beat dominant
throughout, high pushes progressively more of it into half/quarter/
eighth-beat territory. It's not a hard cutoff — every tier stays possible
at any setting, just how *much* of the song reaches each one shifts.

Climax (the densest tier) is a special case, handled by its own pass
(_emit_climax_run) instead of the simple "fill this one beat" rule the
other tiers use: rather than an isolated beat's worth of eighth-beat
circles that can start anywhere in a measure depending on which beat
happened to classify as climax, a run of consecutive climax beats becomes
one *measure-anchored* burst — eighth-beat- (32nd-note-) spaced circles,
one subdivision finer than "intense"'s own quarter-beat rate so climax
actually reads as the fastest thing in the map — starting exactly on the
nearest measure's downbeat and running for 1-4 whole beats (9/17/25/33
circles respectively; a run needing more than a full measure chains into
another burst anchored to the next measure's downbeat). Anchoring to the
downbeat regardless of which specific beat first tipped over into
"climax" is what fixes bursts that otherwise started at an arbitrary,
musically weak point mid-measure.

BPM/offset detection, PreviewTime, and the placeholder Lissajous-curve
positions are unchanged from v1 (reused directly) — this is purely about
*when* circles land, the same "timing only, no visual polish yet" scope
generate_base_beatmap.py itself has. apply_style.py's own positioning
still applies unchanged to whatever a later stage builds on top of this
(see add_sliders_v2.py for the Base Map v2 pathway's own next stage).

A hard cap keeps a run of fast (quarter-beat-or-closer) circles from ever
reading as an unbroken wall: once a fast run's own elapsed span would reach
a full beat, the next circle in it is dropped instead of extending the run
further. This still applies to the quarter-beat ("intense") tier, capping
it at 4 circles in a row (3 gaps = 3/4 beat); climax runs are shaped by
the measure-anchored bursts above instead, which have their own built-in
length limit and never need this cap.

This is not yet wired into main.py's pipeline — it's a standalone
alternative to generate_base_beatmap.py for comparison, the same way
make_easy.py started as an optional add-on stage.

Usage:
    python3 generate_base_beatmap_v2.py song.mp3 --output out/song_base_v2.osu \
        --title "Song Title" --artist "Artist" --creator "Your Name"
"""

from __future__ import annotations

import argparse
import os
import random

import librosa
import numpy as np

from beatmap_utils import HitObject, TimingPoint, default_metadata, write_osu
from add_variety import compute_energy_curve, compute_preview_time_ms, make_energy_lookup, smooth_slot_energy
from generate_base_beatmap import detect_bpm_and_offset, placeholder_positions

# Density tiers, quietest to loudest, and the beat subdivision each one
# gets: 1 = one circle per whole beat, 2 = per half beat, 4 = per quarter
# beat. "silent" has no entry -- it places nothing. "climax" isn't listed
# here at all -- it's handled entirely by _emit_climax_run below instead
# of this simple "fill one beat at a fixed rate" rule (see its own and the
# module docstring -- eighth-beat spacing, one subdivision finer still).
TIER_SUBDIVISION = {"quiet": 1, "normal": 2, "intense": 4}

MEASURE_BEATS = 4  # quarter beats per measure (4/4 time, matching the rest of the codebase)


def classify_intensity(energy_value: float, q_silent: float, q_quiet: float,
                        q_intense: float, q_climax: float) -> str:
    if energy_value < q_silent:
        return "silent"
    if energy_value < q_quiet:
        return "quiet"
    if energy_value < q_intense:
        return "normal"
    if energy_value < q_climax:
        return "intense"
    return "climax"


def intensity_quantiles(intensity: float) -> dict[str, float]:
    """Map a single 0-1 `--intensity` knob onto the four quantile
    thresholds classify_intensity actually uses, interpolated so
    intensity=0.5 reproduces the same tuned defaults this module always
    used (silent<0.10, quiet<0.35, intense>0.90, climax>0.98) — low tilts
    the whole map toward whole-beat dominance (quiet/normal cover nearly
    everything, intense/climax are exceptional), high tilts it toward
    half/quarter/eighth-beat territory (silent/quiet shrink to almost
    nothing, intense/climax cover much more of the song). Every tier
    stays reachable at every setting; only how much of the song lands in
    each one shifts.
    """
    intensity = max(0.0, min(1.0, intensity))
    LOW = {"silent": 0.05, "quiet": 0.95, "intense": 0.999, "climax": 0.9999}
    MID = {"silent": 0.10, "quiet": 0.35, "intense": 0.90, "climax": 0.98}
    HIGH = {"silent": 0.0, "quiet": 0.05, "intense": 0.30, "climax": 0.70}
    a, b, t = (LOW, MID, intensity / 0.5) if intensity <= 0.5 else (MID, HIGH, (intensity - 0.5) / 0.5)
    return {k: a[k] + (b[k] - a[k]) * t for k in LOW}


def _climax_segments(run_start: int, run_end: int) -> list[tuple[int, int]]:
    """List of (measure-downbeat beat-index, span-in-quarter-beats)
    covering the climax run [run_start, run_end] (beat indices, inclusive),
    each segment anchored to its own measure's downbeat and chained
    end-to-end with no gap and no repeated boundary point. `span` is
    always 1-4: the beat offset from that segment's downbeat needed to
    reach `run_end` (or 4, capped, if that would overrun into the next
    measure — the point where a fresh segment anchored to *that* measure's
    own downbeat takes over instead).
    """
    segments = []
    cursor = (run_start // MEASURE_BEATS) * MEASURE_BEATS
    while True:
        span = min(MEASURE_BEATS, max(1, run_end - cursor))
        segments.append((cursor, span))
        end = cursor + span
        if end >= run_end:
            break
        cursor = end
    return segments


def _climax_covered_end(run_start: int, run_end: int) -> int:
    """The last beat index a climax run [run_start, run_end] actually gets
    reshaped through, once anchored to run_start's own measure downbeat
    and rounded up through the canonical 1/2/3/4-quarter-beat span for
    every measure it touches (see _climax_segments and the module
    docstring) — just the final endpoint, not the intermediate chain."""
    seg_start, span = _climax_segments(run_start, run_end)[-1]
    return seg_start + span


def _emit_climax_run(kept_times: list[float], climax_times: set[float], start_beat: int, end_beat: int,
                      offset_seconds: float, beat_seconds: float) -> None:
    """Append one continuous eighth-beat- (32nd-note-) spaced run of
    circles from beat index `start_beat` through `end_beat` (inclusive) —
    climax is the densest tier there is, one subdivision finer than
    "intense"'s own quarter-beat (16th-note) rate, so it actually reads as
    the fastest thing in the map rather than (as an earlier, half-beat-
    spaced version of this did) slower than intense. A climax burst's
    internal measure-to-measure chaining (see _climax_segments) never
    actually changes rate at its own boundaries — every segment is the
    same constant eighth-beat spacing, so the whole covered span is just
    one flat run end to end, nothing measure-shaped about *emitting* it,
    only about how far it was decided to reach.

    Every emitted time is also recorded into `climax_times` — cap_fast_run_
    span must never thin a climax run down (its own quarter-beat-or-closer
    "fast run" cap exists for the incidental, unshaped intense-tier runs
    build_intensity_grid's plain per-beat fill can produce, not for a
    burst deliberately built to span several beats).
    """
    start_ms = (offset_seconds + start_beat * beat_seconds) * 1000.0
    step_ms = beat_seconds * 1000.0 / 8.0
    num_steps = (end_beat - start_beat) * 8
    new_times = [start_ms + k * step_ms for k in range(num_steps + 1)]
    kept_times.extend(new_times)
    climax_times.update(new_times)


def _climax_covered_ranges(tiers: list[str], max_span_beats: float = MEASURE_BEATS) -> list[tuple[int, int]]:
    """Beat-index ranges [start, end] (inclusive) that climax bursts end up
    covering — `start` is always a measure downbeat (a multiple of
    MEASURE_BEATS from the very first beat), regardless of which beat
    within that measure first classified as climax.

    `max_span_beats` caps how far any *one* burst's own coverage can
    reach from its own start — without this, a song staying loud for many
    consecutive measures produced one continuous, ever-growing burst with
    no upper bound at all (the "nonstop" symptom at high --intensity,
    where eighth-beat spacing makes an uncapped burst read as genuinely
    relentless rather than just a longer run). Once a burst hits the cap,
    whatever climax-tier beats follow don't get folded into it or start a
    new burst immediately — scanning resumes a full measure past this
    burst's own end, giving at least one measure of cooldown (ordinary
    per-beat placement, not silence) before a fresh burst can start.

    This has to be computed as a first pass over the *raw* per-beat tiers,
    entirely separate from actually placing anything: a burst's start is
    always snapped back to its measure's downbeat, which can be *earlier*
    than the run that triggered it — placing bursts inline during a single
    left-to-right walk (an earlier version of this did exactly that) means
    beats between that downbeat and the run's real start have often
    already been independently placed under their own (non-climax) tier
    by the time the burst tries to also cover them, producing duplicate or
    overlapping circles at the same instant. Computing every burst's full
    covered range up front, before placing anything, means the main
    placement pass can just skip every beat a burst will cover, once,
    cleanly, at the point it's already known to be covered.
    """
    n = len(tiers)
    covered: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if tiers[i] != "climax":
            i += 1
            continue
        j = i
        while j + 1 < n and tiers[j + 1] == "climax":
            j += 1

        start = (i // MEASURE_BEATS) * MEASURE_BEATS
        capped_run_end = min(j, start + int(max_span_beats) - 1)
        end = _climax_covered_end(start, capped_run_end)
        covered.append((start, end))

        if j > capped_run_end:
            # The underlying climax-tier run kept going past the cap --
            # resume one full measure past this burst's own end, so a
            # fresh burst can't pick right back up on the very next beat.
            i = end + MEASURE_BEATS
        else:
            i = j + 1
    return covered


def build_intensity_grid(offset_seconds: float, bpm: float, duration_seconds: float,
                          energy_at, q_silent: float, q_quiet: float, q_intense: float, q_climax: float,
                          smoothing_beats: float = 2.0, intensity: float = 0.5) -> tuple[list[float], set[float]]:
    """Return hit-object times (ms), one whole beat classified at a time —
    every beat gets exactly one intensity tier, and every circle in that
    beat follows that same tier's subdivision rate for the beat's entire
    span. This is deliberately *not* decided finer than a beat: classifying
    each eighth-beat slot on its own let one beat's density flicker between
    subdivisions mid-beat as smoothed energy drifted past a threshold
    partway through it, which reads as haphazard — circles starting on the
    weak 2nd/4th beat as often as the strong 1st/3rd, no coherent "this
    beat is busy" feel. One classification per beat instead means an
    intense beat is intense *all the way through*, not just wherever its
    own eighth-slots individually happened to clear the bar.

    "Quiet" places one circle at the beat's own start; "silent" places
    none; "normal"/"intense" each place `TIER_SUBDIVISION[tier]` circles
    evenly spanning the whole beat (half/quarter-beat spacing). "climax"
    is handled separately, up front, by _climax_covered_ranges (see its
    own and the module docstring) — a whole *run* of consecutive climax
    beats becomes one measure-anchored burst, at eighth-beat (32nd-note)
    spacing — the densest tier there is, one subdivision finer than
    "intense" — not each beat independently filling itself in.

    Returns `(times, climax_times)`: every circle's time, and the subset
    of them that belong to a climax burst specifically — the caller must
    keep cap_fast_run_span from ever thinning those down (see
    _emit_climax_run).
    """
    beat_seconds = 60.0 / bpm
    n = int((duration_seconds - offset_seconds) / beat_seconds) + 1
    beat_times_ms = [(offset_seconds + i * beat_seconds) * 1000.0 for i in range(n)]

    raw_energy = np.array([energy_at(t) for t in beat_times_ms])
    smoothing_window = max(1, round(smoothing_beats))  # already one sample per beat
    smoothed = smooth_slot_energy(raw_energy, smoothing_window)
    tiers = [classify_intensity(e, q_silent, q_quiet, q_intense, q_climax) for e in smoothed]

    # One measure at intensity<=0.5, gradually up to two measures by
    # intensity=1.0 -- how far a single climax burst is allowed to span
    # before a cooldown measure is forced (see _climax_covered_ranges).
    max_span_beats = MEASURE_BEATS * (1.0 + max(0.0, (intensity - 0.5) / 0.5))
    covered_ranges = _climax_covered_ranges(tiers, max_span_beats=max_span_beats)
    covered_starts = dict(covered_ranges)
    covered_set = {b for start, end in covered_ranges for b in range(start, end + 1)}

    kept_times: list[float] = []
    climax_times: set[float] = set()
    i = 0
    while i < n:
        if i in covered_starts:
            end = covered_starts[i]
            _emit_climax_run(kept_times, climax_times, i, end, offset_seconds, beat_seconds)
            i = end + 1
            continue
        if i in covered_set:
            i += 1  # consumed by an earlier burst's forced (measure-rounded) span; already emitted
            continue

        # A beat can be raw-classified "climax" and still land here,
        # uncovered by any burst -- the deliberate cooldown gap
        # _climax_covered_ranges leaves after a burst hits its own
        # max_span_beats cap, so a fresh burst can't start on literally
        # the very next beat. TIER_SUBDIVISION has no "climax" entry (it's
        # normally handled entirely by a burst, see above), so treat a
        # cooldown-gap climax beat as "intense" instead -- still busy, just
        # not another full burst right on the last one's heels.
        tier = "intense" if tiers[i] == "climax" else tiers[i]
        if tier != "silent":
            t_ms = beat_times_ms[i]
            subdivision = TIER_SUBDIVISION[tier]
            step_ms = (beat_seconds * 1000.0) / subdivision
            kept_times.extend(t_ms + k * step_ms for k in range(subdivision))
        i += 1
    return kept_times, climax_times


def cap_fast_run_span(times_ms: list[float], beat_length_ms: float, quarter_beat_ms: float,
                       protected_ms: set[float] | None = None) -> list[float]:
    """Drop circles as needed so no run of consecutive quarter-beat-or-
    closer circles ever spans a full beat or more from its own first
    surviving member, leaving a full two-quarter-beat gap (not one) before
    the next run picks back up. Only the "intense" tier (or an
    embellishment chain) can still produce a fast (quarter-beat) run under
    this module's current rules; `protected_ms` (see _emit_climax_run) is
    a climax burst's own times, which must never be thinned here no matter
    how fast their own eighth-beat spacing reads by this function's
    quarter-beat-or-closer definition of "fast" — they're a deliberately
    shaped, measure-anchored run, not the kind of incidental fast stretch
    this cap exists to break up.

    Capping to a *two*-quarter-beat gap instead of one: dropping only the
    single circle that would push the run past a full beat still leaves
    consecutive intense beats reading as one continuous run shifted by a
    single quarter-beat step (four kept, one dropped, four kept starting
    one slot later) — an odd, lopsided shape rather than two clearly
    separate bursts. `force_drop_next` makes the circle right after the
    capped one drop too, unconditionally, so back-to-back intense beats
    read as first/second/third/fourth, [gap], third/fourth/next-first/
    next-second — two whole slots skipped, not one.

    `run_start` is reset to whichever circle was just kept (not left
    pointing at the original run's first member) every time one gets
    dropped: a dropped circle still leaves a real gap behind it, and the
    *next* candidate's fast-or-not status, and how much cap budget is left,
    both need judging from that gap — not from a run_start several
    circles back whose own budget was already exhausted. Without this, a
    single drop could cascade into silently dropping every remaining
    circle in the run, well past what the one-beat cap was ever meant to
    enforce.
    """
    protected_ms = protected_ms or set()
    if not times_ms:
        return times_ms
    result = [times_ms[0]]
    run_start = times_ms[0]
    force_drop_next = False
    for t in times_ms[1:]:
        if t in protected_ms:
            result.append(t)
            run_start = t
            force_drop_next = False
            continue
        if force_drop_next:
            # The second of the two circles being dropped at this run
            # boundary, unconditionally -- see force_drop_next's own
            # comment above. run_start is already correctly set to the
            # last circle actually kept, from when the first one dropped.
            force_drop_next = False
            continue
        prev = result[-1]
        is_fast = (t - prev) <= quarter_beat_ms + 1.0
        if not is_fast:
            result.append(t)
            run_start = t
            continue
        if (t - run_start) < beat_length_ms - 1.0:
            result.append(t)
        else:
            # Dropped -- extending the run would reach a full beat's span
            # from run_start. The *next* candidate should still be judged
            # against a fresh one-beat window, anchored at the last circle
            # actually kept (`prev`, still result[-1] since t was just
            # dropped) rather than the same already-exhausted run_start --
            # otherwise every remaining circle in the run keeps failing
            # the same stale check and the whole rest of it silently
            # vanishes instead of just the excess. The circle right after
            # this one drops too (force_drop_next), for the full
            # two-quarter-beat gap described above.
            run_start = prev
            force_drop_next = True
    return result


def add_embellishment_chains(times_ms: list[float], beat_length_ms: float, intensity: float,
                              rng: random.Random) -> list[float]:
    """Sparingly, turn one ordinary gap between consecutive circles into a
    short chain of 16th notes (quarter-beat spacing) -- 5 or 7 of them,
    always an odd count -- rather than leaving every gap at the plain rate
    its own beat's tier already gave it. Only ever inserted strictly
    *inside* an existing gap that's already roomy enough for the whole
    chain (never touching either endpoint circle, so it can't collide with
    or crowd whatever comes right before/after), which naturally keeps a
    7-note chain (1.5 beats of quarter-beat spacing) rare -- most ordinary
    gaps are only about a beat wide, room enough for a 5-note chain (which
    spans exactly one beat) but not a 7-note one.

    Only fires in the higher-intensity half of the --intensity range, and
    even then on a small, intensity-scaled fraction of eligible gaps --
    "added sparingly", not a wholesale rhythm change.
    """
    if intensity <= 0.5 or len(times_ms) < 2:
        return times_ms
    probability = (intensity - 0.5) * 0.3  # up to ~15% of eligible gaps at intensity=1.0
    quarter_beat_ms = beat_length_ms / 4.0
    result = [times_ms[0]]
    for prev, t in zip(times_ms, times_ms[1:]):
        gap_ms = t - prev
        # Only an ordinary (not already fast) gap is eligible -- embellishing
        # a gap that's already a quarter/eighth-beat run piles a chain onto
        # something already busy, rather than livening up a plain stretch.
        if gap_ms >= beat_length_ms - 1.0 and rng.random() < probability:
            length = rng.choice([5, 7])
            span_ms = (length - 1) * quarter_beat_ms
            if span_ms <= gap_ms - 1.0:
                result.extend(prev + k * quarter_beat_ms for k in range(1, length))
        result.append(t)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an intensity-adaptive base osu! beatmap from an MP3.")
    parser.add_argument("audio", help="Path to the input MP3 file.")
    parser.add_argument("--output", required=True, help="Path to write the .osu file to.")
    parser.add_argument("--title", default=None, help="Song title (defaults to the audio filename).")
    parser.add_argument("--artist", default="Unknown Artist")
    parser.add_argument("--creator", default="auto-beatmapper")
    parser.add_argument("--version", default="Auto Base v2", help="Difficulty/version name.")
    parser.add_argument("--audio-filename", default=None,
                         help="Value written into AudioFilename (defaults to the input file's basename).")
    parser.add_argument("--bpm", type=float, default=None,
                         help="Manually set the BPM instead of auto-detecting it.")
    parser.add_argument("--offset", type=float, default=None,
                         help="Manually set the offset (ms, time of the first beat) instead of "
                              "auto-detecting it. Any value works — it's wrapped to the equivalent "
                              "position within one beat, so e.g. -118 and 334 at 137 BPM name the "
                              "same beat and are interchangeable.")
    parser.add_argument("--intensity", type=float, default=0.5,
                         help="Overall density knob, 0-1 (default 0.5). Low keeps the map whole-beat "
                              "dominant throughout (quiet/normal cover nearly all of it); high pushes "
                              "progressively more of it into half/quarter/eighth-beat territory. Every "
                              "tier stays reachable at any setting -- this shifts how much of the song "
                              "reaches each one, not a hard cutoff. See intensity_quantiles().")
    parser.add_argument("--smoothing-beats", type=float, default=2.0,
                         help="Smooth energy over this many beats before classifying intensity, so "
                              "tiers track the song's actual sections instead of flickering slot to "
                              "slot (default 2.0; 0 disables smoothing).")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed, for embellishment chains (see add_embellishment_chains). "
                              "Omit for a different result every run; pass a fixed value (printed on "
                              "every run) to reproduce it later.")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.SystemRandom().randrange(2**32)
    rng = random.Random(args.seed)
    print(f"Using seed: {args.seed}")

    title = args.title or os.path.splitext(os.path.basename(args.audio))[0]
    audio_filename = args.audio_filename or os.path.basename(args.audio)

    print(f"Loading audio: {args.audio}")
    y, sr = librosa.load(args.audio, sr=None, mono=True)
    duration_seconds = len(y) / sr

    if args.bpm is None or args.offset is None:
        print("Detecting BPM and offset...")
        detected_bpm, detected_offset_seconds = detect_bpm_and_offset(y, sr)
    bpm = args.bpm if args.bpm is not None else detected_bpm
    if args.offset is not None:
        offset_seconds = (args.offset / 1000.0) % (60.0 / bpm)
    else:
        offset_seconds = detected_offset_seconds
    print(f"  BPM: {bpm:.2f}" + (" (manual)" if args.bpm is not None else ""))
    print(f"  Offset: {offset_seconds * 1000:.1f} ms" + (" (manual)" if args.offset is not None else ""))

    beat_length_ms = 60000.0 / bpm
    quarter_beat_ms = beat_length_ms / 4.0

    print("Analyzing song energy...")
    times_ms, energy = compute_energy_curve(args.audio)
    energy_at = make_energy_lookup(times_ms, energy)

    # Quantiles are computed from the same one-sample-per-beat grid that
    # actually gets classified (build_intensity_grid), not the raw high-
    # resolution energy curve -- otherwise a short, extreme transient (a
    # single hard hit) could skew a quantile in a way that doesn't reflect
    # how the *beat-grid* energy is actually distributed, since most of the
    # raw curve's samples fall between beats and are never even evaluated.
    beat_seconds = 60.0 / bpm
    n_beats = int((duration_seconds - offset_seconds) / beat_seconds) + 1
    slot_energy = np.array([energy_at((offset_seconds + i * beat_seconds) * 1000.0)
                             for i in range(n_beats)])
    smoothing_window = max(1, round(args.smoothing_beats))
    smoothed = smooth_slot_energy(slot_energy, smoothing_window)
    quantiles = intensity_quantiles(args.intensity)
    q_silent = float(np.quantile(smoothed, quantiles["silent"]))
    q_quiet = float(np.quantile(smoothed, quantiles["quiet"]))
    q_intense = float(np.quantile(smoothed, quantiles["intense"]))
    q_climax = float(np.quantile(smoothed, quantiles["climax"]))
    print(f"  intensity={args.intensity:.2f} -> quantiles silent<{quantiles['silent']:.3f}  "
          f"quiet<{quantiles['quiet']:.3f}  intense>{quantiles['intense']:.3f}  "
          f"climax>{quantiles['climax']:.3f}")
    print(f"  energy thresholds -> silent<{q_silent:.3f}  quiet<{q_quiet:.3f}  "
          f"intense>{q_intense:.3f}  climax>{q_climax:.3f}")

    times, climax_times = build_intensity_grid(offset_seconds, bpm, duration_seconds, energy_at,
                                                 q_silent, q_quiet, q_intense, q_climax,
                                                 smoothing_beats=args.smoothing_beats, intensity=args.intensity)
    before_cap = len(times)
    times = cap_fast_run_span(times, beat_length_ms, quarter_beat_ms, protected_ms=climax_times)
    if before_cap != len(times):
        print(f"Capped fast runs: dropped {before_cap - len(times)} circle(s) to keep every "
              f"quarter/eighth-beat run under a full beat")

    before_embellish = len(times)
    times = add_embellishment_chains(times, beat_length_ms, args.intensity, rng)
    if len(times) != before_embellish:
        print(f"Added {len(times) - before_embellish} embellishment note(s) across short 16th-note chains")
    print(f"Placing {len(times)} circles (intensity-adaptive: whole/half/quarter/eighth beat)...")

    positions = placeholder_positions(len(times))

    bm = default_metadata(title=title, artist=args.artist, creator=args.creator,
                           version=args.version, audio_filename=audio_filename)
    bm.timing_points = [TimingPoint(time=offset_seconds * 1000.0, beat_length=beat_length_ms)]

    preview_ms = compute_preview_time_ms(times_ms, energy)
    bm.general["PreviewTime"] = str(int(round(preview_ms)))

    hit_objects = []
    for i, (t, (x, y_pos)) in enumerate(zip(times, positions)):
        hit_objects.append(HitObject(x=x, y=y_pos, time=t, is_new_combo=(i % 8 == 0)))
    bm.hit_objects = hit_objects

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_osu(bm, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
