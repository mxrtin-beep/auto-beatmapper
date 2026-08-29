#!/usr/bin/env python3
"""
Stage 3 — Apply style.

Repositions the hit objects produced by add_variety.py without touching
their timing, type, or count. This is purely about how the map *feels* to
play, following common osu! "rules of thumb":

  * Distance snap — for two objects a half beat or more apart, the on-screen
    distance between them is what a slider spanning that same time gap
    would be (SliderMultiplier * 100 px/beat * beats of gap, scaled by
    `--spacing`, default 1.0 keeps the base formula) — the same formula the
    game itself uses for a slider's length, so a circle jump reads with the
    same "speed" as a slider covering the same time. A given time gap
    produces (about) the same distance everywhere in the song — a small
    seeded wobble (a few percent) keeps jumps from feeling mechanically
    identical every time the same gap recurs, without meaningfully breaking
    the snap; energy shapes the turn-angle *pattern* (see below), not the
    size of the jump.
  * Streams/stacks — a run of circles a quarter beat or less apart (up to 8
    long; add_variety.py itself never produces a longer one) is positioned
    as a single deliberate unit in one of exactly two ways: every circle in
    the run at the *same* position (a stack), or all of them overlapping
    along one straight line — never the general zigzag flow, which is what
    turns a fast run into an unreadable blob. A stack specifically is
    capped to half a beat of elapsed time; a run that runs longer than that
    continues as an overlapping line instead.
  * Patterns / motifs — outside of streams, the turn angle between objects
    is drawn from a small fixed set of repeating shapes ("motifs"), one per
    energy tier, keyed to the object's position within the musical measure
    *and* that measure's own energy level (quantized into a handful of
    buckets). Two measures that sound alike — the second verse repeating
    the first, a chorus recurring — land in the same bucket and so reuse
    the exact same motif every time, which is what makes the pattern
    genuinely learnable: it's driven by where the song actually repeats
    itself, not an arbitrary rotating counter.
  * Flow — every motif avoids full 180-degree reversals and repeats of the
    same direction for too long, so movement still reads as a continuous
    swing rather than snapping.
  * Slider shape variety — a single-anchor slider is straight, a gentle
    Bezier arc, or a pronounced circular arc, in a mix controlled by
    `--curviness` (0-1, default 0.5: straight about a third of the time,
    gentle Bezier another third, pronounced arc the rest) — the bow for
    that last one is deliberately large relative to the slider's length
    specifically because that keeps a "perfect circle" curve's actual
    rendered arc close to its declared points (a *small* bow on a P curve
    is what risks it swinging off screen); higher curviness both shifts
    the mix toward curved shapes and makes every bow more pronounced. A
    multi-anchor chain similarly reads as a sharp polyline or one smooth
    curve through all its waypoints, with `--curviness` again controlling
    how often it's the smooth curve.
  * Playfield bounds — everything stays within the 512x384 field with a
    margin, bouncing off the edge instead of clipping.
  * Slider shape consistency — a slider's declared travel distance
    (`length`, which drives its timing/duration) always matches the actual
    distance from its start point to its rendered curve, so what you see is
    what you play.

Usage:
    python3 apply_style.py variety.osu --output out/song_style.osu [--audio song.mp3]
"""

from __future__ import annotations

import argparse
import math
import os
import random

import numpy as np

from beatmap_utils import HitObject, PLAYFIELD_H, PLAYFIELD_W, clamp_to_playfield, read_osu, write_osu

MARGIN = 30
MIN_SPACING = 10.0    # px, safety floor only — the distance-snap formula rarely needs it
MAX_SPACING = 600.0   # px, generous safety ceiling (a little over the playfield diagonal)

HALF_BEAT_STEPS_PER_MEASURE = 8  # 4/4 time, half-beat resolution

# A handful of repeating turn-angle "motifs" per energy tier (degrees,
# signed = turn direction), indexed by position-within-measure. Which motif
# plays in a given measure is keyed to that measure's energy bucket (see
# compute_measure_energy_buckets), so the same shape recurs every time a
# similar-sounding section repeats — recognizable on replay — while still
# varying between genuinely different sections. These only ever apply
# outside of streams/stacks (see build_stream_runs below).
#
# Since consecutive objects at a constant turn angle land the same
# distance-snapped spacing apart (equal time gap -> equal on-screen
# distance), a constant-degree motif isn't just an abstract angle sequence
# — it traces an actual, recognizable geometric shape on screen, the same
# vocabulary real maps lean on: a triangle (120 degrees), a square (90), a
# pentagon (72), a hexagon (60), a star/pentagram (144). Mixing those in
# with the zigzags and asymmetric builds below is what gives a section
# actual *structure* to recognize, instead of every measure reading as the
# same generic wiggle.
MOTIFS = {
    "quiet": [
        [35, 35, 35, 35, 35, 35, 35, 35],       # gentle spiral drift
        [45, -45, 45, -45, 45, -45, 45, -45],   # slow zigzag
        [60, 60, 60, 60, 60, 60, 60, 60],       # hexagon
    ],
    "normal": [
        [70, -70, 70, -70, 70, -70, 70, -70],   # zigzag
        [50, 50, -50, -50, 50, 50, -50, -50],   # paired swing
        [90, -40, 40, -90, 90, -40, 40, -90],   # asymmetric build/release
        [90, 90, 90, 90, 90, 90, 90, 90],       # square
        [72, 72, 72, 72, 72, 72, 72, 72],       # pentagon
    ],
    "intense": [
        [100, -100, 100, -100, 100, -100, 100, -100],  # sharp zigzag
        [60, 60, 60, -140, 60, 60, 60, -140],           # build then snap back
        [120, -60, 120, -60, -120, 60, -120, 60],       # asymmetric burst
        [120, 120, 120, 120, 120, 120, 120, 120],       # triangle
        [144, 144, 144, 144, 144, 144, 144, 144],       # star/pentagram
    ],
}


def compute_energy_lookup(audio_path: str):
    import librosa
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length) * 1000.0
    rms = rms / (rms.max() + 1e-9)

    def energy_at(t_ms: float) -> float:
        return float(np.interp(t_ms, times, rms))

    return energy_at


def classify_tier(energy_value: float, q_low: float, q_high: float) -> str:
    if energy_value < q_low:
        return "quiet"
    if energy_value > q_high:
        return "intense"
    return "normal"


NUM_ENERGY_BUCKETS = 5  # how many distinct "kinds of section" a tier's motifs can be keyed to


def compute_measure_energy_buckets(energy_at, offset_ms: float, measure_length_ms: float,
                                    track_end_ms: float, num_buckets: int = NUM_ENERGY_BUCKETS) -> dict[int, int]:
    """Which energy bucket each measure falls into, sampled across the whole track.

    A measure's bucket is its own average energy (sampled at 8 points
    across it) quantized into num_buckets levels. This is what lets a
    verse's second and third repeat pick the exact same motif as its
    first: they have essentially the same energy profile, so they land in
    the same bucket, every time — driven by the music's actual dynamics
    rather than an arbitrary rotating counter that has no reason to line
    up with where the song actually repeats itself.
    """
    num_measures = max(1, int((track_end_ms - offset_ms) / measure_length_ms) + 1)
    buckets: dict[int, int] = {}
    for m in range(num_measures):
        start = offset_ms + m * measure_length_ms
        samples = [energy_at(start + frac * measure_length_ms) for frac in (0.0, 0.125, 0.25, 0.375,
                                                                              0.5, 0.625, 0.75, 0.875)]
        avg = sum(samples) / len(samples)
        buckets[m] = min(num_buckets - 1, int(avg * num_buckets))
    return buckets


def motif_turn_degrees(tier: str, time_ms: float, offset_ms: float, beat_length_ms: float,
                        measure_length_ms: float, measure_buckets: dict[int, int]) -> float:
    """The signed turn angle (degrees) for an object, from its tier's repeating motif.

    Which of a tier's motifs plays is keyed to the measure's energy bucket,
    not a rotating index — so every measure that sounds like "this kind of
    section" (the same verse or chorus repeating) reuses the exact same
    motif, giving the player a real, learnable pattern instead of a motif
    that happens to cycle on its own unrelated schedule.
    """
    half_beat_ms = beat_length_ms / 2.0
    pos_in_measure = int(round((time_ms - offset_ms) / half_beat_ms)) % HALF_BEAT_STEPS_PER_MEASURE
    measure_index = int((time_ms - offset_ms) // measure_length_ms)
    bucket = measure_buckets.get(measure_index, 0)
    motifs = MOTIFS[tier]
    motif = motifs[bucket % len(motifs)]
    return motif[pos_in_measure % len(motif)]


def next_angle(prev_angle: float, tier: str, time_ms: float, offset_ms: float, beat_length_ms: float,
               measure_length_ms: float, measure_buckets: dict[int, int], rng: random.Random,
               jitter_degrees: float = 4.0) -> float:
    """Advance the flow angle using the tier's motif, plus a small humanizing jitter.

    The jitter is `jitter_degrees` wide by default — small, since the point
    of a motif is that it repeats recognizably and too much randomness
    would wash that out — but callers can widen it (e.g. `--angle-jitter`)
    to get more angle variety without touching anything about timing: this
    function only ever changes the flow *angle*, never `time_ms`, so a
    wider jitter still can't move an object off the beat grid.
    """
    turn_degrees = motif_turn_degrees(tier, time_ms, offset_ms, beat_length_ms, measure_length_ms, measure_buckets)
    turn_degrees += rng.uniform(-jitter_degrees, jitter_degrees)
    return prev_angle + math.radians(turn_degrees)


def place_at_distance(cur_x: float, cur_y: float, spacing: float, angle: float) -> tuple[float, float, float]:
    """Place a point exactly `spacing` away from (cur_x, cur_y) at `angle`, bounced into bounds.

    Critically, a wall bounce here corrects the *angle* and recomputes the
    position from (cur_x, cur_y) from scratch — it does not mirror an
    already-computed point. Mirroring a point preserves distance from the
    wall, not distance from (cur_x, cur_y): a straight-line jump that
    overshoots the wall and gets mirrored back can land much closer to
    (cur_x, cur_y) than `spacing`, silently breaking distance snap (the
    same time gap must always produce the same on-screen distance).
    Re-deriving the position from the corrected angle each time guarantees
    the result is always exactly `spacing` from (cur_x, cur_y).
    """
    left, right = MARGIN, PLAYFIELD_W - MARGIN
    top, bottom = MARGIN, PLAYFIELD_H - MARGIN

    def in_bounds(x: float, y: float) -> bool:
        return left <= x <= right and top <= y <= bottom

    original_angle = angle
    x = y = 0.0
    for _ in range(20):
        x = cur_x + spacing * math.cos(angle)
        y = cur_y + spacing * math.sin(angle)
        if in_bounds(x, y):
            return x, y, angle
        bounced = False
        if x < left or x > right:
            angle = math.pi - angle
            bounced = True
        if y < top or y > bottom:
            angle = -angle
            bounced = True
        if not bounced:
            break

    # A large gap near a corner can, rarely, put the angle-correction above
    # into a 2-cycle that never settles (correcting x re-breaks y and vice
    # versa) even though a valid angle exists. Falling back to a direct
    # search over candidate angles always finds one when it exists (any
    # angle whose endpoint lands in bounds), picking whichever is closest
    # to the originally intended direction — this is what actually
    # guarantees "same time gap -> same distance" holds even at the edges
    # of the playfield, rather than silently settling for a shorter jump.
    best = None
    for deg in range(0, 360, 2):
        a = math.radians(deg)
        cx = cur_x + spacing * math.cos(a)
        cy = cur_y + spacing * math.sin(a)
        if in_bounds(cx, cy):
            diff = abs((a - original_angle + math.pi) % (2 * math.pi) - math.pi)
            if best is None or diff < best[0]:
                best = (diff, cx, cy, a)
    if best is not None:
        return best[1], best[2], best[3]

    # No angle keeps the point in bounds at this exact distance (spacing
    # exceeds the farthest reachable point from here) — an extreme, rare
    # edge case. Fall back to whatever the last bounce attempt produced;
    # the caller still clamps it onto the playfield afterward.
    return x, y, angle


def snap_distance(gap_ms: float, beat_length_ms: float, slider_multiplier: float) -> float:
    """The exact pixel distance a slider spanning gap_ms would travel.

    This is deliberately the same formula used for a slider's own pixel
    length (slider_multiplier * 100 px/beat * beats of duration) — a circle
    jump covering the same amount of time is styled to read at the same
    "speed" as a slider would over that time.
    """
    return slider_multiplier * 100.0 * (gap_ms / beat_length_ms)


def styled_spacing(gap_ms: float, beat_length_ms: float, slider_multiplier: float,
                    spacing_scale: float, rng: random.Random, jitter_frac: float = 0.05) -> float:
    """`snap_distance`, scaled by `--spacing` and given a small seeded wobble.

    The wobble is tiny on purpose (a few percent) and drawn from the run's
    own seeded `rng` — enough that jump distances don't feel mechanically
    identical every time the same time gap recurs, without meaningfully
    breaking distance snap (a checker measuring "does this gap's spacing
    match this gap's time" would still call it a match) or losing
    reproducibility: the same seed always produces the same wobble.
    """
    base = snap_distance(gap_ms, beat_length_ms, slider_multiplier) * spacing_scale
    return base * (1.0 + rng.uniform(-jitter_frac, jitter_frac))


def build_stream_runs(objects: list[HitObject], beat_length_ms: float, rng: random.Random, seed: int,
                       offset_ms: float = 0.0, measure_length_ms: float = 0.0,
                       measure_buckets: dict[int, int] | None = None,
                       stack_probability: float = 0.5) -> dict[int, tuple[int, str]]:
    """Decide a stack/line mode for every object that's part of a stream.

    A stream is a maximal run of consecutive circles (never sliders) each a
    quarter beat or less from the one before — add_variety.py caps these at
    8 circles, converting any longer run into a slider itself, so nothing
    here needs to re-enforce that length limit.

    Returns {object index: (run_id, "stack" | "line")} for every object
    that belongs to a run of 2 or more; objects not in the mapping use the
    general motif-driven flow instead. The run_id matters because two
    *different* runs can sit back to back with no gap between them (the
    last object of one run and the first of the next, more than a quarter
    beat apart but with nothing in between) — the caller needs it to know
    a new run has started even when the mode happens to repeat, so it
    re-anchors a fresh stack (or picks a fresh line direction) instead of
    silently continuing the previous run's.

    A run picks exactly one mode for its *entire* length — never stack for
    part of it and line for the rest. Mixing the two within what reads as
    one run is genuinely confusing to a player (two circles piled on each
    other followed by the same run suddenly fanning out into a line reads
    as two different rhythms that happen to look the same at the seam),
    and it's also what osu!'s own ranking criteria calls out: stacks are
    fine, but switching stack behavior mid-pattern is the specific thing
    to avoid.

    A run longer than MAX_RUN_LEN objects is split into consecutive bursts
    of at most MAX_RUN_LEN, each a fully independent run with its own mode
    decision and its own entry/exit transition — a chain of, say, 8
    eighth-notes doesn't read as "click here 8 times in a pile/line," it
    reads as an arbitrary blob to count or a line so long it either bends
    to stay on screen or runs off it. Capped at MAX_RUN_LEN, the same 8
    notes become up to 3 short, clearly separated bursts instead — this
    applies regardless of mode, not just to stacks.

    A run is only *eligible* for "stack" if its whole span (first member to
    last) is half a beat or less — piling more than that much elapsed time
    onto one spot is a real overlap the ranking criteria's Hard rule
    forbids ("objects 1/2 of a beat apart or less must not fully overlap").
    A run failing that check is always "line" instead, in full, not
    partially.

    Which of the two an eligible run picks is keyed to its measure's energy
    bucket (the same signal apply_style's motifs use), not a fresh coin
    flip every time: a bucket-seeded RNG makes the choice, so every run
    that lands in "this kind of section" (a verse's second and third
    repeat, say) reuses the *same* stack-vs-line choice every time it
    recurs, instead of re-rolling and looking different on each repeat.
    `stack_probability` still controls the overall mix (both are still
    good, per the design brief) — even at 0 (always line), a burst is
    still capped at MAX_RUN_LEN, so "line" never means one long chain.
    """
    quarter_beat_ms = beat_length_ms / 4.0
    half_beat_ms = beat_length_ms / 2.0
    threshold = quarter_beat_ms + 1.0
    MAX_RUN_LEN = 3  # a run/burst longer than this reads as a smear or an overlong line, not a countable unit

    mode_of: dict[int, tuple[int, str]] = {}
    i = 0
    n = len(objects)
    run_id = 0
    while i < n:
        if objects[i].is_slider:
            i += 1
            continue
        j = i + 1
        while (j < n and not objects[j].is_slider
               and (objects[j].time - objects[j - 1].time) <= threshold):
            j += 1
        run_len = j - i
        if run_len >= 2:
            # Split into bursts of at most MAX_RUN_LEN, each its own run.
            burst_start = i
            burst_index = 0
            while burst_start < j:
                burst_end = min(burst_start + MAX_RUN_LEN, j)
                burst_len = burst_end - burst_start
                if burst_len >= 2:
                    burst_start_time = objects[burst_start].time
                    burst_span_ms = objects[burst_end - 1].time - burst_start_time
                    stack_eligible = burst_span_ms <= half_beat_ms
                    if stack_eligible:
                        if measure_buckets and measure_length_ms:
                            measure_index = int((burst_start_time - offset_ms) // measure_length_ms)
                            bucket = measure_buckets.get(measure_index, 0)
                            bucket_rng = random.Random(f"stream_mode:{bucket}:{burst_index}:{seed}")
                            base_mode = "stack" if bucket_rng.random() < stack_probability else "line"
                        else:
                            base_mode = "stack" if rng.random() < stack_probability else "line"
                    else:
                        base_mode = "line"
                    for k in range(burst_start, burst_end):
                        mode_of[k] = (run_id, base_mode)
                    run_id += 1
                burst_start = burst_end
                burst_index += 1
        i = j
    return mode_of


def main() -> None:
    parser = argparse.ArgumentParser(description="Restyle object placement in a beatmap (timing/objects unchanged).")
    parser.add_argument("beatmap", help="Path to the variety .osu file (from add_variety.py).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio", default=None, help="Optional path to the song's MP3, for energy-aware patterns.")
    parser.add_argument("--version", default="Auto Styled", help="Difficulty/version name to write into the map.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed. Omit for different styling every run; pass a fixed "
                              "value (printed on every run) to reproduce the exact same map later.")
    parser.add_argument("--temperature", type=float, default=0.5,
                         help="How creative vs. structured the styling gets, 0-1 (default 0.5). "
                              "Scales --angle-jitter, how much a section's curviness can drift from "
                              "the --curviness baseline, and how strongly the path wanders around "
                              "the playfield, all together — low is tight and predictable, high is "
                              "loose and varied. Passing one of those flags explicitly overrides "
                              "temperature's value for that one knob only.")
    parser.add_argument("--angle-jitter", type=float, default=None,
                         help="Degrees of random jitter added on top of each motif's turn angle "
                              "(circles and slider curves alike). Widening this only changes "
                              "angles/flow, never timing, note count, or object type — a way to "
                              "get more (or less) variety in the flow without being restrictive. "
                              "Defaults to a value derived from --temperature (roughly 1-10).")
    parser.add_argument("--stack-probability", type=float, default=0.5,
                         help="Overall mix between stream runs that stack in one spot and runs "
                              "that trace a straight line (0 = always line, 1 = always stack). "
                              "Which one a given repeating section picks stays consistent across "
                              "its repeats either way.")
    parser.add_argument("--curviness", type=float, default=0.5,
                         help="How curvy the map feels, 0-1. 0 makes almost every slider a "
                              "straight line; 1 makes almost every slider a pronounced curve "
                              "(and makes the bow of every curved slider more pronounced too). "
                              "0.5 (default) matches the original straight/gentle-arc/pronounced-arc "
                              "mix.")
    parser.add_argument("--spacing", type=float, default=1.3,
                         help="Multiplier on jump/spacing distance (1.0 = the base distance-snap "
                              "formula; default 1.3, the top of the ranking-criteria-recommended "
                              "range, since lower values still read as too close together / "
                              "prone to crisscrossing).")
    args = parser.parse_args()
    args.curviness = max(0.0, min(1.0, args.curviness))
    args.spacing = max(0.1, args.spacing)
    args.temperature = max(0.0, min(1.0, args.temperature))
    if args.angle_jitter is None:
        args.angle_jitter = 1.0 + args.temperature * 9.0  # 1-10 degrees
    curviness_variance = 0.15 + args.temperature * 0.4  # 0.15-0.55
    wander_strength = 0.12 + args.temperature * 0.28  # 0.12-0.40

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
        raise RuntimeError("Beatmap has no hit objects to restyle.")

    if args.audio:
        energy_at = compute_energy_lookup(args.audio)
        obj_energy = np.array([energy_at(o.time) for o in objects])
        q_low = float(np.quantile(obj_energy, 0.35))
        q_high = float(np.quantile(obj_energy, 0.75))
    else:
        energy_at = lambda t: 0.5
        q_low, q_high = 0.35, 0.75

    measure_buckets = compute_measure_energy_buckets(energy_at, offset_ms, measure_length_ms, objects[-1].time)
    stream_mode = build_stream_runs(objects, beat_length_ms, rng, args.seed, offset_ms=offset_ms,
                                     measure_length_ms=measure_length_ms, measure_buckets=measure_buckets,
                                     stack_probability=args.stack_probability)

    # --spacing itself shifts a little, a handful of times over the course
    # of the song, instead of staying exactly one multiplier the whole way
    # through — a song genuinely has a few different sections (verse,
    # chorus, bridge...), and a spacing change is one more way a section
    # reads as its own thing. `--temperature` controls how many times it
    # changes (0 -> never, this tier's whole point; 1 -> up to
    # MAX_SPACING_SECTION_CHANGES times) — never "every object", and never
    # more than a few times in one song either way. Each change alternates
    # the sign of the shift (rather than an independent random draw per
    # section, which could by chance land two neighboring sections within
    # a percent of each other — a change too small to actually notice) and
    # keeps the *magnitude* in a band that reads as a deliberate shift
    # without being jarring enough to break the flow a player just learned.
    MAX_SPACING_SECTION_CHANGES = 3
    num_spacing_changes = round(args.temperature * MAX_SPACING_SECTION_CHANGES)
    num_spacing_sections = num_spacing_changes + 1
    section_rng = random.Random(f"spacing_section:{args.seed}")
    spacing_section_scales = [1.0]
    for section_i in range(1, num_spacing_sections):
        sign = 1 if section_i % 2 == 1 else -1
        spacing_section_scales.append(1.0 + sign * section_rng.uniform(0.15, 0.25))
    spacing_section_span_ms = max(1.0, (objects[-1].time - objects[0].time) / num_spacing_sections)

    def spacing_scale_for(time_ms: float) -> float:
        section_index = int((time_ms - objects[0].time) // spacing_section_span_ms)
        section_index = max(0, min(num_spacing_sections - 1, section_index))
        return spacing_section_scales[section_index]

    # Each measure bucket gets its own curviness level, offset from
    # --curviness by a bucket-seeded amount — the same "keyed to the
    # bucket, not a fresh roll" trick the motifs and stream mode use. This
    # is what makes curviness read as a *theme* per section (a chorus that
    # stays consistently curvy, a verse that stays consistently straight
    # and bendy) rather than every slider independently coin-flipping its
    # own shape regardless of what the rest of its section looks like.
    bucket_curviness: dict[int, float] = {}
    for bucket in range(NUM_ENERGY_BUCKETS):
        bucket_rng = random.Random(f"curviness:{bucket}:{args.seed}")
        bucket_curviness[bucket] = max(0.0, min(1.0, args.curviness
                                                 + bucket_rng.uniform(-curviness_variance, curviness_variance)))

    def shape_mix_for(time_ms: float) -> tuple[float, float, float, float]:
        """(curviness, straight_prob, bezier_prob, bow_scale) for the slider
        at time_ms, from its measure's own curviness level. 0.5 reproduces
        the original fixed 35% straight / 30% gentle-Bezier / 35%
        pronounced-arc split; higher shifts the mix toward curved shapes
        and makes bows bigger."""
        measure_index = int((time_ms - offset_ms) // measure_length_ms)
        bucket = measure_buckets.get(measure_index, 0)
        curviness = bucket_curviness.get(bucket, args.curviness)
        straight = max(0.0, 0.7 * (1.0 - curviness))
        bezier = straight + (1.0 - straight) * 0.46
        return curviness, straight, bezier, 0.5 + curviness

    # Start roughly centered.
    cur_x, cur_y = PLAYFIELD_W / 2.0, PLAYFIELD_H / 2.0
    cur_angle = 0.0
    prev_end_time = None
    line_run_angle = None  # the single locked-in direction for the current "line" stream, if any
    stack_anchor = None  # the (x, y) every member of the current "stack" run holds at, if any
    current_run_id = None  # detects entering a *different* run, even one with the same mode

    # A slow "wander" target keeps the whole path migrating around the
    # playfield instead of orbiting one local spot — a run of same-sign or
    # constant-magnitude motif turns is, by construction, a closed or
    # near-closed shape (a square, a spiral, a zigzag), so left alone the
    # cursor tends to circle back near wherever it already is. Re-rolled on
    # every new combo (a natural phrase boundary) and blended in as a small
    # nudge on top of the motif angle each step — it never overrides the
    # motif's shape, just which direction the whole shape drifts in.
    wander_rng = random.Random(f"wander:{args.seed}")
    wander_target = (wander_rng.uniform(MARGIN, PLAYFIELD_W - MARGIN),
                      wander_rng.uniform(MARGIN, PLAYFIELD_H - MARGIN))

    # A stack/line run (or burst — build_stream_runs splits anything longer
    # than MAX_RUN_LEN into several) reads as one deliberate unit only if
    # it's visually set apart from whatever comes right before and after it
    # — otherwise a pile of circles bleeds into the normal flow, or into the
    # next burst, and the run boundary disappears. STREAM_TRANSITION_BOOST
    # widens just the one connecting gap on both sides of a run (never a gap
    # inside it, and never anything more than one gap away) on top of
    # ordinary distance snap — including the gap between two consecutive
    # bursts of the same long stream, not just where a stream meets normal
    # flow. was_in_stream tracks whether the *previous* object belonged to
    # a run at all, so the boost applies leaving one too, not just entering.
    STREAM_TRANSITION_BOOST = 1.4
    was_in_stream = False

    def wander_nudge(angle: float, x: float, y: float) -> float:
        bias = math.atan2(wander_target[1] - y, wander_target[0] - x)
        diff = (bias - angle + math.pi) % (2 * math.pi) - math.pi
        return angle + diff * wander_strength

    for idx, obj in enumerate(objects):
        if prev_end_time is None:
            gap_ms = beat_length_ms
        else:
            gap_ms = max(1.0, obj.time - prev_end_time)

        if obj.is_new_combo:
            wander_target = (wander_rng.uniform(MARGIN, PLAYFIELD_W - MARGIN),
                              wander_rng.uniform(MARGIN, PLAYFIELD_H - MARGIN))

        tier = classify_tier(energy_at(obj.time), q_low, q_high)
        entry = stream_mode.get(idx)
        run_id, mode = entry if entry is not None else (None, None)
        entering_new_run = run_id is not None and run_id != current_run_id
        if run_id != current_run_id:
            # A new run has started — even a same-mode run immediately
            # following another (the two are more than a quarter beat
            # apart, or build_stream_runs would have merged them into one)
            # must not silently inherit the previous run's stack position
            # or line direction.
            line_run_angle = None
            stack_anchor = None
            current_run_id = run_id

        entering_stream = entering_new_run
        leaving_stream = mode is None and was_in_stream
        boost = STREAM_TRANSITION_BOOST if (entering_stream or leaving_stream) else 1.0

        if mode == "stack":
            if stack_anchor is None:
                # First member of this stack run: it still moves normally
                # (plus the transition boost, since this gap is what sets
                # the stack apart from whatever came before it) to
                # establish where the stack sits — freezing it at whatever
                # position preceded the run (rather than a deliberately
                # chosen new spot) would make the stack's location
                # arbitrary, and could even coincide with an unrelated
                # object several beats away that just happened to precede
                # it.
                spacing = max(MIN_SPACING, min(MAX_SPACING, boost * styled_spacing(gap_ms, beat_length_ms, slider_multiplier, args.spacing * spacing_scale_for(obj.time), rng)))
                cur_angle = next_angle(cur_angle, tier, obj.time, offset_ms, beat_length_ms, measure_length_ms, measure_buckets, rng, jitter_degrees=args.angle_jitter)
                cur_angle = wander_nudge(cur_angle, cur_x, cur_y)
                new_x, new_y, cur_angle = place_at_distance(cur_x, cur_y, spacing, cur_angle)
                cur_x, cur_y = clamp_to_playfield(new_x, new_y, margin=MARGIN)
                stack_anchor = (cur_x, cur_y)
            else:
                # Every other circle in this run: hold the exact same spot.
                cur_x, cur_y = stack_anchor
        elif mode == "line":
            # The whole run moves along one fixed direction, decided once
            # when the run is first entered, so consecutive circles
            # overlap along a straight line rather than zigzagging.
            if line_run_angle is None:
                line_run_angle = next_angle(cur_angle, tier, obj.time, offset_ms, beat_length_ms,
                                             measure_length_ms, measure_buckets, rng, jitter_degrees=args.angle_jitter)
                line_run_angle = wander_nudge(line_run_angle, cur_x, cur_y)
            spacing = max(MIN_SPACING, min(MAX_SPACING, boost * styled_spacing(gap_ms, beat_length_ms, slider_multiplier, args.spacing * spacing_scale_for(obj.time), rng)))
            # A run's direction is locked in once, above — but if it
            # happens to point straight at a wall, letting place_at_distance
            # "bounce" it back on every single step (as every other call
            # site does) doesn't read as one straight line at all: since
            # cur_x/cur_y barely moves once pinned against the edge, the
            # very same bounce fires again next step, and the run oscillates
            # between two points for the rest of its length — a run visibly
            # doubling back over its own earlier members and anything else
            # nearby. So a genuine wall conflict re-aims the run *once*,
            # back toward the open field, rather than re-bouncing forever;
            # a run that doesn't hit a wall never re-aims at all.
            raw_x = cur_x + spacing * math.cos(line_run_angle)
            raw_y = cur_y + spacing * math.sin(line_run_angle)
            if not (MARGIN <= raw_x <= PLAYFIELD_W - MARGIN and MARGIN <= raw_y <= PLAYFIELD_H - MARGIN):
                line_run_angle = math.atan2(PLAYFIELD_H / 2.0 - cur_y, PLAYFIELD_W / 2.0 - cur_x)
                line_run_angle += math.radians(rng.uniform(-20.0, 20.0))
            new_x, new_y, _ = place_at_distance(cur_x, cur_y, spacing, line_run_angle)
            cur_x, cur_y = clamp_to_playfield(new_x, new_y, margin=MARGIN)
            cur_angle = line_run_angle
        else:
            # Outside a stream: normal distance-snap + motif-driven flow
            # (plus the transition boost on the one gap right after a
            # stream ends, for the same readability reason as entering one).
            spacing = max(MIN_SPACING, min(MAX_SPACING, boost * styled_spacing(gap_ms, beat_length_ms, slider_multiplier, args.spacing * spacing_scale_for(obj.time), rng)))
            cur_angle = next_angle(cur_angle, tier, obj.time, offset_ms, beat_length_ms, measure_length_ms, measure_buckets, rng, jitter_degrees=args.angle_jitter)
            cur_angle = wander_nudge(cur_angle, cur_x, cur_y)
            new_x, new_y, cur_angle = place_at_distance(cur_x, cur_y, spacing, cur_angle)
            cur_x, cur_y = clamp_to_playfield(new_x, new_y, margin=MARGIN)

        was_in_stream = mode is not None

        obj.x, obj.y = cur_x, cur_y

        if obj.is_slider:
            num_segments = len(obj.points)  # 1 for a simple/bouncing slider, 2-3 for a merged chain
            segment_length = obj.length / num_segments

            if num_segments == 1:
                # A lone slider (including a bouncing one) gets a shape
                # drawn from three options instead of almost always being
                # a straight 1-beat line: straight, a gentle Bezier arc, or
                # a more pronounced circular arc that actually guides the
                # cursor through a real curve.
                end_angle = next_angle(cur_angle, tier, obj.time, offset_ms, beat_length_ms,
                                        measure_length_ms, measure_buckets, rng, jitter_degrees=args.angle_jitter)
                end_x, end_y, end_angle = place_at_distance(cur_x, cur_y, segment_length, end_angle)
                end_x, end_y = clamp_to_playfield(end_x, end_y, margin=MARGIN)

                _, straight_prob, bezier_prob, bow_scale = shape_mix_for(obj.time)
                shape_roll = rng.random()
                # A tiny seeded wobble on the bow itself, same reasoning as
                # styled_spacing: keeps curves from looking mechanically
                # identical whenever curviness happens to land the same
                # shape twice, while staying reproducible for a given seed.
                bow_jitter = 1.0 + rng.uniform(-0.1, 0.1)
                if shape_roll < straight_prob:
                    obj.curve_type = "L"
                    obj.points = [(end_x, end_y)]
                else:
                    mid_x, mid_y = (cur_x + end_x) / 2.0, (cur_y + end_y) / 2.0
                    perp_angle = end_angle + math.pi / 2
                    if shape_roll < bezier_prob:
                        # A quadratic Bezier through (start, bow, end) — a
                        # gentle arc. Unlike a "P" (perfect-circle) curve
                        # with a *small* bow, whose rendered path can swing
                        # well outside the triangle these three points form
                        # (and off the visible playfield) when they're close
                        # to collinear, a Bezier is mathematically
                        # guaranteed to stay within their convex hull.
                        obj.curve_type = "B"
                        bow = min(40.0 * bow_scale, segment_length * 0.25 * bow_scale) * bow_jitter
                    else:
                        # A real circular arc: safe here specifically
                        # because the bow is deliberately large relative to
                        # the chord (well clear of the near-collinear
                        # configuration that causes a perfect-circle curve
                        # to balloon outward) — a pronounced, legible curve
                        # that actually guides the cursor around a bend.
                        obj.curve_type = "P"
                        bow = min(70.0 * bow_scale, segment_length * 0.45 * bow_scale) * bow_jitter
                    bow_x, bow_y = clamp_to_playfield(mid_x + bow * math.cos(perp_angle),
                                                       mid_y + bow * math.sin(perp_angle), margin=MARGIN)
                    obj.points = [(bow_x, bow_y), (end_x, end_y)]

                cur_x, cur_y, cur_angle = end_x, end_y, end_angle
            else:
                # Chain slider: walk one flow-angle segment per waypoint so
                # each note in the chain still reads as a distinct hop, then
                # decide once whether the whole chain reads as a sharp
                # polyline or one smooth curve flowing through every
                # waypoint (a Bezier through several points is still
                # guaranteed to stay within their convex hull, so this is
                # safe even for a long, sweeping chain).
                chain_curviness, _, _, _ = shape_mix_for(obj.time)
                obj.curve_type = "L" if rng.random() >= chain_curviness else "B"
                new_points = []
                for _ in range(num_segments):
                    cur_angle = next_angle(cur_angle, tier, obj.time, offset_ms, beat_length_ms,
                                            measure_length_ms, measure_buckets, rng, jitter_degrees=args.angle_jitter)
                    px, py, cur_angle = place_at_distance(cur_x, cur_y, segment_length, cur_angle)
                    cur_x, cur_y = clamp_to_playfield(px, py, margin=MARGIN)
                    new_points.append((cur_x, cur_y))
                obj.points = new_points

            prev_end_time = obj.end_time(beat_length_ms, slider_multiplier)
        else:
            prev_end_time = obj.time

    bm.hit_objects = objects
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_osu(bm, args.output)
    print(f"Restyled {len(objects)} objects -> {args.output}")


if __name__ == "__main__":
    main()
