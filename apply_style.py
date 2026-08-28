#!/usr/bin/env python3
"""
Stage 3 — Apply style.

Repositions the hit objects produced by add_variety.py without touching
their timing, type, or count. This is purely about how the map *feels* to
play, following common osu! "rules of thumb":

  * Distance snap — for two objects a half beat or more apart, the on-screen
    distance between them is *exactly* what a slider spanning that same
    time gap would be (SliderMultiplier * 100 px/beat * beats of gap) — the
    same formula the game itself uses for a slider's length, so a circle
    jump reads with the same "speed" as a slider covering the same time.
    The same time gap always produces the same distance, everywhere in the
    song; energy shapes the turn-angle *pattern* (see below), not the size
    of the jump.
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

    A stack is only ever "stack" for its first half beat of elapsed time —
    beyond that it continues as "line" — so a literal stack (everything
    piled on one spot) never lasts longer than that, while a run can still
    go the rest of the way to the 8-circle cap as an overlapping line.

    Which of the two a given run picks is keyed to its measure's energy
    bucket (the same signal apply_style's motifs use), not a fresh coin
    flip every time: a bucket-seeded RNG makes the choice, so every run
    that lands in "this kind of section" (a verse's second and third
    repeat, say) reuses the *same* stack-vs-line choice every time it
    recurs, instead of re-rolling and looking different on each repeat.
    `stack_probability` still controls the overall mix (both are still
    good, per the design brief — this only makes the mix repeat
    consistently within a given section rather than changing which mix it
    is).
    """
    quarter_beat_ms = beat_length_ms / 4.0
    half_beat_ms = beat_length_ms / 2.0
    threshold = quarter_beat_ms + 1.0

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
            run_start_time = objects[i].time
            if measure_buckets and measure_length_ms:
                measure_index = int((run_start_time - offset_ms) // measure_length_ms)
                bucket = measure_buckets.get(measure_index, 0)
                bucket_rng = random.Random(f"stream_mode:{bucket}:{seed}")
                base_mode = "stack" if bucket_rng.random() < stack_probability else "line"
            else:
                base_mode = "stack" if rng.random() < stack_probability else "line"
            for k in range(i, j):
                elapsed = objects[k].time - run_start_time
                mode = base_mode if (base_mode == "line" or elapsed < half_beat_ms) else "line"
                mode_of[k] = (run_id, mode)
            run_id += 1
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
    parser.add_argument("--angle-jitter", type=float, default=4.0,
                         help="Degrees of random jitter added on top of each motif's turn angle "
                              "(circles and slider curves alike). Widening this only changes "
                              "angles/flow, never timing, note count, or object type — a way to "
                              "get more (or less) variety in the flow without being restrictive.")
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
    args = parser.parse_args()
    args.curviness = max(0.0, min(1.0, args.curviness))

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

    # Single-anchor slider shape mix, derived from --curviness (default 0.5
    # reproduces the original fixed 35% straight / 30% gentle-Bezier / 35%
    # pronounced-arc split). Higher curviness both shifts the mix toward
    # curved shapes and makes every curved bow more pronounced.
    straight_prob = max(0.0, 0.7 * (1.0 - args.curviness))
    bezier_prob = straight_prob + (1.0 - straight_prob) * 0.46
    bow_scale = 0.5 + args.curviness

    # Start roughly centered.
    cur_x, cur_y = PLAYFIELD_W / 2.0, PLAYFIELD_H / 2.0
    cur_angle = 0.0
    prev_end_time = None
    line_run_angle = None  # the single locked-in direction for the current "line" stream, if any
    stack_anchor = None  # the (x, y) every member of the current "stack" run holds at, if any
    current_run_id = None  # detects entering a *different* run, even one with the same mode

    for idx, obj in enumerate(objects):
        if prev_end_time is None:
            gap_ms = beat_length_ms
        else:
            gap_ms = max(1.0, obj.time - prev_end_time)

        tier = classify_tier(energy_at(obj.time), q_low, q_high)
        entry = stream_mode.get(idx)
        run_id, mode = entry if entry is not None else (None, None)
        if run_id != current_run_id:
            # A new run has started — even a same-mode run immediately
            # following another (the two are more than a quarter beat
            # apart, or build_stream_runs would have merged them into one)
            # must not silently inherit the previous run's stack position
            # or line direction.
            line_run_angle = None
            stack_anchor = None
            current_run_id = run_id

        if mode == "stack":
            if stack_anchor is None:
                # First member of this stack run: it still moves normally
                # to establish where the stack sits — freezing it at
                # whatever position preceded the run (rather than a
                # deliberately chosen new spot) would make the stack's
                # location arbitrary, and could even coincide with an
                # unrelated object several beats away that just happened
                # to precede it.
                spacing = max(MIN_SPACING, min(MAX_SPACING, snap_distance(gap_ms, beat_length_ms, slider_multiplier)))
                cur_angle = next_angle(cur_angle, tier, obj.time, offset_ms, beat_length_ms, measure_length_ms, measure_buckets, rng, jitter_degrees=args.angle_jitter)
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
            spacing = max(MIN_SPACING, min(MAX_SPACING, snap_distance(gap_ms, beat_length_ms, slider_multiplier)))
            new_x, new_y, line_run_angle = place_at_distance(cur_x, cur_y, spacing, line_run_angle)
            cur_x, cur_y = clamp_to_playfield(new_x, new_y, margin=MARGIN)
            cur_angle = line_run_angle
        else:
            # Outside a stream: normal distance-snap + motif-driven flow.
            spacing = max(MIN_SPACING, min(MAX_SPACING, snap_distance(gap_ms, beat_length_ms, slider_multiplier)))
            cur_angle = next_angle(cur_angle, tier, obj.time, offset_ms, beat_length_ms, measure_length_ms, measure_buckets, rng, jitter_degrees=args.angle_jitter)
            new_x, new_y, cur_angle = place_at_distance(cur_x, cur_y, spacing, cur_angle)
            cur_x, cur_y = clamp_to_playfield(new_x, new_y, margin=MARGIN)

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

                shape_roll = rng.random()
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
                        bow = min(40.0 * bow_scale, segment_length * 0.25 * bow_scale)
                    else:
                        # A real circular arc: safe here specifically
                        # because the bow is deliberately large relative to
                        # the chord (well clear of the near-collinear
                        # configuration that causes a perfect-circle curve
                        # to balloon outward) — a pronounced, legible curve
                        # that actually guides the cursor around a bend.
                        obj.curve_type = "P"
                        bow = min(70.0 * bow_scale, segment_length * 0.45 * bow_scale)
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
                obj.curve_type = "L" if rng.random() >= args.curviness else "B"
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
