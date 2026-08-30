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


def find_repeating_measure_map(measure_buckets: dict[int, int], window: int = 4) -> dict[int, int]:
    """For every measure that's part of a >=2-times-repeating `window`-
    measure shingle, map it to the *first* occurrence of that same shingle
    — every other measure maps to itself. Mirrors make_easy.py's own
    find_repetitive_measures (same windowed-shingle matching, so a single
    coincidentally-matching measure isn't enough — a real multi-measure
    section has to recur), but returns *which* earlier measure each repeat
    matches, not just whether it's a repeat, so a later occurrence of a
    verse/chorus can be pointed back at its first: checking the reference
    set (example/keha_backstabber/) found a section's second and third
    pass mostly reusing its *first* pass's exact hitsound sequence, and
    frequently its exact circle/slider layout too (e.g. measures 11 and 19
    both come out CCCSSSS; measures 25 and 29 both CCCCCCS) — not just
    landing in the same coarse energy bucket independently each time.

    Lives here (not add_variety.py, which imports it back) since
    apply_style.py's own motif_turn_degrees also uses it directly, to
    remap which measure's motif an occurrence plays -- see its own
    docstring.
    """
    n = max(measure_buckets) + 1 if measure_buckets else 0
    result = {m: m for m in range(n)}
    if n < window * 2:
        return result

    signature_starts: dict[tuple[int, ...], list[int]] = {}
    for start in range(n - window + 1):
        sig = tuple(measure_buckets.get(start + k, 0) for k in range(window))
        signature_starts.setdefault(sig, []).append(start)

    for starts in signature_starts.values():
        distinct_starts = []
        for s in starts:
            if not distinct_starts or s - distinct_starts[-1] >= window:
                distinct_starts.append(s)
        if len(distinct_starts) >= 2:
            first = distinct_starts[0]
            for s in distinct_starts[1:]:
                for k in range(window):
                    result[s + k] = first + k
    return result


def motif_turn_degrees(tier: str, time_ms: float, offset_ms: float, beat_length_ms: float,
                        measure_length_ms: float, measure_buckets: dict[int, int],
                        measure_repeat_map: dict[int, int] | None = None) -> float:
    """The signed turn angle (degrees) for an object, from its tier's repeating motif.

    Which of a tier's motifs plays is keyed to the measure's energy bucket,
    not a rotating index — so every measure that sounds like "this kind of
    section" (the same verse or chorus repeating) reuses the exact same
    motif, giving the player a real, learnable pattern instead of a motif
    that happens to cycle on its own unrelated schedule.

    `measure_repeat_map` (see find_repeating_measure_map), when given,
    tightens that up further: a measure whose own windowed sequence of
    buckets genuinely recurs elsewhere (not just this one measure's bucket
    value coincidentally matching) plays its motif from the *first*
    occurrence's own measure index, rather than its own -- two energy
    passes of the same section can land in adjacent buckets from small
    energy differences alone, which used to read as "close but not quite"
    the same arrangement; this makes a real repeat read as the exact same
    one, matching how hitsounds and (add_sliders_v2.py's own) circle/
    slider layout already reuse a verse/chorus's first pass.
    """
    half_beat_ms = beat_length_ms / 2.0
    pos_in_measure = int(round((time_ms - offset_ms) / half_beat_ms)) % HALF_BEAT_STEPS_PER_MEASURE
    measure_index = int((time_ms - offset_ms) // measure_length_ms)
    if measure_repeat_map is not None:
        measure_index = measure_repeat_map.get(measure_index, measure_index)
    bucket = measure_buckets.get(measure_index, 0)
    motifs = MOTIFS[tier]
    motif = motifs[bucket % len(motifs)]
    return motif[pos_in_measure % len(motif)]


def next_angle(prev_angle: float, tier: str, time_ms: float, offset_ms: float, beat_length_ms: float,
               measure_length_ms: float, measure_buckets: dict[int, int], rng: random.Random,
               jitter_degrees: float = 4.0, measure_repeat_map: dict[int, int] | None = None) -> float:
    """Advance the flow angle using the tier's motif, plus a small humanizing jitter.

    The jitter is `jitter_degrees` wide by default — small, since the point
    of a motif is that it repeats recognizably and too much randomness
    would wash that out — but callers can widen it (e.g. `--angle-jitter`)
    to get more angle variety without touching anything about timing: this
    function only ever changes the flow *angle*, never `time_ms`, so a
    wider jitter still can't move an object off the beat grid.
    """
    turn_degrees = motif_turn_degrees(tier, time_ms, offset_ms, beat_length_ms, measure_length_ms, measure_buckets,
                                       measure_repeat_map=measure_repeat_map)
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


def p_curve_arc_bbox(p0: tuple[float, float], p1: tuple[float, float],
                      p2: tuple[float, float]) -> tuple[float, float, float, float] | None:
    """Bounding box of the actual rendered arc of a "P" (perfect-circle)
    slider through (p0, p1, p2) — the arc's *own* extent, not just the
    bounding box of its three defining points. This is what makes a P
    curve unsafe in a way a Bezier through the same points never is: all
    three points can individually sit well inside the playfield while the
    circular arc connecting them still bulges outside it, whenever the
    arc's radius is large relative to the chord (near-collinear points
    especially). Returns None if the three points are exactly collinear
    (no finite circle fits) — the caller should treat that as unsafe too.
    """
    (ax, ay), (bx, by), (cx, cy) = p0, p1, p2
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)

    a0 = math.atan2(ay - uy, ax - ux)
    a1 = math.atan2(by - uy, bx - ux)
    a2 = math.atan2(cy - uy, cx - ux)

    def normalize_above(angle: float, ref: float) -> float:
        while angle < ref:
            angle += 2 * math.pi
        while angle > ref + 2 * math.pi:
            angle -= 2 * math.pi
        return angle

    # Sweep from a0 to a2 the way that actually passes through a1 (the
    # slider's own bow/anchor point, its declared shape) -- the *other*
    # way around the circle is not the arc osu! renders.
    a2n = normalize_above(a2, a0)
    a1n = normalize_above(a1, a0)
    if not (a0 <= a1n <= a2n):
        a2n = a0 - (2 * math.pi - (a2n - a0))
    lo, hi = min(a0, a2n), max(a0, a2n)

    xs, ys = [ax, bx, cx], [ay, by, cy]
    steps = 32
    for i in range(steps + 1):
        angle = lo + (hi - lo) * i / steps
        xs.append(ux + r * math.cos(angle))
        ys.append(uy + r * math.sin(angle))
    return min(xs), max(xs), min(ys), max(ys)


def p_curve_fits_playfield(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float],
                            margin: float) -> bool:
    """Whether a P-curve slider's actual rendered arc through these three
    points stays within the playfield margin -- see p_curve_arc_bbox."""
    bbox = p_curve_arc_bbox(p0, p1, p2)
    if bbox is None:
        return False
    xlo, xhi, ylo, yhi = bbox
    return xlo >= margin and xhi <= PLAYFIELD_W - margin and ylo >= margin and yhi <= PLAYFIELD_H - margin


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
                       stream_frequency: float = 0.5,
                       stack_probability: float = 0.5) -> dict[int, tuple[int, str]]:
    """Decide a mode — "stack", "line", or "flow" — for every object that's
    part of a fast (quarter-beat-or-closer) run.

    A stream is a maximal run of consecutive circles (never sliders) each a
    quarter beat or less from the one before — add_variety.py caps these at
    8 circles, converting any longer run into a slider itself, so nothing
    here needs to re-enforce that length limit.

    Returns {object index: (run_id, "stack" | "line" | "flow")} for every
    object that belongs to a run of 2 or more; objects not in the mapping
    (a run of 1, or not part of any close-together run at all) use the
    general motif-driven flow directly, with no run/transition handling.
    The run_id matters because two *different* runs — or bursts, see below
    — can sit back to back with no gap between them: the caller needs it
    to know a new one has started even when the mode happens to repeat, so
    it re-anchors a fresh stack (or picks a fresh line direction) instead
    of silently continuing the previous one's, and widens the connecting
    gap between them either way.

    A run longer than MAX_RUN_LEN objects is split into consecutive bursts
    of at most MAX_RUN_LEN (matching add_variety.py's own hard cap on how
    long a run of quarter/eighth-spaced circles is ever allowed to get),
    each with its own mode decision and its own entry/exit transition.

    A burst only counts as an actual "stream" — eligible to be forced into
    a stack or a line at all — once it's 4 or more notes long; that's the
    definition (a run of 2-3 fast notes is just a quick triplet, not a
    stream). Shorter bursts always use ordinary motif-driven flow, the
    same as an object outside any fast run, regardless of
    `stream_frequency` below.

    Two independent knobs govern this, deliberately kept separate since
    they answer two different questions:

    `stream_frequency` (0-1) is the chance that an eligible burst becomes
    a deliberate stream unit *at all* — stacked in one spot or spread
    along a locked-in straight line — rather than "flow": its members just
    follow the ordinary motif-driven placement any other note would, one
    at a time, with no forced overlap or fixed direction. At 0, no burst
    is ever forced into a stack or line at all; the notes are still there
    (bursts are still split at MAX_RUN_LEN, and the surrounding transition
    gap still applies), they just never get piled up or locked onto one
    line. At 1, every eligible burst becomes a stream.

    `stack_probability` (0-1) only matters for whichever bursts
    `stream_frequency` already decided *are* a stream: of those, how many
    pile into one stacked spot versus spread along a line (0 = always
    line, 1 = always stack, whenever the burst is short enough to stack at
    all — see below). It has no say over whether a burst streams in the
    first place; that's `stream_frequency`'s job alone.

    Every roll (frequency, then stack-vs-line) is keyed to the burst's
    measure energy bucket (the same signal apply_style's motifs use), not
    a fresh coin flip every time: a bucket-seeded RNG makes the choice, so
    every burst that lands in "this kind of section" (a verse's second and
    third repeat, say) reuses the exact same choices every time it
    recurs, instead of re-rolling and looking different on each repeat.

    Every member of a streaming burst is individually a quarter beat or
    less from its neighbor (that's the definition of the run in the first
    place — see `threshold` below), which is exactly the gap the ranking
    criteria's own "must not fully overlap" rule is scoped to (it reads
    pairwise, e.g. "objects 1/2 of a beat apart or less must not fully
    overlap" — about each consecutive pair, not the run's total elapsed
    span); a genuine stacked stream several notes long is a normal, legal
    mapping technique. So `stack_probability` alone decides "stack" vs.
    "line" for every burst that streams at all — no separate span-based
    eligibility gate. (An earlier version gated "stack" on the whole
    burst's span being under half a beat, left over from when a burst was
    always capped at 3 notes — trivially always true then, so it was
    silently a no-op — but once bursts could run up to MAX_RUN_LEN notes
    that gate started rejecting nearly every real burst, which is exactly
    why --stack-probability stopped visibly doing anything.)
    """
    quarter_beat_ms = beat_length_ms / 4.0
    eighth_beat_ms = beat_length_ms / 8.0
    threshold = quarter_beat_ms + 1.0
    MAX_RUN_LEN = 8  # matches add_variety.py's own hard cap (cap_stream_length's max_len at frequency 1)
    MIN_STREAM_LEN = 4  # fewer than this is a quick triplet, not a stream (see docstring)

    def gap_rate(gap_ms: float) -> str:
        # "eighth" (a climax burst's own rate) vs. "quarter" (everything
        # else this loop ever sees, since threshold above already only
        # lets a quarter-beat-or-closer gap through in the first place).
        return "eighth" if gap_ms <= eighth_beat_ms + 1.0 else "quarter"

    mode_of: dict[int, tuple[int, str]] = {}
    i = 0
    n = len(objects)
    run_id = 0
    while i < n:
        if objects[i].is_slider:
            i += 1
            continue
        j = i + 1
        run_rate = None
        while (j < n and not objects[j].is_slider
               and (objects[j].time - objects[j - 1].time) <= threshold):
            # A run only ever streams at one consistent pace -- a stack
            # mixing an eighth-beat climax burst with a slower quarter-beat
            # stretch reads as one held-in-place gesture even though the
            # actual pacing changed partway through it, which is
            # disorienting (the same held spot no longer means "hit these
            # all at the same rate"). Splitting into a fresh run right at
            # the rate change gives the change its own entry/exit gap and
            # (if it streams) its own stack position instead.
            rate = gap_rate(objects[j].time - objects[j - 1].time)
            if run_rate is None:
                run_rate = rate
            elif rate != run_rate:
                break
            j += 1
        run_len = j - i
        if run_len >= 2:
            # Split into bursts of at most MAX_RUN_LEN, each its own run.
            burst_start = i
            burst_index = 0
            while burst_start < j:
                burst_end = min(burst_start + MAX_RUN_LEN, j)
                burst_len = burst_end - burst_start
                if burst_len >= MIN_STREAM_LEN:
                    burst_start_time = objects[burst_start].time

                    if measure_buckets and measure_length_ms:
                        measure_index = int((burst_start_time - offset_ms) // measure_length_ms)
                        bucket = measure_buckets.get(measure_index, 0)
                        freq_rng = random.Random(f"stream_freq:{bucket}:{burst_index}:{seed}")
                        is_stream = freq_rng.random() < stream_frequency
                        mix_rng = random.Random(f"stream_mode:{bucket}:{burst_index}:{seed}")
                        wants_stack = mix_rng.random() < stack_probability
                    else:
                        is_stream = rng.random() < stream_frequency
                        wants_stack = rng.random() < stack_probability

                    if is_stream:
                        base_mode = "stack" if wants_stack else "line"
                    else:
                        base_mode = "flow"

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
    parser.add_argument("--stream-frequency", type=float, default=0.1,
                         help="How often a fast (quarter-beat-or-closer) burst of 4+ notes is placed "
                              "as a deliberate stream unit (stacked in one spot, or spread along one "
                              "locked-in line) versus just following ordinary flow like any other "
                              "note (0 = never a stream, 1 = always one). A run longer than 8 is "
                              "always split into separate bursts of at most 8 regardless of this "
                              "setting. Which one a given repeating section picks stays consistent "
                              "across its repeats either way. Default 0.1 — deliberately low, since "
                              "even a modest value here already makes streams a regular occurrence.")
    parser.add_argument("--stack-probability", type=float, default=1.0,
                         help="Of whichever bursts --stream-frequency already decided ARE a "
                              "stream: the mix between piling into one stacked spot and spreading "
                              "along a line (0 = always line, 1 = always stack). Has no effect on "
                              "whether a burst streams in the first place — that's "
                              "--stream-frequency's job. Default 1.0 (always stack).")
    parser.add_argument("--curviness", type=float, default=0.5,
                         help="How curvy the map feels, 0-1. 0 makes almost every slider a "
                              "straight line; 1 makes almost every slider a pronounced curve "
                              "(and makes the bow of every curved slider more pronounced too). "
                              "0.5 (default) matches the original straight/gentle-arc/pronounced-arc "
                              "mix.")
    parser.add_argument("--spacing", type=float, default=1.8,
                         help="Multiplier on jump/spacing distance (1.0 = the base distance-snap "
                              "formula; default 1.8, above the ranking-criteria-recommended "
                              "0.8x-1.3x range, since lower values still read as too close together / "
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
    # A genuinely-repeating measure (its own windowed sequence of buckets
    # recurring elsewhere, not just this one measure's bucket value
    # coincidentally matching) plays its motif from the *first*
    # occurrence's own measure index -- see motif_turn_degrees' own
    # docstring.
    measure_repeat_map = find_repeating_measure_map(measure_buckets)
    stream_mode = build_stream_runs(objects, beat_length_ms, rng, args.seed, offset_ms=offset_ms,
                                     measure_length_ms=measure_length_ms, measure_buckets=measure_buckets,
                                     stream_frequency=args.stream_frequency,
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
    # The variance itself tapers to zero as --curviness nears either
    # extreme (full strength only at 0.5) — otherwise a bucket could drift
    # *away* from an explicit 0 or 1, e.g. landing at curviness 0.15 on a
    # bucket even though the user asked for dead straight (0), producing
    # exactly the "still curvy at 0" complaint this taper fixes.
    bucket_curviness: dict[int, float] = {}
    variance_taper = min(args.curviness, 1.0 - args.curviness) / 0.5  # 0 at the extremes, 1 at 0.5
    for bucket in range(NUM_ENERGY_BUCKETS):
        bucket_rng = random.Random(f"curviness:{bucket}:{args.seed}")
        bucket_curviness[bucket] = max(0.0, min(1.0, args.curviness
                                                 + bucket_rng.uniform(-curviness_variance, curviness_variance)
                                                 * variance_taper))

    def shape_mix_for(time_ms: float) -> tuple[float, float, float, float]:
        """(curviness, straight_prob, bezier_prob, bow_scale) for the slider
        at time_ms, from its measure's own curviness level. `curviness`
        maps linearly onto how often a slider is straight at all (0 ->
        always straight, 1 -> never straight), so the two ends of the
        slider are honored exactly, not just "mostly"; whatever's left over
        is split between a gentle Bezier arc and a pronounced circular one
        the same way as before (0.5 reproduces the original ~35% straight
        / 30% gentle-Bezier / 35% pronounced-arc split)."""
        measure_index = int((time_ms - offset_ms) // measure_length_ms)
        bucket = measure_buckets.get(measure_index, 0)
        curviness = bucket_curviness.get(bucket, args.curviness)
        straight = 1.0 - curviness
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
    last_stream_mode = None  # the mode ("stack"/"line"/"flow") the just-finished run used, if any
    last_stack_anchor = None  # that run's stack spot, if it was a "stack" run — see leaving_stream below

    # Slider shape consistency within a combo: once the *first* slider in a
    # combo lands on straight or curved, every later slider in that same
    # combo (until the next new-combo) is held to the same choice — a
    # combo mixing a straight slider, a gentle Bezier, and a pronounced
    # arc back to back reads as random rather than a deliberate pattern.
    # Only the straight-vs-curved split is locked; a "curved" combo can
    # still vary between a gentle Bezier and a pronounced arc slider to
    # slider (and a chain's own polyline-vs-smooth-curve choice), so there
    # is still real shape variety from one combo to the next and within a
    # curved one, just not a jarring flip mid-phrase.
    combo_curved: bool | None = None

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
            combo_curved = None

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

        if mode is not None:
            # Tracks which mode (and, for "stack", which spot) the run this
            # object belongs to uses — read once, right as the run ends
            # (see leaving_stream below), then cleared, so a stale value
            # from several runs back can never wrongly re-fire once a
            # "line"/"flow" run has come and gone since.
            last_stream_mode = mode
            last_stack_anchor = stack_anchor if mode == "stack" else None

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
                cur_angle = next_angle(cur_angle, tier, obj.time, offset_ms, beat_length_ms, measure_length_ms, measure_buckets, rng, jitter_degrees=args.angle_jitter, measure_repeat_map=measure_repeat_map)
                cur_angle = wander_nudge(cur_angle, cur_x, cur_y)
                new_x, new_y, cur_angle = place_at_distance(cur_x, cur_y, spacing, cur_angle)
                cur_x, cur_y = clamp_to_playfield(new_x, new_y, margin=MARGIN)
                stack_anchor = (cur_x, cur_y)
            else:
                # Every other circle in this run: hold the exact same spot.
                cur_x, cur_y = stack_anchor
            last_stack_anchor = stack_anchor  # this run's just-established anchor, for leaving_stream below
        elif mode == "line":
            # The whole run moves along one fixed direction, decided once
            # when the run is first entered, so consecutive circles
            # overlap along a straight line rather than zigzagging.
            if line_run_angle is None:
                line_run_angle = next_angle(cur_angle, tier, obj.time, offset_ms, beat_length_ms,
                                             measure_length_ms, measure_buckets, rng, jitter_degrees=args.angle_jitter, measure_repeat_map=measure_repeat_map)
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
            # via a proper mirror reflection off whichever wall(s) it hit
            # (not a jump to face the playfield's dead center, which can
            # point back roughly the way the run *came from* and send it
            # straight over its own earlier members again) — a reflection
            # keeps whatever momentum the run had along the wall, the same
            # way a ball bouncing off a surface keeps moving past it rather
            # than doubling back the way it arrived. A run that doesn't hit
            # a wall never re-aims at all.
            raw_x = cur_x + spacing * math.cos(line_run_angle)
            raw_y = cur_y + spacing * math.sin(line_run_angle)
            if raw_x < MARGIN or raw_x > PLAYFIELD_W - MARGIN:
                line_run_angle = math.pi - line_run_angle
            if raw_y < MARGIN or raw_y > PLAYFIELD_H - MARGIN:
                line_run_angle = -line_run_angle
            new_x, new_y, _ = place_at_distance(cur_x, cur_y, spacing, line_run_angle)
            cur_x, cur_y = clamp_to_playfield(new_x, new_y, margin=MARGIN)
            cur_angle = line_run_angle
        elif leaving_stream and last_stream_mode == "stack" and last_stack_anchor is not None:
            # The very first object right after a "stack" run holds the
            # exact same spot as the stack itself, one time only — a
            # stack (all zero px apart) reads as one held-in-place gesture,
            # and having whatever immediately follows it jump away right
            # on its heels undercuts that read; the object *after* this
            # one goes back to normal flow. One-shot: cleared below so a
            # second stream ending later doesn't keep re-triggering it.
            cur_x, cur_y = last_stack_anchor
            last_stack_anchor = None
        else:
            # Outside a stream: normal distance-snap + motif-driven flow
            # (plus the transition boost on the one gap right after a
            # stream ends, for the same readability reason as entering one).
            spacing = max(MIN_SPACING, min(MAX_SPACING, boost * styled_spacing(gap_ms, beat_length_ms, slider_multiplier, args.spacing * spacing_scale_for(obj.time), rng)))
            cur_angle = next_angle(cur_angle, tier, obj.time, offset_ms, beat_length_ms, measure_length_ms, measure_buckets, rng, jitter_degrees=args.angle_jitter, measure_repeat_map=measure_repeat_map)
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
                                        measure_length_ms, measure_buckets, rng, jitter_degrees=args.angle_jitter, measure_repeat_map=measure_repeat_map)
                end_x, end_y, end_angle = place_at_distance(cur_x, cur_y, segment_length, end_angle)
                end_x, end_y = clamp_to_playfield(end_x, end_y, margin=MARGIN)

                _, straight_prob, bezier_prob, bow_scale = shape_mix_for(obj.time)
                # Straight-vs-curved is decided once per combo (see
                # combo_curved's own comment) — only the *first* slider of
                # a combo actually rolls for it; every later slider in the
                # same combo just inherits that choice. Bezier-vs-perfect-
                # circle still gets its own fresh roll per slider (rescaled
                # into the same [straight_prob, 1) range the original single
                # roll used, so the relative odds between them are
                # unchanged), so a curved combo still has real shape
                # variety slider to slider, just never flips to straight
                # mid-combo.
                if combo_curved is None:
                    combo_curved = rng.random() >= straight_prob
                # A tiny seeded wobble on the bow itself, same reasoning as
                # styled_spacing: keeps curves from looking mechanically
                # identical whenever curviness happens to land the same
                # shape twice, while staying reproducible for a given seed.
                bow_jitter = 1.0 + rng.uniform(-0.1, 0.1)
                if not combo_curved:
                    obj.curve_type = "L"
                    obj.points = [(end_x, end_y)]
                else:
                    mid_x, mid_y = (cur_x + end_x) / 2.0, (cur_y + end_y) / 2.0
                    perp_angle = end_angle + math.pi / 2
                    subtype_roll = rng.uniform(straight_prob, 1.0)
                    if subtype_roll < bezier_prob:
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
                        # A real circular arc: a pronounced, legible curve
                        # that actually guides the cursor around a bend.
                        # The bow is deliberately large relative to the
                        # chord specifically to stay clear of the near-
                        # collinear configuration that makes a perfect-
                        # circle curve balloon outward -- but "less likely"
                        # isn't "never," especially at high --curviness
                        # (a larger bow_scale directly widens the bow), so
                        # this is still verified for real below rather than
                        # just trusted.
                        obj.curve_type = "P"
                        bow = min(70.0 * bow_scale, segment_length * 0.45 * bow_scale) * bow_jitter
                    bow_x, bow_y = clamp_to_playfield(mid_x + bow * math.cos(perp_angle),
                                                       mid_y + bow * math.sin(perp_angle), margin=MARGIN)
                    # A P curve's three points can each individually sit in
                    # bounds while the arc actually connecting them still
                    # bulges off the playfield (p_curve_arc_bbox computes
                    # the arc's own extent, not just its points' bounding
                    # box) — a Bezier through the exact same three points
                    # is provably safe instead (always within their convex
                    # hull), so that's the fallback rather than trying to
                    # iteratively shrink the bow until it happens to fit.
                    if obj.curve_type == "P" and not p_curve_fits_playfield(
                            (cur_x, cur_y), (bow_x, bow_y), (end_x, end_y), MARGIN):
                        obj.curve_type = "B"
                    obj.points = [(bow_x, bow_y), (end_x, end_y)]

                if obj.slides % 2 == 1:
                    cur_x, cur_y, cur_angle = end_x, end_y, end_angle
                else:
                    # A bouncing slider with an *even* number of repeats
                    # ends back exactly where it started (HitObject.
                    # end_position() already accounts for this) -- cur_x/
                    # cur_y must match that, or the next object gets
                    # distance-snapped from a point the cursor was never
                    # actually left at, which is exactly what was silently
                    # breaking distance-snap right after a bounce slider.
                    cur_angle = end_angle + math.pi
            else:
                # Chain slider: walk one flow-angle segment per waypoint so
                # each note in the chain still reads as a distinct hop, then
                # decide once whether the whole chain reads as a sharp
                # polyline or one smooth curve flowing through every
                # waypoint (a Bezier through several points is still
                # guaranteed to stay within their convex hull, so this is
                # safe even for a long, sweeping chain).
                chain_curviness, _, _, _ = shape_mix_for(obj.time)
                # Same combo-locked straight-vs-curved rule as the single-
                # anchor case above (see combo_curved's own comment) — a
                # chain only ever chooses between "L" and "B" to begin
                # with, so the combo's lock applies directly with no
                # subtype re-roll needed.
                if combo_curved is None:
                    combo_curved = rng.random() < chain_curviness
                obj.curve_type = "B" if combo_curved else "L"
                new_points = []
                for _ in range(num_segments):
                    cur_angle = next_angle(cur_angle, tier, obj.time, offset_ms, beat_length_ms,
                                            measure_length_ms, measure_buckets, rng, jitter_degrees=args.angle_jitter, measure_repeat_map=measure_repeat_map)
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
