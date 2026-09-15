#!/usr/bin/env python3
"""
Optional post-process — deliberately artistic patterns, in place of the
distance-snap rule.

Every earlier stage (apply_style.py above all) is built around one
invariant: the same time gap between two objects always produces the same
on-screen distance ("distance snap"). That's what makes a map *readable* —
but it also means the whole map's geometry is one long chain of jumps/
flow, and it can never do the things a human mapper reaches for on
purpose: two sliders starting from the exact same point and fanning out,
a circle placed right back on top of one from a few notes ago, a short
run of circles arranged into a clean pentagon or star instead of another
zig-zag. Those all *break* distance snap by design — that's the whole
appeal — which is exactly why they don't belong inside apply_style.py's
own flow logic and instead live here, as a separate pass you opt into
after everything else is already placed and styled.

This module only ever touches `HitObject.x`/`.y` (and, for sliders, the
anchor points in `.points` — translated/reflected as a whole, never
reshaped) on a bounded, randomly-sampled subset of objects. It never
touches timing, object count/type, slider length (so duration is
untouched), hitsounds, or combos — a beatmap run through this is still
byte-for-byte the same rhythm, just with some of its jumps replaced by a
motif.

Five motifs, chosen independently at each candidate spot:

  * "polygon"      — a run of 3-9 consecutive circles placed at the
    vertices of a regular polygon (or, for run lengths that support it,
    a star/flower — the same vertices connected in skip-one-or-two
    order, e.g. a pentagram out of 5 circles) around a center near where
    that run would otherwise have landed.
  * "constellation" — the same polygon/star treatment, but for a run of
    3-9 consecutive sliders: each keeps its own shape, just translated
    so its head lands on the vertex, so a long run of otherwise-isolated
    sliders reads as one deliberate shape instead of a chain of
    unrelated jumps.
  * "fan"          — a run of 2-4 consecutive sliders all starting from
    the same point and fanning out to their own (otherwise unchanged)
    shapes and endpoints, like a mapper reusing one anchor for several
    sliders in a row -- the other look a slider run can get, chosen
    against "constellation" at random.
  * "mirror"       — two same-type objects (both circles or both sliders)
    within a few notes of each other, where the second is placed as a
    reflection of the first across the playfield's center point or one
    of its axes (and, for sliders, given a mirrored copy of the first
    one's shape).
  * "echo"         — the next object is pulled back to start exactly
    where an earlier object ended — "this slider starts where that one
    finished", or two circles stacked on the same spot.

A circle that's 32nd-note-or-closer to its neighbor on either side is
left alone by every motif above, whether as the object a motif would
move or as a partner another motif would move to meet — those runs
already read as one continuous stream/stack, and yanking one note out of
that flow into a polygon vertex or a mirrored/echoed spot reads as a
mistake, not a motif. Eighth and sixteenth notes are fair game (see
STREAM_GAP_BEATS below).

"polygon"/"constellation" only ever claim objects that are actually
*evenly* spaced in time (see `_run_length`'s own docstring) — a run
picked purely by type/eligibility could otherwise span a real musical
gap (nothing disqualifying happened to fall between two notes a whole
phrase apart), and the resulting shape read as "the last vertex has a
weirdly long pause before it" instead of one drawn gesture.

Once a "polygon"/"constellation" lands on a given measure, the exact
same shape (not just the same *kind* of motif) gets replayed on every
other measure with the same rhythm-and-type fingerprint — the same
repetition add_sliders_v2.py's own --uniformity already gives the
underlying circle/slider layout, carried through to flair instead of
every recurrence rolling its own independent shape.

Run standalone against an already-styled/-derived .osu file:

    python3 add_flair.py "Song [Insane].osu" --output "Song [Insane].osu"

or import `apply_flair(bm, rng, probability)` and call it in-process (see
gui_v2.py's "Add artistic flair" checkbox for the intended usage — run
once per final difficulty file, after apply_style.py/make_easy.py have
already produced it).
"""

from __future__ import annotations

import argparse
import math
import random
from typing import List, Optional, Sequence, Tuple

from beatmap_utils import PLAYFIELD_H, PLAYFIELD_W, Beatmap, HitObject, clamp_to_playfield, read_osu, write_osu

# Default inset from the playfield edge for anything this module places --
# a little roomier than clamp_to_playfield's own default margin (50) since
# these are whole shapes (a polygon's radius, a slider's full anchor
# spread), not single points, and a shape that just barely clears the edge
# at one vertex reads as cramped rather than deliberate.
MARGIN = 55.0

# Regular {n} run lengths that can also be drawn as a single connected
# star/flower ({n/skip} in the usual star-polygon notation) instead of a
# plain convex polygon -- skip must be coprime with n, or the path splits
# into multiple disconnected shapes instead of one continuous line through
# every vertex.
STAR_SKIPS = {5: 2, 7: 2, 8: 3, 9: 2}
# Includes every size STAR_SKIPS knows a star for (5, 7, 8, 9), not just
# plain-polygon sizes -- a run that can only support a triangle/square
# never draws a star, but one long enough for a pentagon/heptagon/octagon
# should actually get the option, or a size those skips are keyed to would
# just never show up in practice.
POLYGON_SIZES = (3, 4, 5, 6, 7, 8, 9)
FAN_SIZES = (2, 3, 4)
# How many objects ahead "mirror"/"echo" are allowed to look for a partner
# -- far enough to skip past an object or two already claimed by another
# motif, close enough that the pairing still reads as related rather than
# two coincidentally-similar notes on opposite sides of the map.
PARTNER_LOOKAHEAD = 6

# A circle spaced this close (in beats) to its neighbor on either side --
# 32nd-note or faster -- is part of a stream/stack, not an isolated
# circle. 32nd notes (0.125 beats) are the densest thing this pipeline
# ever actually places (generate_base_beatmap_v2.py's climax bursts, one
# subdivision finer than "intense"'s own sixteenth-note rate); those runs
# already read as one continuous motion by design, and yanking one note
# out to a polygon vertex or a mirrored/echoed spot breaks that flow far
# worse than it would for a circle with normal breathing room around it,
# so every motif here leaves them alone -- both as the object a motif
# would move, and as a partner another motif would move to. Eighth and
# sixteenth notes are *not* excluded -- on a dense/stream-heavy map those
# make up most of the material flair actually has to work with, and a
# run of either still reads fine as a polygon/star/constellation once
# it's placed deliberately rather than along the usual distance-snapped
# flow. A small tolerance covers rounding in the beat-length arithmetic
# that produced the gap in the first place.
STREAM_GAP_BEATS = 0.125
STREAM_GAP_TOLERANCE_MS = 2.0


def _bbox(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _shift_into_bounds(points: Sequence[Tuple[float, float]], margin: float) -> Tuple[float, float]:
    """The (dx, dy) translation that pulls `points`' bounding box fully
    inside the margined playfield, without resizing or reshaping it --
    0 on an axis that's already in bounds."""
    x0, y0, x1, y1 = _bbox(points)
    dx = dy = 0.0
    if x1 - x0 <= PLAYFIELD_W - 2 * margin:
        if x0 < margin:
            dx = margin - x0
        elif x1 > PLAYFIELD_W - margin:
            dx = (PLAYFIELD_W - margin) - x1
    if y1 - y0 <= PLAYFIELD_H - 2 * margin:
        if y0 < margin:
            dy = margin - y0
        elif y1 > PLAYFIELD_H - margin:
            dy = (PLAYFIELD_H - margin) - y1
    return dx, dy


def _translate_object(obj: HitObject, dx: float, dy: float, margin: float) -> None:
    """Shift a hit object's head (and, for a slider, every anchor) by the
    same delta -- a pure translation, so a slider's shape/length (and so
    its duration) is completely unaffected. Anything left poking past the
    margin afterward is clamped point-by-point as a last resort only;
    `_shift_into_bounds` is what normally keeps that from being needed."""
    points = [(obj.x + dx, obj.y + dy)] + [(px + dx, py + dy) for px, py in obj.points]
    sx, sy = _shift_into_bounds(points, margin)
    obj.x, obj.y = clamp_to_playfield(obj.x + dx + sx, obj.y + dy + sy, int(margin))
    obj.points = [clamp_to_playfield(px + dx + sx, py + dy + sy, int(margin)) for px, py in obj.points]


def _reflect_point(x: float, y: float, mode: str) -> Tuple[float, float]:
    if mode == "point":
        return PLAYFIELD_W - x, PLAYFIELD_H - y
    if mode == "horizontal":
        return PLAYFIELD_W - x, y
    return x, PLAYFIELD_H - y  # "vertical"


def _reflect_offset(dx: float, dy: float, mode: str) -> Tuple[float, float]:
    """Same reflection, applied to a *relative* displacement instead of an
    absolute point -- e.g. one slider anchor's offset from its own head."""
    if mode == "point":
        return -dx, -dy
    if mode == "horizontal":
        return -dx, dy
    return dx, -dy


def _safe_center(cx: float, cy: float, radius: float, margin: float) -> Tuple[float, float]:
    """Nudge a candidate polygon/star center just far enough from the
    playfield edge that every vertex at `radius` still clears `margin`,
    without moving it any further than that's needed."""
    lo_x, hi_x = margin + radius, PLAYFIELD_W - margin - radius
    if lo_x > hi_x:
        lo_x = hi_x = PLAYFIELD_W / 2
    lo_y, hi_y = margin + radius, PLAYFIELD_H - margin - radius
    if lo_y > hi_y:
        lo_y = hi_y = PLAYFIELD_H / 2
    return min(max(cx, lo_x), hi_x), min(max(cy, lo_y), hi_y)


def _polygon_vertices(center: Tuple[float, float], radius: float, n: int,
                       start_angle: float, direction: int, skip: int) -> List[Tuple[float, float]]:
    cx, cy = center
    order = [(k * skip) % n for k in range(n)]
    verts = []
    for k in order:
        theta = start_angle + direction * 2 * math.pi * k / n
        verts.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
    return verts


# --- The four motifs ---------------------------------------------------------

def _apply_shape_run(objects: List[HitObject], start: int, n: int, rng: random.Random,
                      spec: Optional[dict] = None) -> dict:
    """Place a run of `n` objects -- circles ("polygon") or, just as well,
    sliders ("constellation": each keeps its own shape, translated so its
    head lands on the vertex) -- on a regular polygon or star/flower.
    Translation (not a direct coordinate overwrite) is what makes this
    safe for sliders too: it moves a slider's whole curve as one rigid
    piece instead of just its head, so the shape/length (and so duration)
    survive untouched.

    `spec` (radius/skip/direction/start_angle) is normally rolled fresh
    here and returned so a caller can cache it -- pass one back in to
    *replay* an earlier roll instead (same shape, same orientation, just
    re-centered on this run's own objects), which is how a repeating
    section of the map ends up with the same motif every time instead of
    a fresh independent one each time it recurs."""
    group = objects[start:start + n]
    cx = sum(o.x for o in group) / n
    cy = sum(o.y for o in group) / n
    if spec is None:
        spec = {
            "radius": rng.uniform(55.0, 120.0),
            # Weighted toward the star/flower crossing pattern (when this
            # run length actually has one) rather than a coin flip --
            # that's the "draw it like a star with a pen" look this
            # motif exists for.
            "skip": STAR_SKIPS.get(n, 1) if rng.random() < 0.65 else 1,
            "direction": rng.choice((1, -1)),
            "start_angle": rng.uniform(0.0, 2 * math.pi),
        }
    center = _safe_center(cx, cy, spec["radius"], MARGIN)
    verts = _polygon_vertices(center, spec["radius"], n, spec["start_angle"], spec["direction"], spec["skip"])
    for obj, (vx, vy) in zip(group, verts):
        _translate_object(obj, vx - obj.x, vy - obj.y, MARGIN)
    return spec


def _apply_fan(objects: List[HitObject], start: int, n: int) -> None:
    anchor = (float(objects[start].x), float(objects[start].y))
    for obj in objects[start + 1:start + n]:
        dx, dy = anchor[0] - obj.x, anchor[1] - obj.y
        _translate_object(obj, dx, dy, MARGIN)


def _apply_mirror(objects: List[HitObject], i: int, j: int, rng: random.Random) -> None:
    a, b = objects[i], objects[j]
    mode = rng.choice(("point", "horizontal", "vertical"))
    bx, by = _reflect_point(a.x, a.y, mode)
    new_bx, new_by = clamp_to_playfield(bx, by, int(MARGIN))

    if a.is_slider and b.is_slider and a.length > 0 and a.points:
        # Copying A's own anchor points onto B isn't enough on its own --
        # `length`, not the points, is what governs a slider's duration,
        # so if the copied shape's actual geometric length doesn't match
        # B's own (unchanged) `length`, osu! extrapolates the rendered
        # curve past the last anchor to make up the difference. That
        # extrapolated tail is exactly what was showing up off-screen:
        # our bounds check below only ever looked at the given anchor
        # points, never at a tail the renderer adds on its own. Scaling
        # A's offsets by B.length/A.length first (a similarity transform,
        # so it scales the curve's real geometric length by exactly the
        # same factor for any curve type this pipeline produces) makes
        # the copied shape's geometric length equal to B's declared
        # `length`, so there's nothing left for osu! to extrapolate.
        scale = b.length / a.length
        offsets = [_reflect_offset((px - a.x) * scale, (py - a.y) * scale, mode) for px, py in a.points]
        candidate = [(new_bx + ox, new_by + oy) for ox, oy in offsets]
        sx, sy = _shift_into_bounds([(new_bx, new_by)] + candidate, MARGIN)
        shifted = [(new_bx + sx, new_by + sy)] + [(px + sx, py + sy) for px, py in candidate]
        # If the shape still doesn't fit even after the best uniform
        # shift (wider than the playfield can hold at this margin), fall
        # through to the translate-only fallback below instead of
        # clamping each point independently -- that would warp the curve
        # into a shape whose *rendered* geometry no longer matches
        # `length` either, reintroducing the same off-screen risk.
        if all(MARGIN <= x <= PLAYFIELD_W - MARGIN and MARGIN <= y <= PLAYFIELD_H - MARGIN
               for x, y in shifted):
            b.curve_type = a.curve_type
            b.x, b.y = round(shifted[0][0]), round(shifted[0][1])
            b.points = [(round(x), round(y)) for x, y in shifted[1:]]
            return

    # Not a slider pair, or the copied shape couldn't be made to fit
    # cleanly -- just mirror the position. If b is a slider, keep its own
    # existing shape and translate it there (a pure rigid translation, so
    # its length/duration is exactly preserved, same as "fan"/"echo").
    if b.is_slider:
        _translate_object(b, new_bx - b.x, new_by - b.y, MARGIN)
    else:
        b.x, b.y = new_bx, new_by


def _apply_echo(objects: List[HitObject], i: int, j: int) -> None:
    anchor = objects[i].end_position()
    obj = objects[j]
    dx, dy = anchor[0] - obj.x, anchor[1] - obj.y
    _translate_object(obj, dx, dy, MARGIN)


# --- Scanning the map for candidate spots -----------------------------------

def _stream_circles(objects: Sequence[HitObject], beat_length_ms: float, slider_multiplier: float) -> List[bool]:
    """Per-object flag: is this a circle sitting at 32nd-note-or-faster
    spacing from the object right before or right after it? See
    STREAM_GAP_BEATS's own comment for why those are off-limits to every
    motif below."""
    n = len(objects)
    threshold = STREAM_GAP_BEATS * beat_length_ms + STREAM_GAP_TOLERANCE_MS
    flags = [False] * n
    for i, obj in enumerate(objects):
        if obj.is_slider or obj.is_spinner:
            continue
        if i > 0:
            prev_end = objects[i - 1].end_time(beat_length_ms, slider_multiplier)
            if obj.time - prev_end < threshold:
                flags[i] = True
                continue
        if i + 1 < n and objects[i + 1].time - obj.time < threshold:
            flags[i] = True
    return flags


def _run_length(objects: Sequence[HitObject], start: int, used: Sequence[bool],
                 stream: Sequence[bool], want_slider: bool,
                 beat_length_ms: float, slider_multiplier: float) -> int:
    """How many objects starting at `start` a shape motif (polygon/
    constellation/fan) can actually claim -- type/stream/used-eligible
    *and* evenly spaced. Eligibility alone isn't enough: two objects can
    sit next to each other in the object list with nothing disqualifying
    between them yet still be a full musical phrase apart in time (no
    slider or stream circle happened to fall between them), and a shape
    built across that gap reads as "the last vertex has a weirdly long
    pause before it" rather than one drawn gesture. Every internal gap in
    the run is snapped to its nearest eighth-of-a-beat and has to match
    the very first gap's snap exactly -- once one doesn't, the run stops
    there rather than folding in the mismatched object."""
    n = len(objects)
    length = 0
    gap_unit_ms = beat_length_ms / 8.0
    reference_bucket = None
    while (start + length < n and not used[start + length] and not stream[start + length]
           and objects[start + length].is_slider == want_slider
           and not objects[start + length].is_spinner):
        if length > 0:
            prev_end = objects[start + length - 1].end_time(beat_length_ms, slider_multiplier)
            gap = objects[start + length].time - prev_end
            bucket = round(gap / gap_unit_ms)
            if reference_bucket is None:
                reference_bucket = bucket
            elif bucket != reference_bucket:
                break
        length += 1
    return length


def _find_partner(objects: Sequence[HitObject], used: Sequence[bool], stream: Sequence[bool], i: int,
                   lookahead: int, same_type: bool) -> Optional[int]:
    n = len(objects)
    for j in range(i + 1, min(n, i + 1 + lookahead)):
        if used[j] or stream[j] or objects[j].is_spinner:
            continue
        if same_type and objects[j].is_slider != objects[i].is_slider:
            continue
        return j
    return None


def _measure_signatures(objects: Sequence[HitObject], offset_ms: float, beat_length_ms: float,
                         meter: int) -> List[Tuple]:
    """One value per object: a fingerprint of the *whole measure* it falls
    in -- every object's own position within the measure (snapped to the
    nearest 32nd note) and whether it's a slider, in time order. Two
    measures get the same fingerprint exactly when they have the same
    rhythm and object types, which is already what a repeating section of
    the map (add_sliders_v2.py's own --uniformity) produces -- this reads
    that repetition back out of the object list itself, no audio needed,
    so a shape motif applied to one occurrence can be replayed on every
    other measure that shares it (see `established` in apply_flair)."""
    measure_length_ms = max(1.0, meter * beat_length_ms)
    unit_ms = beat_length_ms / 8.0
    by_measure: dict = {}
    measure_of: List[int] = []
    for obj in objects:
        idx = int((obj.time - offset_ms) // measure_length_ms)
        measure_of.append(idx)
        rel = obj.time - offset_ms - idx * measure_length_ms
        by_measure.setdefault(idx, []).append((round(rel / unit_ms), obj.is_slider))
    signature_of_measure = {idx: tuple(items) for idx, items in by_measure.items()}
    return [signature_of_measure[idx] for idx in measure_of]


def apply_flair(bm: Beatmap, rng: random.Random, probability: float = 0.35) -> int:
    """Mutate `bm.hit_objects` in place with a randomly-sampled scatter of
    the four motifs above. `probability` is, roughly, the fraction of
    eligible spots in the map that get touched -- 0 leaves the map exactly
    as apply_style.py produced it, 1 tries a motif almost everywhere a
    motif could go. Returns how many motifs were actually applied.

    Objects are visited once, left-to-right in time order; each one that a
    motif claims (every vertex of a polygon, both ends of a mirror/echo
    pair, ...) is marked used so no later motif can also grab it -- motifs
    never overlap or nest.

    The first time a "polygon" or "constellation" lands on a given
    measure, its shape (radius/skip/direction/orientation) is remembered
    against that measure's rhythm-and-type fingerprint; every other
    measure sharing that exact fingerprint (add_sliders_v2.py's own
    --uniformity already repeats sections this way) gets the identical
    shape replayed on it -- deliberately, not re-rolled -- the same way
    apply_style.py's own repeating turn motif works. A repeat replays
    regardless of `probability`, same as the rest of that measure's
    rhythm doesn't need re-deciding either; only the *first* occurrence
    of a fingerprint is gated by it."""
    objects = bm.hit_objects
    objects.sort(key=lambda o: o.time)
    n = len(objects)
    stream = _stream_circles(objects, bm.beat_length, bm.slider_multiplier)
    meter = 4
    for tp in bm.timing_points:
        if tp.uninherited:
            meter = tp.meter
            break
    measure_sig = _measure_signatures(objects, bm.offset, bm.beat_length, meter)
    established: dict = {}
    used = [False] * n
    applied = 0

    i = 0
    while i < n:
        obj = objects[i]
        if used[i] or obj.is_spinner or stream[i]:
            i += 1
            continue

        sig = measure_sig[i]
        repeat_kind = "polygon" if not obj.is_slider else "constellation"
        spec = established.get((sig, repeat_kind))
        if spec is not None:
            run = _run_length(objects, i, used, stream, want_slider=obj.is_slider,
                               beat_length_ms=bm.beat_length, slider_multiplier=bm.slider_multiplier)
            if run >= spec["size"]:
                _apply_shape_run(objects, i, spec["size"], rng, spec=spec)
                for k in range(i, i + spec["size"]):
                    used[k] = True
                applied += 1
                i += spec["size"]
                continue

        if rng.random() >= probability:
            i += 1
            continue

        # The run-based motifs (polygon for a circle run; constellation
        # *or* fan for a slider run) go first whenever one is actually
        # available at this spot -- mirror/echo only need a single free
        # partner somewhere in the next few objects, so they succeed far
        # more often than a real 3+-long run comes along; trying them
        # first (or shuffled in) meant they kept claiming objects out from
        # under the run-based motifs before those ever got a turn, so
        # polygons/stars/constellations showed up far less than the runs
        # in the map could actually support. Between the two slider
        # looks, and between mirror/echo, order is still random.
        if not obj.is_slider:
            primary = ["polygon"]
        else:
            primary = ["constellation", "fan"]
            rng.shuffle(primary)
        rest = ["mirror", "echo"]
        rng.shuffle(rest)
        kinds = primary + rest
        run_based = {"polygon", "constellation", "fan"}
        claimed = 0
        for kind in kinds:
            if kind == "polygon":
                run = _run_length(objects, i, used, stream, want_slider=False,
                                   beat_length_ms=bm.beat_length, slider_multiplier=bm.slider_multiplier)
                sizes = [s for s in POLYGON_SIZES if s <= run]
                if not sizes:
                    continue
                size = rng.choice(sizes)
                resolved = _apply_shape_run(objects, i, size, rng)
                established[(sig, "polygon")] = {**resolved, "size": size}
                claimed = size
            elif kind == "constellation":
                run = _run_length(objects, i, used, stream, want_slider=True,
                                   beat_length_ms=bm.beat_length, slider_multiplier=bm.slider_multiplier)
                sizes = [s for s in POLYGON_SIZES if s <= run]
                if not sizes:
                    continue
                size = rng.choice(sizes)
                resolved = _apply_shape_run(objects, i, size, rng)
                established[(sig, "constellation")] = {**resolved, "size": size}
                claimed = size
            elif kind == "fan":
                run = _run_length(objects, i, used, stream, want_slider=True,
                                   beat_length_ms=bm.beat_length, slider_multiplier=bm.slider_multiplier)
                sizes = [s for s in FAN_SIZES if s <= run]
                if not sizes:
                    continue
                size = rng.choice(sizes)
                _apply_fan(objects, i, size)
                claimed = size
            elif kind == "mirror":
                partner = _find_partner(objects, used, stream, i, PARTNER_LOOKAHEAD, same_type=True)
                if partner is None:
                    continue
                _apply_mirror(objects, i, partner, rng)
                used[partner] = True
                claimed = 1
            else:  # "echo"
                partner = _find_partner(objects, used, stream, i, PARTNER_LOOKAHEAD, same_type=False)
                if partner is None:
                    continue
                _apply_echo(objects, i, partner)
                used[partner] = True
                claimed = 1

            if claimed:
                for k in range(i, i + claimed if kind in run_based else i + 1):
                    used[k] = True
                applied += 1
                i += claimed if kind in run_based else 1
                break
        else:
            i += 1

    return applied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process an already-styled .osu file with deliberate, distance-snap-breaking "
                    "patterns (polygons/stars, fanned sliders, mirrored pairs, echoed positions).")
    parser.add_argument("beatmap", help="Path to a styled/derived .osu file (apply_style.py or "
                                         "make_easy.py output -- anything with real x/y positions already).")
    parser.add_argument("--output", required=True, help="Where to write the result (may be the same path).")
    parser.add_argument("--probability", type=float, default=0.35,
                         help="Roughly the fraction of eligible spots that get a motif applied (0-1, "
                              "default 0.35). Higher values touch more of the map.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: a random one each run).")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    rng = random.Random(seed)

    bm = read_osu(args.beatmap)
    applied = apply_flair(bm, rng, probability=args.probability)
    write_osu(bm, args.output)
    print(f"Applied {applied} flair motif(s) (seed {seed}); wrote {args.output}")


if __name__ == "__main__":
    main()
