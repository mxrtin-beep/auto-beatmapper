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

Four motifs, chosen independently at each candidate spot:

  * "polygon" — a run of 3-6 consecutive circles placed at the vertices
    of a regular polygon (or, for run lengths that support it, a star/
    flower — the same vertices connected in skip-one-or-two order, e.g.
    a pentagram out of 5 circles) around a center near where that run
    would otherwise have landed.
  * "fan"     — a run of 2-4 consecutive sliders all starting from the
    same point and fanning out to their own (otherwise unchanged) shapes
    and endpoints, like a mapper reusing one anchor for several sliders
    in a row.
  * "mirror"  — two same-type objects (both circles or both sliders)
    within a few notes of each other, where the second is placed as a
    reflection of the first across the playfield's center point or one
    of its axes (and, for sliders, given a mirrored copy of the first
    one's shape).
  * "echo"    — the next object is pulled back to start exactly where an
    earlier object ended — "this slider starts where that one finished",
    or two circles stacked on the same spot.

A circle that's eighth-note-or-closer to its neighbor on either side is
left alone by every motif above, whether as the object a motif would move
or as a partner another motif would move to meet — those runs already
read as one continuous stream/stack, and yanking one note out of that
flow into a polygon vertex or a mirrored/echoed spot reads as a mistake,
not a motif (see STREAM_GAP_BEATS below).

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
# eighth-note or faster -- is part of a stream/stack, not an isolated
# circle. Those runs already read as one continuous motion by design (see
# add_variety.py's own "stream"/climax handling); yanking one note out to
# a polygon vertex or a mirrored/echoed spot breaks that flow far worse
# than it would for a circle with normal breathing room around it, so
# every motif here leaves stream circles alone -- both as the object a
# motif would move, and as a partner another motif would move to. A small
# tolerance covers rounding in the beat-length arithmetic that produced
# the gap in the first place.
STREAM_GAP_BEATS = 0.5
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

def _apply_polygon(objects: List[HitObject], start: int, n: int, rng: random.Random) -> None:
    group = objects[start:start + n]
    cx = sum(o.x for o in group) / n
    cy = sum(o.y for o in group) / n
    radius = rng.uniform(55.0, 120.0)
    center = _safe_center(cx, cy, radius, MARGIN)
    # Weighted toward the star/flower crossing pattern (when this run
    # length actually has one) rather than a coin flip -- that's the
    # "draw it like a star with a pen" look this motif exists for.
    skip = STAR_SKIPS.get(n, 1) if rng.random() < 0.65 else 1
    direction = rng.choice((1, -1))
    start_angle = rng.uniform(0.0, 2 * math.pi)
    verts = _polygon_vertices(center, radius, n, start_angle, direction, skip)
    for obj, (x, y) in zip(group, verts):
        obj.x, obj.y = clamp_to_playfield(x, y, int(MARGIN))


def _apply_fan(objects: List[HitObject], start: int, n: int) -> None:
    anchor = (float(objects[start].x), float(objects[start].y))
    for obj in objects[start + 1:start + n]:
        dx, dy = anchor[0] - obj.x, anchor[1] - obj.y
        _translate_object(obj, dx, dy, MARGIN)


def _apply_mirror(objects: List[HitObject], i: int, j: int, rng: random.Random) -> None:
    a, b = objects[i], objects[j]
    mode = rng.choice(("point", "horizontal", "vertical"))
    bx, by = _reflect_point(a.x, a.y, mode)
    b.x, b.y = clamp_to_playfield(bx, by, int(MARGIN))
    if a.is_slider and b.is_slider:
        b.curve_type = a.curve_type
        new_points = []
        for px, py in a.points:
            ox, oy = _reflect_offset(px - a.x, py - a.y, mode)
            new_points.append((b.x + ox, b.y + oy))
        sx, sy = _shift_into_bounds([(b.x, b.y)] + new_points, MARGIN)
        b.x, b.y = clamp_to_playfield(b.x + sx, b.y + sy, int(MARGIN))
        b.points = [clamp_to_playfield(px + sx, py + sy, int(MARGIN)) for px, py in new_points]


def _apply_echo(objects: List[HitObject], i: int, j: int) -> None:
    anchor = objects[i].end_position()
    obj = objects[j]
    dx, dy = anchor[0] - obj.x, anchor[1] - obj.y
    _translate_object(obj, dx, dy, MARGIN)


# --- Scanning the map for candidate spots -----------------------------------

def _stream_circles(objects: Sequence[HitObject], beat_length_ms: float, slider_multiplier: float) -> List[bool]:
    """Per-object flag: is this a circle sitting at eighth-note-or-faster
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
                 stream: Sequence[bool], want_slider: bool) -> int:
    n = len(objects)
    length = 0
    while (start + length < n and not used[start + length] and not stream[start + length]
           and objects[start + length].is_slider == want_slider
           and not objects[start + length].is_spinner):
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


def apply_flair(bm: Beatmap, rng: random.Random, probability: float = 0.35) -> int:
    """Mutate `bm.hit_objects` in place with a randomly-sampled scatter of
    the four motifs above. `probability` is, roughly, the fraction of
    eligible spots in the map that get touched -- 0 leaves the map exactly
    as apply_style.py produced it, 1 tries a motif almost everywhere a
    motif could go. Returns how many motifs were actually applied.

    Objects are visited once, left-to-right in time order; each one that a
    motif claims (every vertex of a polygon, both ends of a mirror/echo
    pair, ...) is marked used so no later motif can also grab it -- motifs
    never overlap or nest."""
    objects = bm.hit_objects
    objects.sort(key=lambda o: o.time)
    n = len(objects)
    stream = _stream_circles(objects, bm.beat_length, bm.slider_multiplier)
    used = [False] * n
    applied = 0

    i = 0
    while i < n:
        obj = objects[i]
        if used[i] or obj.is_spinner or stream[i] or rng.random() >= probability:
            i += 1
            continue

        # The run-based motif (polygon for a circle run, fan for a slider
        # run) goes first whenever it's actually available at this spot --
        # mirror/echo only need a single free partner somewhere in the next
        # few objects, so they succeed far more often than a real 3+-long
        # run comes along; trying them first (or shuffled in) meant they
        # kept claiming objects out from under polygon/fan before those
        # ever got a turn, so polygons/stars showed up far less than the
        # runs in the map could actually support. mirror and echo still
        # split any leftover chance at this spot in random order.
        rest = ["mirror", "echo"]
        rng.shuffle(rest)
        kinds = ["polygon" if not obj.is_slider else "fan"] + rest
        claimed = 0
        for kind in kinds:
            if kind == "polygon":
                run = _run_length(objects, i, used, stream, want_slider=False)
                sizes = [s for s in POLYGON_SIZES if s <= run]
                if not sizes:
                    continue
                size = rng.choice(sizes)
                _apply_polygon(objects, i, size, rng)
                claimed = size
            elif kind == "fan":
                run = _run_length(objects, i, used, stream, want_slider=True)
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
                for k in range(i, i + claimed if kind in ("polygon", "fan") else i + 1):
                    used[k] = True
                applied += 1
                i += claimed if kind in ("polygon", "fan") else 1
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
