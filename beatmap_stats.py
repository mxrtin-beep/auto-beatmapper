#!/usr/bin/env python3
"""
Compute descriptive statistics — whole distributions, not single numbers —
for a finished .osu file: slider length, jump/spacing distance, delay
(time between consecutive objects), slider shape mix, and a few others.
Used both as a standalone analysis tool (see main() below) and by gui.py,
which runs it against the Insane difficulty right after a beatmap is
generated and shows the result in the log box.

Usage:
    python3 beatmap_stats.py song.osu [more.osu ...]
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass, field

from beatmap_utils import HitObject, read_osu

# Curve types, per beatmap_utils.HitObject.curve_type: L(inear) is a
# straight line; B(ezier) and P(erfect circle) are both curved, just by
# different math; C(atmull) is a rare legacy curved type. "angled" here
# means a multi-anchor chain slider (several waypoints strung together,
# usually a sharp polyline even when it's technically curve_type B) as
# opposed to a single-anchor slider's plain straight/curved choice.
STRAIGHT_CURVE_TYPES = {"L"}
CURVED_CURVE_TYPES = {"B", "P", "C"}


@dataclass
class Distribution:
    """A distribution's summary: enough quantiles to describe its shape
    without dumping every raw value, plus a compact ASCII histogram."""
    n: int
    minimum: float
    p25: float
    median: float
    p75: float
    maximum: float
    mean: float
    stdev: float
    histogram: list[tuple[float, float, int]] = field(default_factory=list)

    @staticmethod
    def from_values(values: list[float], n_bins: int = 12) -> "Distribution | None":
        if not values:
            return None
        s = sorted(values)
        n = len(s)

        def quantile(p: float) -> float:
            idx = min(n - 1, max(0, round(p * (n - 1))))
            return s[idx]

        lo, hi = s[0], s[-1]
        width = (hi - lo) / n_bins if hi > lo else 1.0
        counts = [0] * n_bins
        for v in s:
            idx = min(n_bins - 1, max(0, int((v - lo) / width)))
            counts[idx] += 1
        histogram = [(lo + i * width, lo + (i + 1) * width, counts[i]) for i in range(n_bins)]

        return Distribution(
            n=n, minimum=lo, p25=quantile(0.25), median=quantile(0.5), p75=quantile(0.75),
            maximum=hi, mean=statistics.fmean(s), stdev=statistics.pstdev(s) if n > 1 else 0.0,
            histogram=histogram,
        )

    def format(self, unit: str = "", bar_width: int = 30) -> str:
        lines = [f"  n={self.n}  min={self.minimum:.1f}{unit}  p25={self.p25:.1f}{unit}  "
                 f"median={self.median:.1f}{unit}  p75={self.p75:.1f}{unit}  "
                 f"max={self.maximum:.1f}{unit}  mean={self.mean:.1f}{unit}  stdev={self.stdev:.1f}{unit}"]
        max_count = max((c for _, _, c in self.histogram), default=1)
        for lo, hi, count in self.histogram:
            bar_len = round(bar_width * count / max_count) if max_count else 0
            lines.append(f"    {lo:8.1f}-{hi:<8.1f} {'#' * bar_len}{' ' * (bar_width - bar_len)} {count}")
        return "\n".join(lines)


@dataclass
class BeatmapStats:
    source: str
    n_objects: int
    n_circles: int
    n_sliders: int
    curve_type_counts: dict[str, int]
    single_anchor_straight: int
    single_anchor_curved: int
    multi_anchor_angled: int
    stack_fraction: float  # fraction of consecutive pairs sitting within 3px of each other
    delay_ms: Distribution | None
    spacing_px: Distribution | None
    slider_length_px: Distribution | None
    slider_beats: Distribution | None
    turn_angle_deg: Distribution | None

    def format(self) -> str:
        lines = [f"=== {self.source} ===",
                 f"Objects: {self.n_objects}  (circles: {self.n_circles}, sliders: {self.n_sliders})"]

        total_single_anchor = self.single_anchor_straight + self.single_anchor_curved
        if total_single_anchor:
            straight_pct = 100.0 * self.single_anchor_straight / total_single_anchor
            curved_pct = 100.0 * self.single_anchor_curved / total_single_anchor
            lines.append(f"Single-anchor slider shape: {straight_pct:.0f}% straight, "
                         f"{curved_pct:.0f}% curved  (angled/chain sliders: {self.multi_anchor_angled})")
        curve_breakdown = ", ".join(f"{k}={v}" for k, v in sorted(self.curve_type_counts.items()))
        if curve_breakdown:
            lines.append(f"Curve types: {curve_breakdown}")
        lines.append(f"Stacked pairs (<3px apart): {100.0 * self.stack_fraction:.1f}% of consecutive pairs")

        for label, dist, unit in (
            ("Delay between consecutive objects", self.delay_ms, "ms"),
            ("Spacing (on-screen distance) between consecutive objects", self.spacing_px, "px"),
            ("Slider length", self.slider_length_px, "px"),
            ("Slider duration", self.slider_beats, " beats"),
            ("Turn angle (direction change at each object)", self.turn_angle_deg, "°"),
        ):
            if dist is not None:
                lines.append(f"{label}:")
                lines.append(dist.format(unit))
        return "\n".join(lines)


def compute_stats(osu_path: str) -> BeatmapStats:
    bm = read_osu(osu_path)
    objects = sorted(bm.hit_objects, key=lambda h: h.time)
    beat_length_ms = bm.beat_length
    slider_multiplier = bm.slider_multiplier

    n_circles = sum(1 for o in objects if not o.is_slider)
    n_sliders = sum(1 for o in objects if o.is_slider)

    curve_type_counts: dict[str, int] = {}
    single_anchor_straight = single_anchor_curved = multi_anchor_angled = 0
    slider_lengths: list[float] = []
    slider_beats: list[float] = []
    for o in objects:
        if not o.is_slider:
            continue
        curve_type_counts[o.curve_type] = curve_type_counts.get(o.curve_type, 0) + 1
        if len(o.points) > 1:
            multi_anchor_angled += 1
        elif o.curve_type in STRAIGHT_CURVE_TYPES:
            single_anchor_straight += 1
        elif o.curve_type in CURVED_CURVE_TYPES:
            single_anchor_curved += 1
        if o.length > 0:
            slider_lengths.append(o.length)
            slider_beats.append(o.length / (slider_multiplier * 100.0))

    delays = [b.time - a.time for a, b in zip(objects, objects[1:]) if b.time > a.time]
    spacings = [math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(objects, objects[1:])]
    stack_fraction = (sum(1 for d in spacings if d < 3.0) / len(spacings)) if spacings else 0.0

    turn_angles: list[float] = []
    for a, b, c in zip(objects, objects[1:], objects[2:]):
        v1x, v1y = b.x - a.x, b.y - a.y
        v2x, v2y = c.x - b.x, c.y - b.y
        n1, n2 = math.hypot(v1x, v1y), math.hypot(v2x, v2y)
        if n1 < 1.0 or n2 < 1.0:
            continue  # a stack has no defined direction -- skip rather than injecting a fake angle
        diff = (math.atan2(v2y, v2x) - math.atan2(v1y, v1x) + math.pi) % (2 * math.pi) - math.pi
        turn_angles.append(math.degrees(diff))

    return BeatmapStats(
        source=osu_path,
        n_objects=len(objects), n_circles=n_circles, n_sliders=n_sliders,
        curve_type_counts=curve_type_counts,
        single_anchor_straight=single_anchor_straight, single_anchor_curved=single_anchor_curved,
        multi_anchor_angled=multi_anchor_angled,
        stack_fraction=stack_fraction,
        delay_ms=Distribution.from_values(delays),
        spacing_px=Distribution.from_values(spacings),
        slider_length_px=Distribution.from_values(slider_lengths),
        slider_beats=Distribution.from_values(slider_beats),
        turn_angle_deg=Distribution.from_values(turn_angles),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute distribution statistics for one or more .osu files.")
    parser.add_argument("beatmaps", nargs="+", help="Path(s) to .osu file(s) to analyze.")
    args = parser.parse_args()
    for path in args.beatmaps:
        print(compute_stats(path).format())
        print()


if __name__ == "__main__":
    main()
