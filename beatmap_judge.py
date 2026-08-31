#!/usr/bin/env python3
"""
Rules-of-thumb / ranking-criteria judgment for a finished .osu file.

beatmap_stats.py describes *what a map measures as*; this module describes
*whether that's good*, by checking the same measurements against the
concrete, tier-specific thresholds in docs/osu_ranking_criteria.txt (the
"General", "Spread", and "Difficulty-specific" sections). Only checks that
are actually decidable from the .osu file's own numbers are implemented —
anything the criteria ties to the *music* (whether a circle sits on a real
musical cue, whether hitsounds are audible, difficulty spikes relative to a
song's intensity, ...) can't be judged from geometry/timing alone and is
intentionally left out rather than faked.

Each check produces a Finding: which criteria clause it's checking, whether
it's a hard Rule or a soft Guideline, a pass/warn/fail verdict, the measured
value, and a human-readable explanation. Usage:

    python3 beatmap_judge.py mymap.osu --tier insane
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from beatmap_stats import BeatmapStats, compute_stats
from beatmap_utils import Beatmap, PLAYFIELD_H, PLAYFIELD_W, read_osu

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

# Difficulty-setting guideline ranges, straight out of the "Difficulty
# setting guidelines" subsection for each tier. (lo, hi) inclusive.
DIFFICULTY_SETTING_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "easy":   {"ApproachRate": (0, 5), "OverallDifficulty": (1, 3), "HPDrainRate": (1, 3), "CircleSize": (0, 4)},
    "normal": {"ApproachRate": (4, 6), "OverallDifficulty": (3, 5), "HPDrainRate": (3, 5), "CircleSize": (0, 5)},
    "hard":   {"ApproachRate": (6, 8), "OverallDifficulty": (5, 7), "HPDrainRate": (4, 6), "CircleSize": (0, 6)},
    "insane": {"ApproachRate": (7, 9.3), "OverallDifficulty": (7, 9), "HPDrainRate": (5, 8), "CircleSize": (0, 7)},
    "expert": {"ApproachRate": (8, 10), "OverallDifficulty": (8, 10), "HPDrainRate": (5, 10), "CircleSize": (0, 7)},
}

# "Objects <X> apart or less must not fully overlap" -- the beat-fraction
# threshold below which two non-slider-linked objects stacking exactly (the
# stack_fraction stat) is a rule violation rather than an intentional stack.
FULL_OVERLAP_BEAT_THRESHOLD = {"easy": 1.0, "normal": 1.0, "hard": 0.5, "insane": 0.25, "expert": 0.0}

# "Avoid slider velocity above 1.3" (Easy/Normal guideline).
MAX_RECOMMENDED_SLIDER_VELOCITY = {"easy": 1.3, "normal": 1.3}

# "Avoid streams made of more than 5 notes" (Hard guideline). A "stream" here
# is approximated as a run of consecutive 1/4-beat-or-faster gaps.
MAX_STREAM_LEN = {"hard": 5}
STREAM_GAP_BEATS = 0.26  # slightly above exact 1/4 to absorb rounding


@dataclass
class Finding:
    clause: str          # short name of the criteria clause being checked
    kind: str            # "Rule" or "Guideline"
    verdict: str         # PASS | WARN | FAIL
    detail: str          # human-readable explanation with the measured value


def _check_difficulty_settings(bm: Beatmap, tier: str) -> list[Finding]:
    findings = []
    ranges = DIFFICULTY_SETTING_RANGES.get(tier)
    if not ranges:
        return findings
    for key, (lo, hi) in ranges.items():
        raw = bm.difficulty.get(key)
        if raw is None:
            findings.append(Finding(f"{key} guideline", "Guideline", WARN, f"{key} not set in [Difficulty]."))
            continue
        value = float(raw)
        if lo <= value <= hi:
            findings.append(Finding(f"{key} guideline", "Guideline", PASS,
                                     f"{key}={value:g} is within the recommended {lo:g}-{hi:g} range for {tier}."))
        else:
            findings.append(Finding(f"{key} guideline", "Guideline", WARN,
                                     f"{key}={value:g} is outside the recommended {lo:g}-{hi:g} range for {tier}."))
    return findings


def _check_off_screen(bm: Beatmap) -> Finding:
    """Rule: hit objects must never be off-screen (4:3 playfield).

    Only the object's own clickable anchors -- its head, and for a slider
    its far end (`points[-1]`) -- are checked, not every intermediate
    Bezier/Catmull control point: those steer a curve without necessarily
    lying on it themselves, so an off-playfield control point on an
    otherwise on-screen slider path is not itself a ranking violation.
    """
    offending = 0
    for ho in bm.hit_objects:
        points = [(ho.x, ho.y)]
        if ho.is_slider and ho.points:
            points.append(ho.points[-1])
        if any(x < 0 or x > PLAYFIELD_W or y < 0 or y > PLAYFIELD_H for x, y in points):
            offending += 1
    if offending == 0:
        return Finding("Hit objects off-screen", "Rule", PASS, "No hit objects fall outside the 512x384 playfield.")
    return Finding("Hit objects off-screen", "Rule", FAIL,
                    f"{offending} hit object(s) have a point outside the 512x384 playfield.")


def _check_full_overlap(stats: BeatmapStats, bm: Beatmap, tier: str) -> Finding:
    """Rule: objects within a tier-specific beat gap must not fully overlap
    (a genuine stack, not just close spacing -- approximated the same way
    beatmap_stats.py's stack_fraction does, via <3px consecutive spacing)."""
    threshold = FULL_OVERLAP_BEAT_THRESHOLD.get(tier, 0.0)
    if threshold <= 0 or stats.delay_beats is None:
        return Finding("Full overlap of close objects", "Rule", PASS,
                        f"No overlap threshold applies at {tier} (Insane/Expert allow tight stacks by design).")
    objects = sorted(bm.hit_objects, key=lambda h: h.time)
    beat_length = bm.beat_length
    offenders = 0
    for a, b in zip(objects, objects[1:]):
        gap_beats = (b.time - a.time) / beat_length if beat_length else 999
        if gap_beats <= threshold:
            spacing = ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5
            if spacing < 3.0:
                offenders += 1
    if offenders == 0:
        return Finding("Full overlap of close objects", "Rule", PASS,
                        f"No object pairs {threshold:g} beat(s) or less apart fully overlap.")
    return Finding("Full overlap of close objects", "Rule", FAIL,
                    f"{offenders} object pair(s) {threshold:g} beat(s) or less apart fully overlap "
                    f"(not allowed at {tier}).")


def _check_slider_velocity(bm: Beatmap, tier: str) -> Finding | None:
    cap = MAX_RECOMMENDED_SLIDER_VELOCITY.get(tier)
    if cap is None:
        return None
    sv = bm.slider_multiplier
    if sv <= cap:
        return Finding("Slider velocity guideline", "Guideline", PASS,
                        f"SliderMultiplier={sv:g} is at or below the recommended {cap:g} cap for {tier}.")
    return Finding("Slider velocity guideline", "Guideline", WARN,
                    f"SliderMultiplier={sv:g} exceeds the recommended {cap:g} cap for {tier}.")


def _check_streams(bm: Beatmap, tier: str) -> Finding | None:
    max_len = MAX_STREAM_LEN.get(tier)
    if max_len is None:
        return None
    objects = sorted(bm.hit_objects, key=lambda h: h.time)
    beat_length = bm.beat_length
    if not beat_length:
        return None
    longest = run = 0
    worst_examples = 0
    for a, b in zip(objects, objects[1:]):
        gap_beats = (b.time - a.time) / beat_length
        if gap_beats <= STREAM_GAP_BEATS:
            run += 1
        else:
            if run + 1 > max_len:
                worst_examples += 1
            longest = max(longest, run + 1)
            run = 0
    if run + 1 > max_len:
        worst_examples += 1
    longest = max(longest, run + 1)
    if worst_examples == 0:
        return Finding("Stream length guideline", "Guideline", PASS,
                        f"Longest 1/4-or-faster run is {longest} notes (guideline: avoid more than {max_len}).")
    return Finding("Stream length guideline", "Guideline", WARN,
                    f"{worst_examples} run(s) of 1/4-or-faster notes exceed {max_len} notes "
                    f"(longest: {longest}); short reversing sliders are the suggested alternative.")


def _check_note_density(stats: BeatmapStats, tier: str) -> Finding | None:
    """Easy guideline: note density should be mostly 1/1, 2/1, or slower."""
    if tier != "easy" or stats.delay_beats is None:
        return None
    median = stats.delay_beats.median
    if median >= 0.9:
        return Finding("Note density guideline", "Guideline", PASS,
                        f"Median delay is {median:.2f} beats, consistent with 1/1-or-slower rhythms.")
    return Finding("Note density guideline", "Guideline", WARN,
                    f"Median delay is {median:.2f} beats -- denser than the 1/1-or-slower rhythms "
                    f"recommended for Easy.")


def _check_short_sliders(stats: BeatmapStats, tier: str) -> Finding | None:
    """Easy guideline: avoid sliders shorter than 1/2 of a beat."""
    if tier != "easy" or stats.slider_beats is None:
        return None
    total = sum(c for _, _, c in stats.slider_beats.histogram)
    under = sum(c for lo, hi, c in stats.slider_beats.histogram if hi <= 0.5)
    if total == 0 or under == 0:
        return Finding("Short sliders guideline", "Guideline", PASS,
                        "No sliders shorter than 1/2 beat found.")
    frac = 100.0 * under / total
    return Finding("Short sliders guideline", "Guideline", WARN,
                    f"~{frac:.0f}% of sliders read shorter than 1/2 beat, which new players may not "
                    f"be able to read on Easy.")


def judge_beatmap(osu_path: str, tier: str) -> list[Finding]:
    tier = tier.lower()
    bm = read_osu(osu_path)
    stats = compute_stats(osu_path)

    findings: list[Finding] = []
    findings.extend(_check_difficulty_settings(bm, tier))
    findings.append(_check_off_screen(bm))
    findings.append(_check_full_overlap(stats, bm, tier))
    for f in (_check_slider_velocity(bm, tier), _check_streams(bm, tier),
              _check_note_density(stats, tier), _check_short_sliders(stats, tier)):
        if f is not None:
            findings.append(f)
    return findings


def format_findings(findings: list[Finding], tier: str, source: str) -> str:
    lines = [f"=== Ranking-criteria judgment: {source} ({tier}) ==="]
    n_fail = sum(1 for f in findings if f.verdict == FAIL)
    n_warn = sum(1 for f in findings if f.verdict == WARN)
    n_pass = sum(1 for f in findings if f.verdict == PASS)
    lines.append(f"{n_pass} pass, {n_warn} warn, {n_fail} fail (of {len(findings)} checks)")
    for f in findings:
        lines.append(f"  [{f.verdict:4s}] ({f.kind}) {f.clause}: {f.detail}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge a .osu file against osu! ranking criteria rules "
                                                   "of thumb, for the checks decidable from the file alone.")
    parser.add_argument("beatmap", help="Path to the .osu file to judge.")
    parser.add_argument("--tier", default="insane",
                         help="Difficulty tier to judge against (easy/normal/hard/insane/expert).")
    args = parser.parse_args()
    findings = judge_beatmap(args.beatmap, args.tier)
    print(format_findings(findings, args.tier, args.beatmap))


if __name__ == "__main__":
    main()
