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
from beatmap_utils import Beatmap, PLAYFIELD_H, PLAYFIELD_W, extract_osz, guess_tier, read_osu

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

# "Note density should consist of mostly <rhythm>, or slower" -- each tier's
# own median-delay floor (in beats) and the rhythm description to report,
# straight out of the Guidelines subsection for that tier.
NOTE_DENSITY_GUIDELINE = {
    "easy": (0.9, "1/1, 2/1, or slower"),
    "normal": (0.45, "1/1, occasional 1/2, or slower"),
    "hard": (0.22, "1/2, occasional 1/4, or slower"),
}

# "When distance snap is used, try to keep it between <lo>x and <hi>x"
# (Easy/Normal guideline). A map's *actual* distance snap is approximated
# as its median spacing_per_beat divided by SliderMultiplier*100 (the
# px-per-beat a 1.0x snap would move at that slider velocity).
DISTANCE_SNAP_RANGE = {"easy": (0.8, 1.2), "normal": (0.8, 1.3)}

# "Ensure that your combos are not unreasonably short or long" (Overall
# guideline, every tier). No numeric threshold is given in the criteria
# text itself, so these are a generous sanity range, not a strict reading
# of the rule -- flagged only when a map's *typical* (median) combo is
# well outside what any real mapset would use.
REASONABLE_COMBO_LENGTH = (2, 24)

# "Spinners must be long enough for Auto to achieve 1000 bonus score"
# (General rule). The real threshold depends on OD; 4 beats is a simple,
# tier-independent floor used as a decidable proxy.
SPINNER_MIN_BEATS_GENERAL = 4.0

# "There should be at least <N> beats between a spinner's end and the next
# object" (difficulty-specific guideline; no requirement stated for
# Insane/Expert).
SPINNER_GAP_BEATS = {"easy": 4.0, "normal": 2.0, "hard": 1.0}

# Beat subdivisions a sliderend "should be snapped according to" (straight
# beat: 1/2, 1/4, 1/8, 1/16; swing beat: 1/3, 1/6, 1/12) -- checked together
# since a plain .osu file doesn't record which feel the song actually uses.
SNAP_DIVISORS = (1.0, 1/2, 1/3, 1/4, 1/6, 1/8, 1/12, 1/16)
SNAP_TOLERANCE_MS = 3.0

# "Avoid using combo colours ... with ~50 luminosity or lower" (General
# guideline). Luminosity here is the standard perceived-brightness formula,
# scaled to 0-100.
DARK_LUMINOSITY_THRESHOLD = 50.0

# "Avoid slider-only sections" (Easy/Normal guideline) -- a run of at least
# this many consecutive sliders with no circle or rest moment between them.
SLIDER_ONLY_RUN_THRESHOLD = {"easy": 3, "normal": 4}

# "Slider tick hitsounds are discouraged" (Hard/Insane/Expert guideline).
# Not directly decidable from a .osu file (skin/sample driven), so this
# uses SliderTickRate > 1 as the closest available proxy: a tick rate
# above the default 1 makes slider ticks fire more often, and so be more
# audible, than the guideline's "used sparingly" framing assumes.
TICK_DISCOURAGED_TIERS = {"hard", "insane", "expert"}

# "Frequently manipulating slider velocity is discouraged" (Easy/Normal
# guideline) -- flagged once a map's inherited (green) timing points imply
# more than this many actual slider-velocity *changes* (not just lines).
SV_CHANGE_TIERS = {"easy", "normal"}
SV_CHANGE_THRESHOLD = 3


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
    """Note density guideline (Easy/Normal/Hard, each with its own rhythm
    floor -- see NOTE_DENSITY_GUIDELINE)."""
    floor = NOTE_DENSITY_GUIDELINE.get(tier)
    if floor is None or stats.delay_beats is None:
        return None
    min_beats, rhythm_desc = floor
    median = stats.delay_beats.median
    if median >= min_beats:
        return Finding("Note density guideline", "Guideline", PASS,
                        f"Median delay is {median:.2f} beats, consistent with {rhythm_desc} rhythms.")
    return Finding("Note density guideline", "Guideline", WARN,
                    f"Median delay is {median:.2f} beats -- denser than the {rhythm_desc} rhythms "
                    f"recommended for {tier.capitalize()}.")


def _check_distance_snap(bm: Beatmap, stats: BeatmapStats, tier: str) -> Finding | None:
    """Easy/Normal guideline: keep distance snap within a tier-specific
    multiple of SliderMultiplier's own px-per-beat, so spacing and slider
    velocity don't imply contradictory speeds to a new player."""
    snap_range = DISTANCE_SNAP_RANGE.get(tier)
    if snap_range is None or stats.spacing_per_beat is None:
        return None
    lo, hi = snap_range
    px_per_beat_at_1x = bm.slider_multiplier * 100.0
    if px_per_beat_at_1x <= 0:
        return None
    actual_snap = stats.spacing_per_beat.median / px_per_beat_at_1x
    if lo <= actual_snap <= hi:
        return Finding("Distance snap guideline", "Guideline", PASS,
                        f"Median spacing implies a ~{actual_snap:.2f}x distance snap, within the "
                        f"recommended {lo:g}x-{hi:g}x range for {tier}.")
    return Finding("Distance snap guideline", "Guideline", WARN,
                    f"Median spacing implies a ~{actual_snap:.2f}x distance snap, outside the "
                    f"recommended {lo:g}x-{hi:g}x range for {tier} -- spacing and slider velocity may "
                    f"read as inconsistent.")


def _check_combo_length(stats: BeatmapStats) -> Finding | None:
    """Overall guideline: combos should be neither unreasonably short nor
    unreasonably long (see REASONABLE_COMBO_LENGTH's own docstring on why
    this range is a generous sanity check, not a criteria-specified one)."""
    if stats.combo_length is None:
        return None
    lo, hi = REASONABLE_COMBO_LENGTH
    median = stats.combo_length.median
    if lo <= median <= hi:
        return Finding("Combo length guideline", "Guideline", PASS,
                        f"Median combo length is {median:.0f} objects, a reasonable size.")
    which = "short" if median < lo else "long"
    return Finding("Combo length guideline", "Guideline", WARN,
                    f"Median combo length is {median:.0f} objects, unusually {which} -- combos should "
                    f"reflect musical phrasing (bars, vocal/instrumental phrases), not read as arbitrary.")


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


def _parse_combo_colours(bm: Beatmap) -> list[tuple[int, int, int]]:
    """RGB triples for every `ComboN : r,g,b` line under [Colours] --
    SliderBorder/SliderTrackOverride and any malformed line are skipped."""
    colours = []
    for line in bm.colours:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if not key.strip().startswith("Combo"):
            continue
        try:
            r, g, b = (int(v.strip()) for v in value.strip().split(",")[:3])
        except ValueError:
            continue
        colours.append((r, g, b))
    return colours


def _check_combo_colour_count(bm: Beatmap) -> Finding:
    """Rule: at least two different custom combo colours (unless the
    default skin is forced, which no custom Combo lines at all implies)."""
    distinct = set(_parse_combo_colours(bm))
    if not distinct:
        return Finding("Combo colours", "Rule", PASS,
                        "No custom combo colours set -- the default skin's colours apply.")
    if len(distinct) >= 2:
        return Finding("Combo colours", "Rule", PASS,
                        f"{len(distinct)} distinct custom combo colour(s) defined.")
    return Finding("Combo colours", "Rule", FAIL,
                    "Only 1 distinct custom combo colour is defined; at least 2 are required "
                    "unless the default skin is forced.")


def _check_combo_colour_luminosity(bm: Beatmap) -> Finding | None:
    """Guideline: avoid combo colours at ~50 luminosity or lower."""
    colours = _parse_combo_colours(bm)
    if not colours:
        return None
    def luminosity(rgb: tuple[int, int, int]) -> float:
        r, g, b = rgb
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0 * 100.0
    dark = sorted({c for c in colours if luminosity(c) <= DARK_LUMINOSITY_THRESHOLD})
    if not dark:
        return Finding("Combo colour brightness guideline", "Guideline", PASS,
                        f"No combo colours read at or below {DARK_LUMINOSITY_THRESHOLD:g} luminosity.")
    listing = ", ".join(f"({r},{g},{b})" for r, g, b in dark)
    return Finding("Combo colour brightness guideline", "Guideline", WARN,
                    f"{len(dark)} combo colour(s) at or below {DARK_LUMINOSITY_THRESHOLD:g} luminosity "
                    f"({listing}) -- dark colours hurt approach-circle readability at high background dim.")


def _check_hitsound_audibility(bm: Beatmap) -> Finding:
    """Rule: every actively clicked part must have an audible hitsound.
    Approximated via timing-point volume, the one thing in the file that
    can make a hitsound outright silent regardless of what sample plays."""
    tps = sorted(((tp.time, tp.volume) for tp in bm.timing_points), key=lambda t: t[0])
    if not tps:
        return Finding("Audible hitsounds", "Rule", WARN,
                        "No timing points found; can't verify hitsound volume.")
    silent = 0
    for ho in bm.hit_objects:
        volume = tps[0][1]
        for t, v in tps:
            if t <= ho.time + 0.5:
                volume = v
            else:
                break
        if volume <= 0:
            silent += 1
    if silent == 0:
        return Finding("Audible hitsounds", "Rule", PASS,
                        "Every hit object falls under a timing section with nonzero hitsound volume.")
    return Finding("Audible hitsounds", "Rule", FAIL,
                    f"{silent}/{len(bm.hit_objects)} hit object(s) fall under a timing section with "
                    f"volume=0 -- silent, with no feedback when clicked.")


def _check_sliderend_snapping(bm: Beatmap) -> Finding | None:
    """Guideline: sliderends not representing a specific musical sound
    should be snapped to the beat grid (1/2, 1/4, 1/8, 1/16, or the swing
    equivalents 1/3, 1/6, 1/12). A .osu file can't say which sliderends are
    meant to land on a real musical cue vs. just be beat-snapped, so this
    flags any sliderend that isn't close to *either* kind of grid point --
    a slider deliberately ending exactly on a musical hit will typically
    still land near the grid anyway, since that's usually itself a snapped
    beat position."""
    sliders = [o for o in bm.hit_objects if o.is_slider]
    if not sliders:
        return None
    beat_length, offset, slider_multiplier = bm.beat_length, bm.offset, bm.slider_multiplier
    if not beat_length:
        return None
    unsnapped = 0
    for o in sliders:
        end = o.end_time(beat_length, slider_multiplier)
        rel = (end - offset) % beat_length
        deviation = min(min(abs(rel - d * beat_length) for d in SNAP_DIVISORS), abs(rel - beat_length))
        if deviation > SNAP_TOLERANCE_MS:
            unsnapped += 1
    if unsnapped == 0:
        return Finding("Slider end snapping guideline", "Guideline", PASS,
                        f"All {len(sliders)} slider end(s) land within {SNAP_TOLERANCE_MS:g}ms of a "
                        f"recognized beat subdivision.")
    frac = 100.0 * unsnapped / len(sliders)
    return Finding("Slider end snapping guideline", "Guideline", WARN,
                    f"{unsnapped}/{len(sliders)} slider end(s) (~{frac:.0f}%) don't land on a recognized "
                    f"beat subdivision (1/2, 1/3, 1/4, 1/6, 1/8, 1/12, 1/16) -- sliderends not tied to a "
                    f"specific sound in the music should be snapped to the beat structure.")


def _check_hitsound_feedback(bm: Beatmap) -> Finding | None:
    """Guideline: spinner ends, slider ends, and slider reverses should
    have hitsound feedback -- approximated as: is *any* such edge point in
    the whole map using a nonzero hitsound addition at all. A map with
    literally none anywhere is the one confidently flaggable case; how
    often any individual map *should* use one varies by held-sound
    exceptions the file itself can't distinguish."""
    edge_sounds: list[int] = []
    for o in bm.hit_objects:
        if o.is_slider:
            sounds = o.edge_hitsounds or [0] * (o.slides + 1)
            edge_sounds.extend(sounds[1:])  # exclude the head -- covered by the general audibility check
        elif o.is_spinner:
            edge_sounds.append(o.hitsound)
    if not edge_sounds:
        return None
    with_feedback = sum(1 for h in edge_sounds if h != 0)
    if with_feedback > 0:
        return Finding("Slider/spinner end hitsound feedback guideline", "Guideline", PASS,
                        f"{with_feedback}/{len(edge_sounds)} slider-end/reverse/spinner-end point(s) "
                        f"carry an explicit hitsound addition.")
    return Finding("Slider/spinner end hitsound feedback guideline", "Guideline", WARN,
                    f"None of {len(edge_sounds)} slider-end/reverse/spinner-end point(s) carry an "
                    f"explicit hitsound addition -- add feedback unless these represent a held sound.")


def _check_slider_tick_hitsound(bm: Beatmap, tier: str) -> Finding | None:
    if tier not in TICK_DISCOURAGED_TIERS:
        return None
    tick_rate = float(bm.difficulty.get("SliderTickRate", 1))
    if tick_rate <= 1:
        return Finding("Slider tick hitsound guideline", "Guideline", PASS,
                        f"SliderTickRate={tick_rate:g} keeps slider ticks (and their hitsound) "
                        f"infrequent on {tier}.")
    return Finding("Slider tick hitsound guideline", "Guideline", WARN,
                    f"SliderTickRate={tick_rate:g} on {tier} -- slider tick hitsounds are discouraged "
                    f"here; if used, make sure their volume is balanced (notably quieter) against "
                    f"regular hitsounds.")


def _check_spinner_length(bm: Beatmap) -> Finding | None:
    """Rule: spinners must be long enough for Auto to reach 1000 bonus
    score -- approximated as a flat beat-length floor (see
    SPINNER_MIN_BEATS_GENERAL's own docstring)."""
    spinners = [o for o in bm.hit_objects if o.is_spinner]
    beat_length = bm.beat_length
    if not spinners or not beat_length:
        return None
    short = [(o.spinner_end_time - o.time) / beat_length for o in spinners]
    short = [d for d in short if d < SPINNER_MIN_BEATS_GENERAL]
    if not short:
        return Finding("Spinner length", "Rule", PASS,
                        f"All {len(spinners)} spinner(s) are at least {SPINNER_MIN_BEATS_GENERAL:g} "
                        f"beats long.")
    return Finding("Spinner length", "Rule", FAIL,
                    f"{len(short)}/{len(spinners)} spinner(s) are shorter than "
                    f"{SPINNER_MIN_BEATS_GENERAL:g} beats (shortest: {min(short):.2f}) -- too short "
                    f"for Auto to achieve 1000 bonus score.")


def _check_spinner_gap(bm: Beatmap, tier: str) -> Finding | None:
    min_gap = SPINNER_GAP_BEATS.get(tier)
    spinners = [o for o in bm.hit_objects if o.is_spinner]
    beat_length = bm.beat_length
    if min_gap is None or not spinners or not beat_length:
        return None
    objects_sorted = sorted(bm.hit_objects, key=lambda h: h.time)
    violations = 0
    for spinner in spinners:
        following = next((h for h in objects_sorted
                           if h.time >= spinner.spinner_end_time and h is not spinner), None)
        if following is None:
            continue
        gap_beats = (following.time - spinner.spinner_end_time) / beat_length
        if gap_beats < min_gap:
            violations += 1
    if violations == 0:
        return Finding("Spinner recovery gap guideline", "Guideline", PASS,
                        f"Every spinner has at least {min_gap:g} beats before the next object "
                        f"(required on {tier}).")
    return Finding("Spinner recovery gap guideline", "Guideline", WARN,
                    f"{violations} spinner(s) have less than {min_gap:g} beats before the next object "
                    f"-- not enough time to recognize a hit object after spinning, on {tier}.")


def _check_slider_only_sections(bm: Beatmap, tier: str) -> Finding | None:
    threshold = SLIDER_ONLY_RUN_THRESHOLD.get(tier)
    if threshold is None:
        return None
    objects = sorted(bm.hit_objects, key=lambda h: h.time)
    longest = run = offending_runs = 0
    for o in objects:
        if o.is_slider:
            run += 1
            continue
        if run >= threshold:
            offending_runs += 1
        longest = max(longest, run)
        run = 0
    if run >= threshold:
        offending_runs += 1
    longest = max(longest, run)
    if offending_runs == 0:
        return Finding("Slider-only section guideline", "Guideline", PASS,
                        f"Longest all-slider run is {longest} slider(s) (guideline: avoid "
                        f"{threshold}+ in a row with no circle or rest, on {tier}).")
    return Finding("Slider-only section guideline", "Guideline", WARN,
                    f"{offending_runs} run(s) of {threshold}+ consecutive sliders with no circle or "
                    f"rest between them (longest: {longest}) -- can be tiring to aim/follow on {tier}.")


def _check_frequent_sv_changes(bm: Beatmap, tier: str) -> Finding | None:
    if tier not in SV_CHANGE_TIERS:
        return None
    green_lines = sorted((tp for tp in bm.timing_points if not tp.uninherited), key=lambda tp: tp.time)
    if len(green_lines) < 2:
        return Finding("Slider velocity stability guideline", "Guideline", PASS,
                        "No (or only one) slider-velocity change found.")

    def sv_of(tp) -> float:
        return -100.0 / tp.beat_length if tp.beat_length < 0 else 1.0

    transitions = 0
    prev: float | None = None
    for tp in green_lines:
        sv = sv_of(tp)
        if prev is not None and abs(sv - prev) > 1e-6:
            transitions += 1
        prev = sv
    if transitions <= SV_CHANGE_THRESHOLD:
        return Finding("Slider velocity stability guideline", "Guideline", PASS,
                        f"{transitions} slider-velocity change(s) found -- within a reasonable range "
                        f"for {tier}.")
    return Finding("Slider velocity stability guideline", "Guideline", WARN,
                    f"{transitions} slider-velocity changes found -- frequently manipulating slider "
                    f"velocity is discouraged on {tier}; reserve SV changes for real pacing changes.")


def judge_beatmap(osu_path: str, tier: str) -> list[Finding]:
    tier = tier.lower()
    bm = read_osu(osu_path)
    stats = compute_stats(osu_path)

    findings: list[Finding] = []
    findings.extend(_check_difficulty_settings(bm, tier))
    findings.append(_check_off_screen(bm))
    findings.append(_check_full_overlap(stats, bm, tier))
    findings.append(_check_combo_colour_count(bm))
    findings.append(_check_hitsound_audibility(bm))
    for f in (_check_slider_velocity(bm, tier), _check_streams(bm, tier),
              _check_note_density(stats, tier), _check_short_sliders(stats, tier),
              _check_distance_snap(bm, stats, tier), _check_combo_length(stats),
              _check_combo_colour_luminosity(bm), _check_sliderend_snapping(bm),
              _check_hitsound_feedback(bm), _check_slider_tick_hitsound(bm, tier),
              _check_spinner_length(bm), _check_spinner_gap(bm, tier),
              _check_slider_only_sections(bm, tier), _check_frequent_sv_changes(bm, tier)):
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
    parser.add_argument("beatmap", help="Path to the .osu file to judge, or a packaged .osz -- every "
                                          "difficulty inside whose Version: names a recognized tier "
                                          "(easy/normal/hard/insane/expert) is judged against its own tier.")
    parser.add_argument("--tier", default=None,
                         help="Difficulty tier to judge against (easy/normal/hard/insane/expert). "
                              "Required for a single .osu input (default: insane); for a .osz input, "
                              "restricts judging to just this one recognized tier instead of every "
                              "tier found inside it.")
    args = parser.parse_args()

    if args.beatmap.lower().endswith(".osz"):
        osu_paths = extract_osz(args.beatmap)
        tier_paths: dict[str, str] = {}
        for path in osu_paths:
            tier = guess_tier(path)
            if tier is not None and tier not in tier_paths:  # first match wins over a same-tier guest diff
                tier_paths[tier] = path
        if args.tier:
            if args.tier.lower() not in tier_paths:
                raise SystemExit(f"No difficulty matching --tier {args.tier!r} found inside {args.beatmap}. "
                                  f"Found: {', '.join(sorted(tier_paths)) or '(none recognized)'}")
            tier_paths = {args.tier.lower(): tier_paths[args.tier.lower()]}
        if not tier_paths:
            raise SystemExit(f"No recognizable difficulty names (easy/normal/hard/insane/expert) found "
                              f"inside {args.beatmap}.")
        for tier, path in tier_paths.items():
            findings = judge_beatmap(path, tier)
            print(format_findings(findings, tier, path))
            print()
        return

    tier = args.tier or "insane"
    findings = judge_beatmap(args.beatmap, tier)
    print(format_findings(findings, tier, args.beatmap))


if __name__ == "__main__":
    main()
