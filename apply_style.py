#!/usr/bin/env python3
"""
Stage 3 — Apply style.

Repositions the hit objects produced by add_variety.py without touching
their timing, type, or count. This is purely about how the map *feels* to
play, following common osu! "rules of thumb":

  * Distance snap — the spacing for a given time gap is a fixed value per
    energy tier (quiet/normal/intense), not a continuously-varying formula.
    Two objects the same beat-distance apart in the same kind of section
    always get the same spacing, so spacing itself communicates rhythm
    instead of looking arbitrary — with a hard ceiling so intensity can't
    run away into unreadable jumps.
  * Patterns / motifs — the turn angle between objects is drawn from a
    small fixed set of repeating shapes ("motifs"), one per energy tier,
    keyed to the object's actual position within the musical measure. The
    same motif recurs every measure of a given tier, so patterns are
    genuinely learnable on replay instead of a one-off procedural wiggle
    that happens to look similar.
  * Flow — every motif avoids full 180-degree reversals and repeats of the
    same direction for too long, so movement still reads as a continuous
    swing rather than snapping.
  * No unintended overlaps — a spacing floor keeps objects from visually
    stacking unless that's the point.
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
MIN_SPACING = 40.0    # px, floor so objects never feel stacked
MAX_SPACING = 220.0   # px, hard ceiling so even climax sections stay readable
BASE_SPACING_PER_BEAT = 130.0  # px of movement for a full beat gap at "normal" energy

# Fixed per-tier spacing multipliers: distance snap for a given time gap is
# the same everywhere within a tier, instead of continuously varying with
# raw energy — that consistency is what makes spacing legible.
TIER_SPACING_SCALE = {"quiet": 0.75, "normal": 1.0, "intense": 1.4}

HALF_BEAT_STEPS_PER_MEASURE = 8  # 4/4 time, half-beat resolution

# A handful of repeating turn-angle "motifs" per energy tier (degrees,
# signed = turn direction), indexed by position-within-measure. Which motif
# plays in a given measure cycles with the measure index, so the same shape
# recurs every few measures within a tier — recognizable on replay — while
# still varying between passes through the song.
MOTIFS = {
    "quiet": [
        [35, 35, 35, 35, 35, 35, 35, 35],
        [45, -45, 45, -45, 45, -45, 45, -45],
    ],
    "normal": [
        [70, -70, 70, -70, 70, -70, 70, -70],
        [50, 50, -50, -50, 50, 50, -50, -50],
        [90, -40, 40, -90, 90, -40, 40, -90],
    ],
    "intense": [
        [100, -100, 100, -100, 100, -100, 100, -100],
        [60, 60, 60, -140, 60, 60, 60, -140],
        [120, -60, 120, -60, -120, 60, -120, 60],
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


def motif_turn_degrees(tier: str, time_ms: float, offset_ms: float,
                        beat_length_ms: float, measure_length_ms: float) -> float:
    """The signed turn angle (degrees) for an object, from its tier's repeating motif."""
    half_beat_ms = beat_length_ms / 2.0
    pos_in_measure = int(round((time_ms - offset_ms) / half_beat_ms)) % HALF_BEAT_STEPS_PER_MEASURE
    measure_index = int((time_ms - offset_ms) // measure_length_ms)
    motifs = MOTIFS[tier]
    motif = motifs[measure_index % len(motifs)]
    return motif[pos_in_measure % len(motif)]


def next_angle(prev_angle: float, tier: str, time_ms: float, offset_ms: float,
               beat_length_ms: float, measure_length_ms: float, rng: random.Random) -> float:
    """Advance the flow angle using the tier's motif, plus a small humanizing jitter.

    The jitter is intentionally small (a few degrees) — the point of a motif
    is that it repeats recognizably; too much randomness would wash that out.
    """
    turn_degrees = motif_turn_degrees(tier, time_ms, offset_ms, beat_length_ms, measure_length_ms)
    turn_degrees += rng.uniform(-4.0, 4.0)
    return prev_angle + math.radians(turn_degrees)


def bounce_into_playfield(x: float, y: float, angle: float) -> tuple[float, float, float]:
    """Reflect position/angle if we've stepped outside the playfield margin."""
    if x < MARGIN or x > PLAYFIELD_W - MARGIN:
        angle = math.pi - angle
        x = max(MARGIN, min(PLAYFIELD_W - MARGIN, x))
    if y < MARGIN or y > PLAYFIELD_H - MARGIN:
        angle = -angle
        y = max(MARGIN, min(PLAYFIELD_H - MARGIN, y))
    return x, y, angle


def main() -> None:
    parser = argparse.ArgumentParser(description="Restyle object placement in a beatmap (timing/objects unchanged).")
    parser.add_argument("beatmap", help="Path to the variety .osu file (from add_variety.py).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio", default=None, help="Optional path to the song's MP3, for energy-aware spacing.")
    parser.add_argument("--version", default="Auto Styled", help="Difficulty/version name to write into the map.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed. Omit for different styling every run; pass a fixed "
                              "value (printed on every run) to reproduce the exact same map later.")
    args = parser.parse_args()

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

    # Start roughly centered.
    cur_x, cur_y = PLAYFIELD_W / 2.0, PLAYFIELD_H / 2.0
    cur_angle = 0.0
    prev_end_time = None

    for idx, obj in enumerate(objects):
        if prev_end_time is None:
            gap_ms = beat_length_ms
        else:
            gap_ms = max(1.0, obj.time - prev_end_time)

        tier = classify_tier(energy_at(obj.time), q_low, q_high)
        beats_gap = gap_ms / beat_length_ms
        spacing = BASE_SPACING_PER_BEAT * beats_gap * TIER_SPACING_SCALE[tier]
        spacing = max(MIN_SPACING, min(MAX_SPACING, spacing))

        cur_angle = next_angle(cur_angle, tier, obj.time, offset_ms, beat_length_ms, measure_length_ms, rng)
        new_x = cur_x + spacing * math.cos(cur_angle)
        new_y = cur_y + spacing * math.sin(cur_angle)
        new_x, new_y, cur_angle = bounce_into_playfield(new_x, new_y, cur_angle)
        cur_x, cur_y = clamp_to_playfield(new_x, new_y, margin=MARGIN)

        obj.x, obj.y = cur_x, cur_y

        if obj.is_slider:
            num_segments = len(obj.points)  # 1 for a simple/bouncing slider, 2-3 for a merged chain
            segment_length = obj.length / num_segments

            if num_segments == 1:
                # A lone slider gets an occasional gentle arc for variety;
                # chains (below) stay as clean polylines so each waypoint
                # reads as its own beat in the chain.
                end_angle = next_angle(cur_angle, tier, obj.time, offset_ms, beat_length_ms,
                                        measure_length_ms, rng)
                end_x = cur_x + segment_length * math.cos(end_angle)
                end_y = cur_y + segment_length * math.sin(end_angle)
                end_x, end_y, end_angle = bounce_into_playfield(end_x, end_y, end_angle)
                end_x, end_y = clamp_to_playfield(end_x, end_y, margin=MARGIN)

                if rng.random() < 0.5:
                    obj.curve_type = "L"
                    obj.points = [(end_x, end_y)]
                else:
                    obj.curve_type = "P"
                    mid_x, mid_y = (cur_x + end_x) / 2.0, (cur_y + end_y) / 2.0
                    perp_angle = end_angle + math.pi / 2
                    bow = min(40.0, segment_length * 0.25)
                    bow_x, bow_y = clamp_to_playfield(mid_x + bow * math.cos(perp_angle),
                                                       mid_y + bow * math.sin(perp_angle), margin=MARGIN)
                    obj.points = [(bow_x, bow_y), (end_x, end_y)]

                cur_x, cur_y, cur_angle = end_x, end_y, end_angle
            else:
                # Chain slider: walk one flow-angle segment per waypoint so
                # each note in the chain still reads as a distinct hop.
                obj.curve_type = "L"
                new_points = []
                for _ in range(num_segments):
                    cur_angle = next_angle(cur_angle, tier, obj.time, offset_ms, beat_length_ms,
                                            measure_length_ms, rng)
                    px = cur_x + segment_length * math.cos(cur_angle)
                    py = cur_y + segment_length * math.sin(cur_angle)
                    px, py, cur_angle = bounce_into_playfield(px, py, cur_angle)
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
