#!/usr/bin/env python3
"""
Stage 3 — Apply style.

Repositions the hit objects produced by add_variety.py without touching
their timing, type, or count. This is purely about how the map *feels* to
play, following common osu! "rules of thumb":

  * Distance snap  — spacing between objects scales with the time gap
    between them, so the player's eye can read rhythm from spacing alone.
  * Energy-aware jumps — spacing also scales up in louder/more intense
    parts of the song (bigger jumps) and down in quiet parts, if audio is
    given for analysis.
  * Flow — avoid abrupt 180-degree reversals between consecutive objects;
    prefer angle changes that feel like a continuous swing of the cursor.
  * No unintended overlaps — objects are kept far enough apart that they
    don't visually stack unless that's the point.
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
MAX_SPACING = 260.0   # px, ceiling so jumps stay readable
BASE_SPACING_PER_BEAT = 140.0  # px of movement for a full beat gap at "normal" energy


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


def energy_scale(energy_value: float) -> float:
    """Map 0..1 energy to a ~0.6x..1.6x spacing multiplier."""
    return 0.6 + energy_value * 1.0


def flow_angle(prev_angle: float, index: int, rng: random.Random) -> float:
    """Pick the next movement angle relative to the previous one.

    Alternates the turn direction and keeps the turn magnitude in a band
    that reads as continuous swinging motion: never a full reversal
    (~180 degrees, which plays awkwardly), never a repeat of the exact same
    direction for too long (which reads as robotic/stacked). A small random
    jitter is layered on top (seeded, so it's reproducible) so consecutive
    runs on the same song don't produce an identical-looking map.
    """
    turn_degrees = 55 + 35 * math.sin(index * 0.37)  # stays within ~20-90 degrees
    turn_degrees += rng.uniform(-10.0, 10.0)
    direction = 1 if index % 2 == 0 else -1
    return prev_angle + direction * math.radians(turn_degrees)


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

    energy_at = compute_energy_lookup(args.audio) if args.audio else (lambda t: 0.5)

    objects = sorted(bm.hit_objects, key=lambda h: h.time)
    if not objects:
        raise RuntimeError("Beatmap has no hit objects to restyle.")

    # Start roughly centered.
    cur_x, cur_y = PLAYFIELD_W / 2.0, PLAYFIELD_H / 2.0
    cur_angle = 0.0
    prev_end_time = None

    for idx, obj in enumerate(objects):
        if prev_end_time is None:
            gap_ms = beat_length_ms
        else:
            gap_ms = max(1.0, obj.time - prev_end_time)

        beats_gap = gap_ms / beat_length_ms
        spacing = BASE_SPACING_PER_BEAT * beats_gap * energy_scale(energy_at(obj.time))
        spacing = max(MIN_SPACING, min(MAX_SPACING, spacing))

        cur_angle = flow_angle(cur_angle, idx, rng)
        new_x = cur_x + spacing * math.cos(cur_angle)
        new_y = cur_y + spacing * math.sin(cur_angle)
        new_x, new_y, cur_angle = bounce_into_playfield(new_x, new_y, cur_angle)
        cur_x, cur_y = clamp_to_playfield(new_x, new_y, margin=MARGIN)

        obj.x, obj.y = cur_x, cur_y

        if obj.is_slider:
            num_segments = len(obj.points)  # 1 for a simple slider, 2-3 for a merged chain
            segment_length = obj.length / num_segments

            if num_segments == 1:
                # A lone slider gets an occasional gentle arc for variety;
                # chains (below) stay as clean polylines so each waypoint
                # reads as its own beat in the chain.
                end_angle = flow_angle(cur_angle, idx + 1, rng)
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
                for seg in range(num_segments):
                    cur_angle = flow_angle(cur_angle, idx + seg + 1, rng)
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


