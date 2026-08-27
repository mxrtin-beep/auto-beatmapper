#!/usr/bin/env python3
"""
Stage 1 — Base beatmap generator.

Analyzes an MP3 to determine its BPM and offset (the time of the first
beat), then lays down a plain hit-circle on every *half beat* for the
length of the track. This is the rhythmic skeleton that add_variety.py and
apply_style.py build on — it deliberately does nothing clever with note
placement or object variety yet.

Usage:
    python3 generate_base_beatmap.py song.mp3 --output out/song_base.osu \
        --title "Song Title" --artist "Artist" --creator "Your Name"
"""

from __future__ import annotations

import argparse
import math
import os

import librosa
import numpy as np

from beatmap_utils import HitObject, PLAYFIELD_H, PLAYFIELD_W, TimingPoint, default_metadata, write_osu


def detect_bpm_and_offset(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Estimate a constant BPM and the offset (seconds) of the first beat.

    librosa's beat tracker gives us a full grid of beat times; we take the
    median inter-beat interval (robust to a handful of missed/extra beats)
    to get the BPM, and the first detected beat as the offset.
    """
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames", trim=False)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) < 2:
        raise RuntimeError("Could not detect enough beats in this track to estimate a BPM.")

    intervals = np.diff(beat_times)
    median_interval = float(np.median(intervals))
    bpm = 60.0 / median_interval

    offset_seconds = float(beat_times[0])
    return bpm, offset_seconds


def build_half_beat_grid(offset_seconds: float, bpm: float, duration_seconds: float) -> list[float]:
    """Return hit-object times (ms), one every half beat, from the offset to the end of the track."""
    half_beat_seconds = 30.0 / bpm  # (60 / bpm) / 2
    times = []
    t = offset_seconds
    while t < duration_seconds:
        times.append(t * 1000.0)
        t += half_beat_seconds
    return times


def placeholder_positions(n: int) -> list[tuple[int, int]]:
    """A simple, deterministic path across the playfield.

    Stage 1 only cares about *timing* being correct — visual polish is
    apply_style.py's job — so this just walks a gentle Lissajous-style curve
    that keeps consecutive circles a comfortable, non-overlapping distance
    apart without any music awareness.
    """
    cx, cy = PLAYFIELD_W / 2, PLAYFIELD_H / 2
    rx, ry = PLAYFIELD_W / 2 - 60, PLAYFIELD_H / 2 - 60
    positions = []
    for i in range(n):
        angle = i * 0.6
        x = cx + rx * math.sin(angle)
        y = cy + ry * math.sin(angle * 1.5 + 1.0)
        positions.append((int(round(x)), int(round(y))))
    return positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a base half-beat osu! beatmap from an MP3.")
    parser.add_argument("audio", help="Path to the input MP3 file.")
    parser.add_argument("--output", required=True, help="Path to write the .osu file to.")
    parser.add_argument("--title", default=None, help="Song title (defaults to the audio filename).")
    parser.add_argument("--artist", default="Unknown Artist")
    parser.add_argument("--creator", default="auto-beatmapper")
    parser.add_argument("--version", default="Auto Base", help="Difficulty/version name.")
    parser.add_argument("--audio-filename", default=None,
                         help="Value written into AudioFilename (defaults to the input file's basename).")
    args = parser.parse_args()

    title = args.title or os.path.splitext(os.path.basename(args.audio))[0]
    audio_filename = args.audio_filename or os.path.basename(args.audio)

    print(f"Loading audio: {args.audio}")
    y, sr = librosa.load(args.audio, sr=None, mono=True)
    duration_seconds = len(y) / sr

    print("Detecting BPM and offset...")
    bpm, offset_seconds = detect_bpm_and_offset(y, sr)
    print(f"  BPM: {bpm:.2f}")
    print(f"  Offset: {offset_seconds * 1000:.1f} ms")

    times = build_half_beat_grid(offset_seconds, bpm, duration_seconds)
    print(f"Placing {len(times)} circles (one per half beat)...")

    positions = placeholder_positions(len(times))

    bm = default_metadata(title=title, artist=args.artist, creator=args.creator,
                           version=args.version, audio_filename=audio_filename)
    bm.timing_points = [TimingPoint(time=offset_seconds * 1000.0, beat_length=60000.0 / bpm)]

    hit_objects = []
    for i, (t, (x, y_pos)) in enumerate(zip(times, positions)):
        hit_objects.append(HitObject(x=x, y=y_pos, time=t, is_new_combo=(i % 8 == 0)))
    bm.hit_objects = hit_objects

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_osu(bm, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
