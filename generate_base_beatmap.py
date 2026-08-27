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


def _low_frequency_onset_envelope(y: np.ndarray, sr: int, hop_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Onset strength restricted to the sub-200Hz band (kick drum range).

    A broadband onset envelope also fires on hi-hats, snares, and other
    percussion that can land on every subdivision of the beat, which makes
    it a poor guide to *which* subdivision is actually the downbeat. Kick
    drums overwhelmingly land on the beat itself across genres, so
    restricting to the low end gives a much cleaner signal for both BPM and
    phase (offset) than the raw waveform does.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # librosa warns about empty mel filters this low; harmless here
        S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, fmax=200, n_mels=40)
    onset_env = librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=sr, hop_length=hop_length)
    times_seconds = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    return times_seconds, onset_env


def detect_bpm_and_offset(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Estimate a constant, whole-number BPM and the offset (seconds) of the first beat.

    Real songs are essentially always a whole-number BPM, and even a
    fraction of a BPM off compounds into audible drift by the end of a
    track — so both stats are found by directly testing candidate
    whole-number BPMs against how well they line up with the song's actual
    kick-drum hits, rather than trusting a single continuous estimate:

    1. A broadband beat tracker gives a rough starting BPM guess (this can
       be off by a whole BPM or more on its own).
    2. Every integer BPM within a few of that guess is scored by how well a
       beat grid at that tempo lines up with a low-frequency ("kick drum")
       onset envelope, tried at every possible phase — the best-scoring
       (bpm, phase) pair is a much more reliable whole-number BPM than the
       broadband estimate alone.
    3. The offset is then refined further: discrete kick onset events are
       detected (with sample-accurate backtracking to each attack's actual
       start, not just its peak), and the offset is the median of how far
       each one sits from the nearest multiple of the beat length — far
       less sensitive to any single noisy detection than using the first
       beat alone, and precise enough to hold in sync for the whole track.
    """
    hop_length = 512
    times_seconds, onset_env = _low_frequency_onset_envelope(y, sr, hop_length)
    dt = times_seconds[1] - times_seconds[0]

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    guess_bpm = int(round(float(np.atleast_1d(tempo)[0])))

    def grid_score(bpm: float, offset: float) -> float:
        beat_length = 60.0 / bpm
        n_beats = int((times_seconds[-1] - offset) / beat_length)
        beat_times = offset + np.arange(n_beats) * beat_length
        idx = np.round(beat_times / dt).astype(int)
        idx = idx[(idx >= 0) & (idx < len(onset_env))]
        if len(idx) == 0:
            return 0.0
        return float(onset_env[idx].sum() / len(idx))  # per-beat average, comparable across candidate BPMs

    best_score, bpm = None, float(guess_bpm)
    for candidate_bpm in range(guess_bpm - 2, guess_bpm + 3):
        beat_length = 60.0 / candidate_bpm
        num_offsets = max(20, int(beat_length / dt) * 2)
        for offset in np.linspace(0.0, beat_length, num_offsets, endpoint=False):
            score = grid_score(candidate_bpm, offset)
            if best_score is None or score > best_score:
                best_score, bpm = score, float(candidate_bpm)

    beat_length_seconds = 60.0 / bpm
    onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length,
                                              backtrack=True, units="time")
    if len(onset_times) == 0:
        raise RuntimeError("Could not detect enough onsets in this track to estimate an offset.")

    beat_indices = np.round(onset_times / beat_length_seconds)
    residuals = onset_times - beat_indices * beat_length_seconds
    offset_seconds = float(np.median(residuals))
    # The fit can land slightly before 0 (a whole beat "before" the first
    # detected one); nudge it forward into the track if so.
    while offset_seconds < 0:
        offset_seconds += beat_length_seconds

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
