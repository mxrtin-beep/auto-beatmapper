#!/usr/bin/env python3
"""
Stage 1, v2 — Intensity-adaptive base beatmap generator.

A second, independent take on Stage 1 (generate_base_beatmap.py), which
always places one plain circle every half beat regardless of the song. This
version instead adapts *how often* a circle lands to the song's own local
loudness (RMS energy, the same signal add_variety.py already uses for its
own quiet/normal/intense split) — the base skeleton itself is denser or
sparser before add_variety.py or apply_style.py ever touch it:

  * Silent/very quiet -> no circles at all (nothing to click to).
  * Quiet             -> one circle per whole beat.
  * Normal            -> one circle per half beat (matches v1's fixed rate).
  * Intense           -> one circle per quarter beat.
  * Climax            -> one circle per eighth beat.

BPM/offset detection, PreviewTime, and the placeholder Lissajous-curve
positions are unchanged from v1 (reused directly) — this is purely about
*when* circles land, the same "timing only, no visual polish yet" scope
generate_base_beatmap.py itself has. apply_style.py's own positioning
still applies unchanged to whatever a later stage builds on top of this.

A hard cap keeps a run of fast (quarter-beat-or-closer) circles from ever
reading as an unbroken wall: once a fast run's own elapsed span would reach
a full beat, the next circle in it is dropped instead of extending the run
further. At the finest (eighth-beat) rate that caps a run at 8 circles (7
gaps of an eighth beat = 7/8 beat, one full circle short of a full beat);
at quarter-beat rate it caps at 4 (3 gaps = 3/4 beat) — both "no chain of
fast circles longer than a full beat," just expressed at whatever rate the
run actually happens to run at.

This is not yet wired into main.py's pipeline — it's a standalone
alternative to generate_base_beatmap.py for comparison, the same way
make_easy.py started as an optional add-on stage.

Usage:
    python3 generate_base_beatmap_v2.py song.mp3 --output out/song_base_v2.osu \
        --title "Song Title" --artist "Artist" --creator "Your Name"
"""

from __future__ import annotations

import argparse
import os

import librosa
import numpy as np

from beatmap_utils import HitObject, TimingPoint, default_metadata, write_osu
from add_variety import compute_energy_curve, compute_preview_time_ms, make_energy_lookup, smooth_slot_energy
from generate_base_beatmap import detect_bpm_and_offset, placeholder_positions

# Density tiers, quietest to loudest, and the beat subdivision each one
# gets: 1 = one circle per whole beat, 2 = per half beat (v1's fixed
# rate), 4 = per quarter beat, 8 = per eighth beat. "silent" has no entry
# -- it places nothing.
TIER_SUBDIVISION = {"quiet": 1, "normal": 2, "intense": 4, "climax": 8}


def classify_intensity(energy_value: float, q_silent: float, q_quiet: float,
                        q_intense: float, q_climax: float) -> str:
    if energy_value < q_silent:
        return "silent"
    if energy_value < q_quiet:
        return "quiet"
    if energy_value < q_intense:
        return "normal"
    if energy_value < q_climax:
        return "intense"
    return "climax"


def build_intensity_grid(offset_seconds: float, bpm: float, duration_seconds: float,
                          energy_at, q_silent: float, q_quiet: float, q_intense: float, q_climax: float,
                          smoothing_beats: float = 2.0) -> list[float]:
    """Return hit-object times (ms) on an eighth-beat grid, kept or dropped
    per-slot according to that slot's own (smoothed) local energy tier.

    Every slot is evaluated on the same finest (eighth-beat) grid, but a
    quieter tier only actually keeps the slots that also land on its own
    coarser subdivision (e.g. "quiet" only keeps slots that land on a whole
    beat) — the same "keep whichever slot lands on the real coarser beat,
    not just every Nth array position" trick add_variety.py's own quiet-
    section thinning uses, so a quiet section that starts on an off-beat
    eighth slot doesn't end up consistently one subdivision late.
    """
    eighth_beat_seconds = (60.0 / bpm) / 8.0
    n = int((duration_seconds - offset_seconds) / eighth_beat_seconds) + 1
    slot_times_ms = [(offset_seconds + i * eighth_beat_seconds) * 1000.0 for i in range(n)]

    raw_energy = np.array([energy_at(t) for t in slot_times_ms])
    smoothing_window = max(1, round(smoothing_beats * 8))  # beats -> eighth-beat slots
    smoothed = smooth_slot_energy(raw_energy, smoothing_window)

    kept_times: list[float] = []
    for i, t_ms in enumerate(slot_times_ms):
        tier = classify_intensity(smoothed[i], q_silent, q_quiet, q_intense, q_climax)
        if tier == "silent":
            continue
        subdivision = TIER_SUBDIVISION[tier]
        step = 8 // subdivision  # how many eighth-slots apart this tier's own grid is
        if i % step == 0:
            kept_times.append(t_ms)
    return kept_times


def cap_fast_run_span(times_ms: list[float], beat_length_ms: float,
                       quarter_beat_ms: float) -> list[float]:
    """Drop circles as needed so no run of consecutive quarter-beat-or-
    closer circles ever spans a full beat or more from its own first
    member — the finer the run's actual rate, the more circles that still
    allows (8 at eighth-beat spacing, 4 at quarter-beat), but the elapsed
    time itself never grows past one beat regardless of rate.
    """
    if not times_ms:
        return times_ms
    result = [times_ms[0]]
    run_start = times_ms[0]
    for t in times_ms[1:]:
        prev = result[-1]
        is_fast = (t - prev) <= quarter_beat_ms + 1.0
        if not is_fast:
            result.append(t)
            run_start = t
            continue
        if (t - run_start) < beat_length_ms - 1.0:
            result.append(t)
        # else: dropped -- extending the run would reach a full beat's span
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an intensity-adaptive base osu! beatmap from an MP3.")
    parser.add_argument("audio", help="Path to the input MP3 file.")
    parser.add_argument("--output", required=True, help="Path to write the .osu file to.")
    parser.add_argument("--title", default=None, help="Song title (defaults to the audio filename).")
    parser.add_argument("--artist", default="Unknown Artist")
    parser.add_argument("--creator", default="auto-beatmapper")
    parser.add_argument("--version", default="Auto Base v2", help="Difficulty/version name.")
    parser.add_argument("--audio-filename", default=None,
                         help="Value written into AudioFilename (defaults to the input file's basename).")
    parser.add_argument("--bpm", type=float, default=None,
                         help="Manually set the BPM instead of auto-detecting it.")
    parser.add_argument("--offset", type=float, default=None,
                         help="Manually set the offset (ms, time of the first beat) instead of "
                              "auto-detecting it. Any value works — it's wrapped to the equivalent "
                              "position within one beat, so e.g. -118 and 334 at 137 BPM name the "
                              "same beat and are interchangeable.")
    parser.add_argument("--silent-quantile", type=float, default=0.10,
                         help="Energy quantile below which a slot gets no circle at all (default 0.10).")
    parser.add_argument("--quiet-quantile", type=float, default=0.35,
                         help="Energy quantile below which a slot only keeps whole-beat circles "
                              "(default 0.35, matching add_variety.py's own quiet threshold).")
    parser.add_argument("--intense-quantile", type=float, default=0.75,
                         help="Energy quantile above which a slot gets quarter-beat circles "
                              "(default 0.75, matching add_variety.py's own intense threshold).")
    parser.add_argument("--climax-quantile", type=float, default=0.92,
                         help="Energy quantile above which a slot gets eighth-beat circles "
                              "(default 0.92, matching add_variety.py's own climax threshold).")
    parser.add_argument("--smoothing-beats", type=float, default=2.0,
                         help="Smooth energy over this many beats before classifying intensity, so "
                              "tiers track the song's actual sections instead of flickering slot to "
                              "slot (default 2.0; 0 disables smoothing).")
    args = parser.parse_args()

    title = args.title or os.path.splitext(os.path.basename(args.audio))[0]
    audio_filename = args.audio_filename or os.path.basename(args.audio)

    print(f"Loading audio: {args.audio}")
    y, sr = librosa.load(args.audio, sr=None, mono=True)
    duration_seconds = len(y) / sr

    if args.bpm is None or args.offset is None:
        print("Detecting BPM and offset...")
        detected_bpm, detected_offset_seconds = detect_bpm_and_offset(y, sr)
    bpm = args.bpm if args.bpm is not None else detected_bpm
    if args.offset is not None:
        offset_seconds = (args.offset / 1000.0) % (60.0 / bpm)
    else:
        offset_seconds = detected_offset_seconds
    print(f"  BPM: {bpm:.2f}" + (" (manual)" if args.bpm is not None else ""))
    print(f"  Offset: {offset_seconds * 1000:.1f} ms" + (" (manual)" if args.offset is not None else ""))

    beat_length_ms = 60000.0 / bpm
    quarter_beat_ms = beat_length_ms / 4.0

    print("Analyzing song energy...")
    times_ms, energy = compute_energy_curve(args.audio)
    energy_at = make_energy_lookup(times_ms, energy)

    # Quantiles are computed from the same eighth-beat-grid samples that get
    # classified, not the raw high-resolution energy curve -- otherwise a
    # short, extreme transient (a single hard hit) could skew a quantile in
    # a way that doesn't reflect how the *beat-grid* energy is actually
    # distributed, since most of the raw curve's samples fall between grid
    # points and are never even evaluated for classification.
    eighth_beat_seconds = (60.0 / bpm) / 8.0
    n_slots = int((duration_seconds - offset_seconds) / eighth_beat_seconds) + 1
    slot_energy = np.array([energy_at((offset_seconds + i * eighth_beat_seconds) * 1000.0)
                             for i in range(n_slots)])
    smoothing_window = max(1, round(args.smoothing_beats * 8))
    smoothed = smooth_slot_energy(slot_energy, smoothing_window)
    q_silent = float(np.quantile(smoothed, args.silent_quantile))
    q_quiet = float(np.quantile(smoothed, args.quiet_quantile))
    q_intense = float(np.quantile(smoothed, args.intense_quantile))
    q_climax = float(np.quantile(smoothed, args.climax_quantile))
    print(f"  energy quantiles -> silent<{q_silent:.3f}  quiet<{q_quiet:.3f}  "
          f"intense>{q_intense:.3f}  climax>{q_climax:.3f}")

    times = build_intensity_grid(offset_seconds, bpm, duration_seconds, energy_at,
                                  q_silent, q_quiet, q_intense, q_climax,
                                  smoothing_beats=args.smoothing_beats)
    before_cap = len(times)
    times = cap_fast_run_span(times, beat_length_ms, quarter_beat_ms)
    if before_cap != len(times):
        print(f"Capped fast runs: dropped {before_cap - len(times)} circle(s) to keep every "
              f"quarter/eighth-beat run under a full beat")
    print(f"Placing {len(times)} circles (intensity-adaptive: whole/half/quarter/eighth beat)...")

    positions = placeholder_positions(len(times))

    bm = default_metadata(title=title, artist=args.artist, creator=args.creator,
                           version=args.version, audio_filename=audio_filename)
    bm.timing_points = [TimingPoint(time=offset_seconds * 1000.0, beat_length=beat_length_ms)]

    preview_ms = compute_preview_time_ms(times_ms, energy)
    bm.general["PreviewTime"] = str(int(round(preview_ms)))

    hit_objects = []
    for i, (t, (x, y_pos)) in enumerate(zip(times, positions)):
        hit_objects.append(HitObject(x=x, y=y_pos, time=t, is_new_combo=(i % 8 == 0)))
    bm.hit_objects = hit_objects

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_osu(bm, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
