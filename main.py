#!/usr/bin/env python3
"""
End-to-end driver: MP3 in, three .osu beatmaps (and optionally a ready-to-
import .osz) out. This just wires together the three pipeline stages that
also work as standalone scripts:

    generate_base_beatmap.py  -> add_variety.py  -> apply_style.py

Usage:
    python3 main.py song.mp3 --title "Song Title" --artist "Artist Name"

This produces, in --outdir (default: output/<song name>/):
    <Song> (Base).osu
    <Song> (Variety).osu
    <Song> (Styled).osu

Pass --osz to also zip those three .osu files together with the MP3 into a
single .osz package that can be dragged straight into osu!.
"""

from __future__ import annotations

import argparse
import os

import generate_base_beatmap
import add_variety
import apply_style
from build_osz import build_osz


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a full osu! beatmap set (3 difficulties) from an MP3.")
    parser.add_argument("audio", help="Path to the input MP3 file.")
    parser.add_argument("--title", default=None, help="Song title (defaults to the audio filename).")
    parser.add_argument("--artist", default="Unknown Artist")
    parser.add_argument("--creator", default="auto-beatmapper")
    parser.add_argument("--outdir", default=None,
                         help="Directory to write the .osu files into (default: output/<title>/).")
    parser.add_argument("--osz", action="store_true", help="Also package the result into a .osz file.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for stages 2 and 3. Omit for a different map every run; "
                              "pass a fixed value (printed on every run) to reproduce it later.")
    args = parser.parse_args()

    if args.seed is None:
        import random
        args.seed = random.SystemRandom().randrange(2**32)
    print(f"Using seed: {args.seed}")

    title = args.title or os.path.splitext(os.path.basename(args.audio))[0]
    outdir = args.outdir or os.path.join("output", title)
    os.makedirs(outdir, exist_ok=True)

    audio_filename = os.path.basename(args.audio)
    base_path = os.path.join(outdir, f"{title} (Base).osu")
    variety_path = os.path.join(outdir, f"{title} (Variety).osu")
    styled_path = os.path.join(outdir, f"{title} (Styled).osu")

    print("=== Stage 1: base beatmap ===")
    y, sr = _load_audio_once(args.audio)
    bpm, offset_seconds = generate_base_beatmap.detect_bpm_and_offset(y, sr)
    print(f"  BPM: {bpm:.2f}  offset: {offset_seconds * 1000:.1f} ms")
    duration_seconds = len(y) / sr
    times = generate_base_beatmap.build_half_beat_grid(offset_seconds, bpm, duration_seconds)
    positions = generate_base_beatmap.placeholder_positions(len(times))
    bm = generate_base_beatmap.default_metadata(
        title=title, artist=args.artist, creator=args.creator, version="Auto Base", audio_filename=audio_filename)
    from beatmap_utils import HitObject, TimingPoint, write_osu
    bm.timing_points = [TimingPoint(time=offset_seconds * 1000.0, beat_length=60000.0 / bpm)]
    bm.hit_objects = [
        HitObject(x=x, y=y_pos, time=t, is_new_combo=(i % 8 == 0))
        for i, (t, (x, y_pos)) in enumerate(zip(times, positions))
    ]
    write_osu(bm, base_path)
    print(f"  wrote {base_path} ({len(times)} circles)")

    print("=== Stage 2: add variety ===")
    _run_module_main(add_variety, [base_path, args.audio, "--output", variety_path,
                                    "--version", "Auto Variety", "--seed", str(args.seed)])

    print("=== Stage 3: apply style ===")
    _run_module_main(apply_style, [variety_path, "--output", styled_path, "--audio", args.audio,
                                    "--version", "Auto Styled", "--seed", str(args.seed)])

    if args.osz:
        osz_path = os.path.join(outdir, f"{title}.osz")
        build_osz([base_path, variety_path, styled_path], args.audio, osz_path, audio_filename)
        print(f"=== Packaged {osz_path} ===")

    print("Done.")


def _load_audio_once(audio_path: str):
    import librosa
    return librosa.load(audio_path, sr=None, mono=True)


def _run_module_main(module, argv: list[str]) -> None:
    """Invoke a stage script's own main() with a specific argv, so main.py
    stays a thin wrapper instead of re-implementing each stage's logic."""
    import sys
    old_argv = sys.argv
    sys.argv = [module.__file__] + argv
    try:
        module.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
