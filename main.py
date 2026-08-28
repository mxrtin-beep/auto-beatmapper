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
    <Song> (Easy).osu

Pass --osz to also zip those .osu files together with the MP3 into a
single .osz package that can be dragged straight into osu!.

Pass --restyle-only to re-run *just* Stage 3 against an already-generated
Variety file with a new --seed — apply_style.py never touches timing, type,
or object count, so this is how you get a different flow/angle pattern
(and a fresh mix of stacks/lines, slider curves, etc.) without regenerating
the rhythm at all:

    python3 main.py song.mp3 --restyle-only "out/Song (Variety).osu" --seed 123

A second, easier difficulty is derived from the Styled beatmap
(make_easy.py) automatically: lower Difficulty settings, and some of the
song's repetitive (verse/chorus-like) stream density thinned — merged into
sliders or dropped outright — plus more predictable hitsounds there.
Non-repetitive sections (a bridge, an intro/outro) are left as the Styled
difficulty had them. Pass --no-easy to skip it and produce only the three
main difficulties.
"""

from __future__ import annotations

import argparse
import os

import generate_base_beatmap
import add_variety
import apply_style
import make_easy
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
    parser.add_argument("--restyle-only", metavar="VARIETY_OSU", default=None,
                         help="Skip stages 1-2 and just re-run apply_style.py against this existing "
                              "Variety .osu with a new --seed — a way to get a different flow/angle "
                              "pattern without changing the rhythm at all.")
    parser.add_argument("--no-easy", dest="easy", action="store_false", default=True,
                         help="Skip deriving the easier difficulty from the Styled beatmap "
                              "(make_easy.py) — by default it's generated automatically.")
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
    easy_path = os.path.join(outdir, f"{title} (Easy).osu")

    if args.restyle_only:
        variety_path = args.restyle_only
        print("=== Stage 3 only: re-styling existing Variety file ===")
        _run_module_main(apply_style, [variety_path, "--output", styled_path, "--audio", args.audio,
                                        "--version", "Auto Styled", "--seed", str(args.seed)])
        output_paths = [styled_path]
        if args.easy:
            print("=== Stage 4: make easy ===")
            _run_module_main(make_easy, [styled_path, "--audio", args.audio, "--output", easy_path, "--seed", str(args.seed)])
            output_paths.append(easy_path)
        if args.osz:
            osz_path = os.path.join(outdir, f"{title}.osz")
            build_osz(output_paths, args.audio, osz_path, audio_filename)
            print(f"=== Packaged {osz_path} ===")
        print("Done.")
        return

    print("=== Stage 1: base beatmap ===")
    _run_module_main(generate_base_beatmap, [args.audio, "--output", base_path, "--title", title,
                                              "--artist", args.artist, "--creator", args.creator,
                                              "--version", "Auto Base", "--audio-filename", audio_filename])

    print("=== Stage 2: add variety ===")
    _run_module_main(add_variety, [base_path, args.audio, "--output", variety_path,
                                    "--version", "Auto Variety", "--seed", str(args.seed)])

    print("=== Stage 3: apply style ===")
    _run_module_main(apply_style, [variety_path, "--output", styled_path, "--audio", args.audio,
                                    "--version", "Auto Styled", "--seed", str(args.seed)])

    output_paths = [base_path, variety_path, styled_path]

    if args.easy:
        print("=== Stage 4: make easy ===")
        _run_module_main(make_easy, [styled_path, "--audio", args.audio, "--output", easy_path, "--seed", str(args.seed)])
        output_paths.append(easy_path)

    if args.osz:
        osz_path = os.path.join(outdir, f"{title}.osz")
        build_osz(output_paths, args.audio, osz_path, audio_filename)
        print(f"=== Packaged {osz_path} ===")

    print("Done.")


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
