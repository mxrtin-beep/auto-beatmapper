#!/usr/bin/env python3
"""
End-to-end driver: MP3 in, a full 4-difficulty spread (and optionally a
ready-to-import .osz) out. Wires together the pipeline stages that also
work as standalone scripts:

    generate_base_beatmap.py -> add_variety.py -> apply_style.py -> make_easy.py

Usage:
    python3 main.py song.mp3 --title "Song Title" --artist "Artist Name"

This produces, in --outdir (default: output/<song name>/), four real,
playable difficulties named the way a finished osu! beatmap set names
them — no pipeline-stage labels, just the difficulty itself:
    <Song> [Easy].osu
    <Song> [Normal].osu
    <Song> [Hard].osu
    <Song> [Insane].osu

(<Song> (Base).osu, (Variety).osu, and (Styled).osu are the intermediate
pipeline stages Insane/Hard/Normal/Easy are all derived from — internal
working files, not one of the four difficulties, deleted by default once
--osz packages everything; see --keep-osu-files.)

Pass --osz to also zip those four difficulties plus the MP3 into a single
.osz package that can be dragged straight into osu!.

Pass --restyle-only to re-run *just* the styling stage against an already-
generated Variety file with a new --seed — apply_style.py never touches
timing, type, or object count, so this is how you get a different
flow/angle pattern (and a fresh mix of stacks/lines, slider curves, etc.)
without regenerating the rhythm at all:

    python3 main.py song.mp3 --restyle-only "out/Song (Variety).osu" --seed 123

The Hard/Normal/Easy difficulties are derived from Insane (make_easy.py,
run once per tier): each tier's Difficulty settings (AR/OD/HP/CS, and for
Normal/Easy, slider velocity) are clamped to osu!'s own ranking-criteria
range for that tier, and each tier thins the previous one's stream density
a bit further — Hard only in the song's *repetitive* sections (a
verse/chorus that recurs), Normal and Easy everywhere. Pass --no-spread to
skip Hard/Normal/Easy and produce only Insane.
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
    parser.add_argument("--keep-osu-files", action="store_true",
                         help="Keep the intermediate .osu files after building the .osz (default: "
                              "delete them, since the .osz already contains everything they hold). "
                              "Ignored if --osz isn't passed — the .osu files are the only output then.")
    parser.add_argument("--spacing", type=float, default=None,
                         help="Forwarded to apply_style.py's --spacing (jump/spacing distance "
                              "multiplier). Omit to use apply_style.py's own default.")
    parser.add_argument("--curviness", type=float, default=None,
                         help="Forwarded to apply_style.py's --curviness (0-1, how curvy sliders "
                              "feel). Omit to use apply_style.py's own default.")
    parser.add_argument("--stack-probability", type=float, default=None,
                         help="Forwarded to apply_style.py's --stack-probability. Omit to use "
                              "apply_style.py's own default.")
    parser.add_argument("--angle-jitter", type=float, default=None,
                         help="Forwarded to apply_style.py's --angle-jitter. Omit to use "
                              "apply_style.py's own default.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for stages 2 and 3. Omit for a different map every run; "
                              "pass a fixed value (printed on every run) to reproduce it later.")
    parser.add_argument("--restyle-only", metavar="VARIETY_OSU", default=None,
                         help="Skip stages 1-2 and just re-run apply_style.py against this existing "
                              "Variety .osu with a new --seed — a way to get a different flow/angle "
                              "pattern without changing the rhythm at all.")
    parser.add_argument("--no-spread", dest="spread", action="store_false", default=True,
                         help="Skip deriving Hard/Normal/Easy from Insane (make_easy.py) — by "
                              "default the full 4-difficulty spread is generated automatically.")
    args = parser.parse_args()

    if args.seed is None:
        import random
        args.seed = random.SystemRandom().randrange(2**32)
    print(f"Using seed: {args.seed}")

    style_extra_args: list[str] = []
    for flag, value in (("--spacing", args.spacing), ("--curviness", args.curviness),
                         ("--stack-probability", args.stack_probability),
                         ("--angle-jitter", args.angle_jitter)):
        if value is not None:
            style_extra_args += [flag, str(value)]

    title = args.title or os.path.splitext(os.path.basename(args.audio))[0]
    outdir = args.outdir or os.path.join("output", title)
    os.makedirs(outdir, exist_ok=True)

    audio_filename = os.path.basename(args.audio)
    # Intermediate pipeline stages — internal working files, not one of the
    # four finished difficulties, so they keep the old parenthetical
    # pipeline-stage naming and get cleaned up with the rest of output_paths.
    base_path = os.path.join(outdir, f"{title} (Base).osu")
    variety_path = os.path.join(outdir, f"{title} (Variety).osu")
    styled_path = os.path.join(outdir, f"{title} (Styled).osu")
    # The four finished difficulties: named the way a real, finished osu!
    # beatmap set names them — just the difficulty, no pipeline labels.
    tier_paths = {tier: os.path.join(outdir, f"{title} [{tier.capitalize()}].osu")
                  for tier in ("insane", "hard", "normal", "easy")}

    def derive_tiers() -> list[str]:
        """Run make_easy.py once per tier from styled_path, Insane first (no
        thinning, just Difficulty-setting clamping) and, unless --no-spread,
        Hard/Normal/Easy after (each thinning Insane's density further). Each
        tier gets its own seed derived from --seed so they don't all make the
        exact same random thinning/drop choices. Returns the paths written."""
        tiers = ["insane"] + (["hard", "normal", "easy"] if args.spread else [])
        written = []
        for i, tier in enumerate(tiers):
            print(f"=== Stage 4: {tier.capitalize()} ===")
            tier_seed = args.seed + i
            _run_module_main(make_easy, [styled_path, "--audio", args.audio, "--tier", tier,
                                          "--output", tier_paths[tier], "--seed", str(tier_seed)])
            written.append(tier_paths[tier])
        return written

    if args.restyle_only:
        variety_path = args.restyle_only
        print("=== Stage 3 only: re-styling existing Variety file ===")
        _run_module_main(apply_style, [variety_path, "--output", styled_path, "--audio", args.audio,
                                        "--version", "Styled", "--seed", str(args.seed)] + style_extra_args)
        output_paths = [styled_path] + derive_tiers()
        if args.osz:
            osz_path = os.path.join(outdir, f"{title}.osz")
            build_osz(output_paths, args.audio, osz_path, audio_filename)
            print(f"=== Packaged {osz_path} ===")
            _cleanup_osu_files(output_paths, args.keep_osu_files)
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
                                    "--version", "Styled", "--seed", str(args.seed)] + style_extra_args)

    output_paths = [base_path, variety_path, styled_path] + derive_tiers()

    if args.osz:
        osz_path = os.path.join(outdir, f"{title}.osz")
        build_osz(output_paths, args.audio, osz_path, audio_filename)
        print(f"=== Packaged {osz_path} ===")
        _cleanup_osu_files(output_paths, args.keep_osu_files)

    print("Done.")


def _cleanup_osu_files(paths: list[str], keep: bool) -> None:
    """Delete the intermediate .osu files once they're safely packaged into
    the .osz — the .osz already contains everything they hold, so keeping
    them around by default is just clutter. Skipped entirely if --keep-osu-files
    was passed. Only ever called after a successful build_osz(), so the .osz
    is guaranteed to exist before any .osu file is removed."""
    if keep:
        return
    for path in paths:
        try:
            os.remove(path)
        except OSError as e:
            print(f"Warning: couldn't remove {path}: {e}")


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
