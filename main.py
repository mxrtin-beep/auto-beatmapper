#!/usr/bin/env python3
"""
End-to-end driver: MP3 in, a full 4-difficulty spread (and optionally a
ready-to-import .osz) out. Wires together the pipeline stages that also
work as standalone scripts:

    generate_base_beatmap.py -> add_variety.py -> apply_style.py -> make_easy.py

Usage:
    python3 main.py song.mp3 --title "Song Title" --artist "Artist Name"

This produces, in --outdir (default: output/ — every song lands in the
same flat directory, not a per-song subfolder; files are already
distinguished by their title-prefixed names), four real, playable
difficulties named the way a finished osu! beatmap set names them — no
pipeline-stage labels, just the difficulty itself:
    <Song> [Easy].osu
    <Song> [Normal].osu
    <Song> [Hard].osu
    <Song> [Insane].osu

The Base/Variety/Styled intermediate pipeline stages Insane/Hard/Normal/
Easy are all derived from are internal working files, not one of the four
difficulties — always deleted once those four are derived from them.

Pass --osz to also zip those four difficulties plus the MP3 into a single
.osz package that can be dragged straight into osu! (and delete the four
loose .osu files too, once packaged — see --keep-osu-files).

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
    parser.add_argument("--outdir", default="output",
                         help="Directory to write files into (default: output/ — every song's files "
                              "land in the same flat directory, distinguished by their own title-"
                              "prefixed names, not a per-song subfolder).")
    parser.add_argument("--osz", action="store_true", help="Also package the result into a .osz file.")
    parser.add_argument("--keep-osu-files", action="store_true",
                         help="Keep the four difficulty .osu files after building the .osz (default: "
                              "delete them, since the .osz already contains everything they hold). "
                              "Ignored if --osz isn't passed — the .osu files are the only output then. "
                              "The Base/Variety/Styled intermediate files are always deleted once the "
                              "difficulties are derived from them, regardless of this flag — see "
                              "--keep-intermediate-files for those.")
    parser.add_argument("--keep-intermediate-files", action="store_true",
                         help="Also keep the Base and Variety pipeline-stage .osu files (stages 1 and "
                              "2 — the plain beat skeleton, and that skeleton with sliders/density "
                              "variation added, both before apply_style.py's positioning) instead of "
                              "deleting them once the difficulty spread is derived. If --osz is also "
                              "passed, both are bundled into the same .osz as the four difficulties "
                              "(each keeps its own Version name, so they show up as two extra "
                              "selectable difficulties in-game) rather than just left as loose, "
                              "unimportable files; --keep-osu-files then governs whether they also "
                              "survive on disk as loose files afterward, the same as it already does "
                              "for the four difficulties. The Styled file is still always deleted — "
                              "it's redundant with the Insane difficulty, identical to it bar "
                              "Difficulty-setting clamping. Ignored with --restyle-only (stages 1-2 "
                              "aren't run then).")
    parser.add_argument("--spacing", type=float, default=None,
                         help="Forwarded to apply_style.py's --spacing (jump/spacing distance "
                              "multiplier). Omit to use apply_style.py's own default.")
    parser.add_argument("--curviness", type=float, default=None,
                         help="Forwarded to apply_style.py's --curviness (0-1, how curvy sliders "
                              "feel). Omit to use apply_style.py's own default.")
    parser.add_argument("--slider-length-bias", type=float, default=None,
                         help="Forwarded to add_variety.py's --slider-length-bias (0-1, fewer/"
                              "longer sliders vs. more/shorter ones). Omit to use add_variety.py's "
                              "own default.")
    parser.add_argument("--stream-frequency", type=float, default=None,
                         help="Forwarded to both add_variety.py's and apply_style.py's own "
                              "--stream-frequency (add_variety.py decides how often fast runs of "
                              "notes exist at all; apply_style.py decides how they're placed on "
                              "screen once they do). Omit to use each stage's own default.")
    parser.add_argument("--stack-probability", type=float, default=None,
                         help="Forwarded to apply_style.py's --stack-probability. Omit to use "
                              "apply_style.py's own default.")
    parser.add_argument("--angle-jitter", type=float, default=None,
                         help="Forwarded to apply_style.py's --angle-jitter. Omit to use "
                              "apply_style.py's own default.")
    parser.add_argument("--temperature", type=float, default=None,
                         help="Forwarded to apply_style.py's --temperature (0-1, how creative vs. "
                              "structured the styling gets). Omit to use apply_style.py's own default.")
    parser.add_argument("--bpm", type=float, default=None,
                         help="Forwarded to generate_base_beatmap.py's --bpm, to set it manually "
                              "instead of auto-detecting it. Ignored with --restyle-only.")
    parser.add_argument("--offset", type=float, default=None,
                         help="Forwarded to generate_base_beatmap.py's --offset (ms), to set it "
                              "manually instead of auto-detecting it. Ignored with --restyle-only.")
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
                         ("--stream-frequency", args.stream_frequency),
                         ("--stack-probability", args.stack_probability),
                         ("--angle-jitter", args.angle_jitter),
                         ("--temperature", args.temperature)):
        if value is not None:
            style_extra_args += [flag, str(value)]

    # add_variety.py's own --stream-frequency governs whether fast runs of
    # notes exist at all (see its own help text); apply_style.py's same-
    # named flag only decides how an already-existing run is placed on
    # screen (stacked/lined up vs. normal flow). Both need the same value
    # forwarded independently — they're two different stages' parsers.
    variety_extra_args: list[str] = []
    if args.stream_frequency is not None:
        variety_extra_args += ["--stream-frequency", str(args.stream_frequency)]
    if args.slider_length_bias is not None:
        variety_extra_args += ["--slider-length-bias", str(args.slider_length_bias)]

    title = args.title or os.path.splitext(os.path.basename(args.audio))[0]
    outdir = args.outdir
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
        tier_output_paths = derive_tiers()
        _cleanup_osu_files([styled_path], keep=False)  # never one of the four difficulties
        if args.osz:
            osz_path = os.path.join(outdir, f"{title}.osz")
            build_osz(tier_output_paths, args.audio, osz_path, audio_filename)
            print(f"=== Packaged {osz_path} ===")
            _cleanup_osu_files(tier_output_paths, args.keep_osu_files)
        print("Done.")
        return

    base_extra_args: list[str] = []
    for flag, value in (("--bpm", args.bpm), ("--offset", args.offset)):
        if value is not None:
            base_extra_args += [flag, str(value)]

    print("=== Stage 1: base beatmap ===")
    _run_module_main(generate_base_beatmap, [args.audio, "--output", base_path, "--title", title,
                                              "--artist", args.artist, "--creator", args.creator,
                                              "--version", "Auto Base", "--audio-filename", audio_filename]
                      + base_extra_args)

    print("=== Stage 2: add variety ===")
    _run_module_main(add_variety, [base_path, args.audio, "--output", variety_path,
                                    "--version", "Auto Variety", "--seed", str(args.seed)]
                      + variety_extra_args)

    print("=== Stage 3: apply style ===")
    _run_module_main(apply_style, [variety_path, "--output", styled_path, "--audio", args.audio,
                                    "--version", "Styled", "--seed", str(args.seed)] + style_extra_args)

    tier_output_paths = derive_tiers()

    # Styled is always cleaned up — it's redundant with Insane, identical to
    # it bar Difficulty-setting clamping, so there's nothing it adds even as
    # an extra in-game difficulty. Base/Variety are a different story: with
    # --keep-intermediate-files, they're genuinely worth having playable —
    # each carries its own Version name ("Auto Base"/"Auto Variety"), so
    # bundled into the same .osz as the four difficulties, they show up as
    # two additional selectable difficulties in-game rather than being
    # loose files nothing can import. Cleanup happens *after* packaging
    # (unlike Styled, which never needs to be in the package) so there's
    # still something on disk to package when --keep-intermediate-files
    # was passed; --keep-osu-files (a separate knob, same as it is for the
    # four difficulties) decides whether they also survive as loose files
    # once packaging is done.
    _cleanup_osu_files([styled_path], keep=False)
    extra_osz_files = [base_path, variety_path] if args.keep_intermediate_files else []

    if args.osz:
        osz_path = os.path.join(outdir, f"{title}.osz")
        build_osz(tier_output_paths + extra_osz_files, args.audio, osz_path, audio_filename)
        print(f"=== Packaged {osz_path} ===")
        _cleanup_osu_files(tier_output_paths, args.keep_osu_files)
        _cleanup_osu_files(extra_osz_files, args.keep_osu_files)
    else:
        _cleanup_osu_files(extra_osz_files, keep=True)  # nothing to package into -- always leave them as loose files

    print("Done.")


def _cleanup_osu_files(paths: list[str], keep: bool) -> None:
    """Delete the given .osu files. Used two ways: unconditionally (keep=False)
    for the Base/Variety/Styled intermediates, which are never one of the four
    finished difficulties; and, only after a successful build_osz() so the
    .osz is guaranteed to already hold everything they do, for the four
    difficulty files themselves — there `keep` is --keep-osu-files, letting
    the user opt out of that second cleanup."""
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
