#!/usr/bin/env python3
"""
Package one or more .osu files plus their audio into a .osz — the zip
format osu! actually imports (drag it into the client, or double-click it).

Useful on its own once you already have .osu file(s) from the pipeline (or
edited by hand) and just want a playable package, without re-running the
whole generator.

Usage:
    python3 build_osz.py song.mp3 "out/Song (Base).osu" "out/Song (Variety).osu" \
        "out/Song (Styled).osu" --output "out/Song.osz"

    # or grab every .osu in a folder:
    python3 build_osz.py song.mp3 out/*.osu --output out/Song.osz
"""

from __future__ import annotations

import argparse
import os
import zipfile


def build_osz(osu_paths: list[str], audio_path: str, osz_path: str, audio_filename: str | None = None) -> str:
    """Zip the given .osu file(s) together with the audio file into osz_path.

    `audio_filename` is the name written inside the archive (and must match
    each .osu's `AudioFilename:` line) — defaults to audio_path's basename.
    Returns osz_path.
    """
    if not osu_paths:
        raise ValueError("Need at least one .osu file to package.")
    audio_filename = audio_filename or os.path.basename(audio_path)

    os.makedirs(os.path.dirname(os.path.abspath(osz_path)) or ".", exist_ok=True)
    with zipfile.ZipFile(osz_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in osu_paths:
            zf.write(path, arcname=os.path.basename(path))
        zf.write(audio_path, arcname=audio_filename)
    return osz_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package .osu file(s) and their audio into a .osz.")
    parser.add_argument("audio", help="Path to the song's audio file (e.g. an MP3).")
    parser.add_argument("osu_files", nargs="+", help="One or more .osu files to include.")
    parser.add_argument("--output", required=True, help="Path to write the .osz to.")
    parser.add_argument("--audio-filename", default=None,
                         help="Name to store the audio under inside the archive "
                              "(defaults to the audio file's own name; must match each "
                              ".osu's AudioFilename: line).")
    args = parser.parse_args()

    osz_path = build_osz(args.osu_files, args.audio, args.output, args.audio_filename)
    print(f"Wrote {osz_path}")


if __name__ == "__main__":
    main()
