# auto-beatmapper

Automatically generates an [osu!](https://osu.ppy.sh/) beatmap (`.osu`) from
an MP3, in three stages that each produce their own `.osu` file so you can
inspect (or play) the map at every step.

## Pipeline

1. **`generate_base_beatmap.py`** — analyzes the song to estimate its BPM and
   offset (the time of the first beat), then places a hit-circle on every
   half beat for the whole track. Pure rhythm skeleton, no styling.

2. **`add_variety.py`** — takes a base beatmap and reshapes it using the
   song's loudness (RMS energy) over time:
   - quiet sections are thinned out to one object per full beat,
   - normal sections get some adjacent circle pairs combined into sliders,
   - intense sections get extra circles inserted on quarter/eighth-beat
     subdivisions.

   Objects are never allowed to overlap in time (a slider "occupies" the
   timeline for its whole duration).

3. **`apply_style.py`** — repositions every object (without changing its
   timing, type, or count) following common mapping rules of thumb:
   distance snap (on-screen distance for a half-beat-or-larger gap exactly
   matches what a slider spanning that time would travel), a fast
   quarter-beat-or-less run of circles reading as a deliberate stack or an
   overlapping line rather than a zigzag blob, and a small set of
   repeating turn-angle patterns keyed to each measure's own energy level
   — so a verse or chorus repeating the song's structure also repeats the
   same visual pattern, instead of a one-off procedural wiggle.

4. **`make_easy.py`** *(optional, via `--easy`)* — derives a second, easier
   difficulty from the Styled beatmap: lower Difficulty settings, and some
   of the *repetitive* sections' stream density thinned into sliders
   (using the same positions apply_style.py already chose). Non-repetitive
   sections — a bridge, an intro/outro, anything that only happens once —
   are left as they were.

`beatmap_utils.py` holds the shared `.osu` parsing/writing code and data
structures used by all four stages.

## Usage

The easiest way to run the whole pipeline is `main.py`:

```bash
pip install -r requirements.txt

python3 main.py song.mp3 --title "Song Title" --artist "Artist Name" --osz
```

This writes `output/<Song Title>/<Song Title> (Base|Variety|Styled).osu`,
and with `--osz` also bundles those three difficulties plus the MP3 into
`output/<Song Title>/<Song Title>.osz` — drag that straight into osu! to
import all three difficulties at once.

Add `--easy` to also generate a fourth, easier difficulty (`make_easy.py`,
see above) and include it in the `.osz`:

```bash
python3 main.py song.mp3 --easy --osz
```

### Re-styling without changing the rhythm

`apply_style.py` never touches timing, object type, or object count — only
where things are placed. To get a different flow/angle pattern (a fresh
mix of stacks vs. overlapping lines, different slider curves, etc.)
without regenerating the beatmap's rhythm at all, re-run just that stage
with a new seed:

```bash
python3 apply_style.py "out/Song (Variety).osu" \
    --output "out/Song (Styled2).osu" --audio song.mp3 --seed 123

# or, via main.py:
python3 main.py song.mp3 --restyle-only "out/Song (Variety).osu" --seed 123
```

### Randomness / seeds

`add_variety.py` and `apply_style.py` make a lot of small randomized
choices (which eligible circle pairs become sliders, how the flow angle
jitters) so running the pipeline twice on the same song gives you a
different-feeling map each time. Every run prints the seed it used
(`Using seed: 123456789`); pass `--seed 123456789` back in to reproduce
that exact map again:

```bash
python3 main.py song.mp3 --seed 123456789 --osz
```

`main.py` picks one random seed per run and forwards it to both stages, so
a single `--seed` value reproduces the whole pipeline's output.

### Building a .osz from existing .osu files

If you already have `.osu` file(s) — from a previous run, or hand-edited —
and just want a playable package without regenerating anything:

```bash
python3 build_osz.py song.mp3 "out/Song (Base).osu" "out/Song (Variety).osu" \
    "out/Song (Styled).osu" --output "out/Song.osz"
```

`build_osz.py` also exposes a `build_osz()` function you can import and
call directly from other Python code.

### Running the stages individually

Each stage is also its own script, useful if you want to inspect or tweak
the output of one stage before feeding it to the next:

```bash
python3 generate_base_beatmap.py song.mp3 \
    --output "out/Song (Base).osu" \
    --title "Song Title" --artist "Artist Name" --creator "Your Name"

python3 add_variety.py "out/Song (Base).osu" song.mp3 \
    --output "out/Song (Variety).osu"

python3 apply_style.py "out/Song (Variety).osu" \
    --output "out/Song (Styled).osu" --audio song.mp3
```

To turn any of those into a `.osz` yourself: put the `.osu` file(s) and the
source MP3 (renamed to match `AudioFilename` in the map) in the same
folder, zip it, and rename the zip's extension to `.osz`.

## Example output

`output/Scar Tissue/` contains all three stages generated from Red Hot
Chili Peppers' "Scar Tissue" (`songs/Scar Tissue.mp3`, not committed):

- `Scar Tissue (Base).osu`
- `Scar Tissue (Variety).osu`
- `Scar Tissue (Styled).osu`

`example/keha_backstabber/` holds a real community beatmap set (audio and
images stripped out) used as the format reference while building this tool.
