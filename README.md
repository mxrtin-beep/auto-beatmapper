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
   distance snap (spacing scales with the time gap between notes),
   energy-aware jump sizing, smooth flow (no abrupt 180° reversals),
   no unintended stacking, and staying within the playfield.

`beatmap_utils.py` holds the shared `.osu` parsing/writing code and data
structures used by all three stages.

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
