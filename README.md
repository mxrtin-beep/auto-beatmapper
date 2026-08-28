# auto-beatmapper

Automatically generates a full 4-difficulty [osu!](https://osu.ppy.sh/)
beatmap set (`.osu`) from an MP3, through a pipeline of stages that each
produce their own `.osu` file so you can inspect (or play) the map at
every step.

## Pipeline

1. **`generate_base_beatmap.py`** — analyzes the song to estimate its BPM and
   offset (the time of the first beat) — every integer BPM near a rough
   broadband guess is scored, at every possible phase, against how well a
   beat grid lines up with a low-frequency ("kick drum") onset envelope,
   giving a reliable BPM and a *close* offset; a low-frequency transient's
   onset-strength envelope peaks a real (if small) amount after the actual
   attack a player would feel as "the beat," so that offset is refined
   once more by backtracking each grid beat to the local minimum right
   before it (the same technique librosa's own onset detector uses),
   anchored to the phase already found rather than an independent guess —
   then places a hit-circle on every half beat for the whole track. Pure
   rhythm skeleton, no styling. Pass `--bpm`/`--offset` (ms) to set either
   manually instead of auto-detecting it.

2. **`add_variety.py`** — takes a base beatmap and reshapes it using the
   song's loudness (RMS energy) over time:
   - quiet sections are thinned out to one object per full beat, and a
     long (16s+) uninterrupted quiet stretch has its middle carved out
     into a real break instead — a rest for the player, and truer to how
     the song actually ebbs and flows, than clicking through minutes of
     near-nothing,
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
   overlapping line rather than a zigzag blob (never a mix of the two
   within one run), a small set of repeating turn-angle patterns keyed to
   each measure's own energy level — so a verse or chorus repeating the
   song's structure also repeats the same visual pattern — and a slow
   "wander" target that keeps the whole path migrating around the
   playfield instead of orbiting one local spot. This is the pipeline's
   hardest difficulty output, Insane.

4. **`make_easy.py`** *(runs 4 times by default, once per tier; skip with
   `--no-spread` to get just Insane)* — derives Hard, Normal, and Easy from
   Insane: each tier's Difficulty settings are clamped to osu!'s own
   ranking-criteria range for that tier, and note density is thinned a lot
   further at each step down, calibrated against a real hand-mapped
   Easy/Normal/Hard/Insane spread (`example/keha_backstabber/`) rather
   than just "somewhat less than Insane" — a real Easy never has two
   objects closer than a full beat, and leans heavily on long slider
   chains over individual clicks. `merge_gap_beats` (a quarter beat for
   Hard, up to a full beat for Easy) controls how wide a net the merge
   pass casts: **any** run of adjacent circles that close together — not
   just add_variety.py's own fast "stream" bursts — becomes one held
   slider chain instead of several clicks, with a tier-scaled probability
   of even happening at all (so density actually falls off tier to tier,
   instead of Hard converging on Normal's level), on top of a tier-scaled
   chance to drop a note outright — plus more predictable, regular
   hitsounds where it thins (only ever adding a downbeat accent, never
   silencing an existing one). Hard only thins the song's *repetitive*
   sections (a verse/chorus that recurs), lightly; Normal and Easy thin
   everywhere, progressively more. Also derives Insane itself (no
   thinning, just Difficulty-setting clamping) from the Styled output, so
   all four tiers go through the same clamping logic. SliderMultiplier is
   never touched at any tier — a real Easy's long, slow-*reading* sliders
   actually use a *higher* slider velocity (more on-screen distance per
   beat) than Insane's, not a lower one.

   The same merge-adjacent-circles-into-one-slider mechanism
   (`merge_chain` in `make_easy.py`) is a general style tool, not just a
   thinning one — reach for it anywhere a run of circles reads better as
   one held slider than several separate clicks, on any difficulty.

`beatmap_utils.py` holds the shared `.osu` parsing/writing code and data
structures used by every stage.

## Usage

The easiest way to run the whole pipeline is `main.py`:

```bash
pip install -r requirements.txt

python3 main.py song.mp3 --title "Song Title" --artist "Artist Name" --osz
```

This writes, in `--outdir` (default `output/` — every song lands in the
same flat directory, not a per-song subfolder; files are already
distinguished by their title-prefixed names):

- `<Song Title> [Easy].osu`, `[Normal].osu`, `[Hard].osu`, `[Insane].osu` —
  the four finished difficulties, named the way a real, finished osu!
  beatmap set names them (just the difficulty, in square brackets — no
  pipeline-stage labels).

The Base/Variety/Styled intermediate pipeline stages the four difficulties
are derived from are internal working files, not difficulties themselves
— always deleted once the four difficulties are derived from them,
whether or not you pass `--osz`.

With `--osz`, the four difficulties plus the MP3 are also bundled into
`<Song Title>.osz` — drag that straight into osu! to import the full
4-difficulty spread at once — and the four loose `.osu` files are deleted
too, since everything they hold is already in the `.osz`; pass
`--keep-osu-files` to keep them around alongside it.

The full spread (Hard/Normal/Easy derived from Insane) is generated by
default — pass `--no-spread` to skip that and produce only Insane:

```bash
python3 main.py song.mp3 --no-spread --osz
```

`main.py` also forwards `--spacing`, `--curviness`, `--stack-probability`,
`--angle-jitter`, and `--temperature` straight through to `apply_style.py`
(see below), and `--bpm`/`--offset` straight through to
`generate_base_beatmap.py`, if you pass them.

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

A few `apply_style.py` flags tune *how* it restyles, independent of
timing/object count/type, which are never touched either way. Every one
of these also gets a small (a few percent) wobble seeded off `--seed`, so
values don't feel mechanically identical every time the same situation
recurs — the same seed always reproduces the same wobble.

- `--spacing S` (default `1.3`, the top of the ranking-criteria-recommended
  range) — multiplier on jump/spacing distance (`1.0` = the base
  distance-snap formula). Raise it further if objects still feel too
  close together or prone to crisscrossing.
- `--temperature T` (default `0.5`, `0`-`1`) — how creative vs. structured
  the styling gets. Scales `--angle-jitter`, how much a section's
  curviness can drift from the `--curviness` baseline, and how strongly
  the path wanders around the playfield, all together — low is tight and
  predictable, high is loose and varied. Passing one of those flags
  explicitly overrides temperature's value for that one knob only.
- `--angle-jitter DEGREES` (default: derived from `--temperature`, roughly
  `1`-`10`) — how much extra random wiggle gets added on top of each
  repeating motif's turn angle, for circles and slider curves alike.
  Turning this up gives more varied flow/angles without changing when
  anything is hit; turning it down makes the motif patterns read more
  rigidly.
- `--stack-probability P` (default `0.5`) — the overall mix between stream
  runs that stack in one spot and runs that trace a straight line (`0` =
  always line, `1` = always stack). A run picks exactly one of the two for
  its whole length (never a mix), and whichever one a given repeating
  section picks stays consistent every time that section recurs. A run
  spanning more than half a beat is never a stack (only ever a line),
  matching the ranking criteria's Hard-difficulty rule against fully
  overlapping objects more than half a beat apart.
- `--curviness C` (default `0.5`) — how curvy the map feels, `0`-`1`. `0`
  makes almost every slider a straight line; `1` makes almost every
  slider a pronounced curve, and makes the bow of every curved slider
  more pronounced too. This is a *theme*, not a flat per-slider coin
  flip: each measure's energy bucket gets its own curviness level offset
  from `--curviness` (seeded, so it's consistent across a run but varies
  section to section) — a section keeps reading as consistently curvy or
  consistently straight-and-bendy, the same way a repeating chorus reuses
  the same motif.

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

## Ranking criteria compliance

`docs/osu_ranking_criteria.txt` holds osu!'s own beatmap ranking criteria
(General/Spread/Skinning + difficulty-specific rules). Concretely, against
that document:

- **Objects fully overlapping** — `apply_style.py` never mixes a stack and
  a line within one run, and a run longer than half a beat is always a
  line, never a stack — matching the Hard-difficulty rule against fully
  overlapping objects more than half a beat apart. Shorter, genuine
  same-spot stacks are kept (relying on osu!'s own stack-leniency visual
  cascade, `StackLeniency: 0.7`), matching the document's "stacks are
  acceptable" guidance.
- **Difficulty settings per tier** — `make_easy.py`'s `TIER_SETTINGS` clamps
  AR/OD/HP/CS directly to each tier's own document range, not a relative
  shift from Insane's own settings. SliderMultiplier is deliberately *not*
  clamped, even though the document's Easy/Normal guideline says to avoid
  slider velocity above 1.3 — a real reference Easy difficulty
  (`example/keha_backstabber/`) uses a SliderMultiplier of 3.54, leaning
  on long, slow-*reading* sliders rather than a cramped multiplier; that
  concrete example took priority over the document's guideline here.
- **Drain time spread** — `make_easy.py` asserts each tier's first/last
  object *end* time exactly matches Insane's (a merged slider's own start
  can be earlier than the last original object it absorbed, so the
  comparison uses each side's actual end); thinning only ever touches
  objects strictly between the first and last.
- **Objects never off-screen, snapping, timing overlaps** — enforced
  throughout (playfield margin, `slider_length_for_gap`'s rounded-gap
  derivation, `rounded_gap_ms`); see the pipeline stage docstrings.
- **Combo colours / hitsounds** — `beatmap_utils.default_metadata` sets
  three custom combo colours and every hittable edge gets an audible
  hitsound (never silent); `make_easy.py`'s `regularize_hitsounds` only
  ever *adds* a downbeat accent in a thinned measure, never removes an
  existing one, so a tier that thins broadly (Normal/Easy) can't end up
  with long silent-feeling stretches the way blanket-simplifying every
  non-downbeat hit down to plain would.
- **Known gap**: the document's Easy/Normal-tier note-density guideline
  ("mostly 1/1, 2/1, or slower") is met for objects `make_easy.py` can
  touch (plain circle runs), but not for sliders `add_variety.py` already
  built for Insane's intense sections (bounce/chain sliders) — `make_easy.py`
  only ever merges/drops *circles*, never reshapes an existing slider, so
  a fast passage that Insane already turned entirely into sliders keeps
  that same density in Easy too. Spinner rules, skinning rules, and
  BPM-scaling nuances aren't addressed — this tool doesn't generate
  spinners or skin elements at all.

## Example output

`output/Scar Tissue/` contains all three stages generated from Red Hot
Chili Peppers' "Scar Tissue" (`songs/Scar Tissue.mp3`, not committed):

- `Scar Tissue (Base).osu`
- `Scar Tissue (Variety).osu`
- `Scar Tissue (Styled).osu`

`example/keha_backstabber/` holds a real community beatmap set (audio and
images stripped out) used as the format reference while building this tool.
