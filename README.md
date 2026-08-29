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
   giving a reliable BPM and a *close* offset; that offset is then refined
   the way a mapper sets it by eye in the editor's waveform view — a
   kick's amplitude rises fast and decays slowly, so the marker goes right
   at the peak — by finding the actual local amplitude peak of the raw
   (low-passed, same kick band) waveform near each grid beat and nudging
   the offset by the median correction — then places a hit-circle on
   every half beat for the whole track. Pure rhythm skeleton, no styling.
   Pass `--bpm`/`--offset` (ms) to set either manually instead of
   auto-detecting it — any value works for `--offset`, since it's wrapped
   to the equivalent position within one beat (e.g. `-118` and `334` at
   137 BPM name the same beat).

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
   Insane by deleting objects, nothing else: no merging, no reshaping. An
   object either survives exactly as apply_style.py placed it, or it's
   gone entirely — never partially edited into something new, which used
   to be exactly what produced "bad spacing" complaints from checkers (a
   merged slider's geometry doesn't automatically read as correctly
   spaced the way two untouched, already-validated objects do). Circles on
   a quarter/eighth-beat subdivision are deleted first and most often —
   the fast subdivisions a lower tier has the least business keeping —
   then half-beat circles, then (rarely) whole-beat circles or whole
   sliders, at a tier-scaled probability per category. A downbeat is never
   deleted, so `regularize_hitsounds` can always find something to accent
   every measure (only ever *adding* an accent there, never silencing an
   existing one elsewhere) — no gap without a hitsound. A deterministic
   final pass also clears any pair still closer than the tier's own
   minimum gap (a coin flip can, by chance, still leave one too many close
   together, and that reads as a real visual overlap at a lower tier's
   larger circle size). Hard only thins the song's *repetitive* sections
   (a verse/chorus that recurs), lightly; Normal and Easy thin everywhere,
   progressively more. Also derives Insane itself (no thinning, just
   Difficulty-setting clamping) from the Styled output, so all four tiers
   go through the same clamping logic. SliderMultiplier is never touched
   at any tier — a real hand-mapped Easy's long, slow-*reading* sliders
   (see `example/keha_backstabber/`) actually use a *higher* slider
   velocity (more on-screen distance per beat) than Insane's, not a lower
   one.

`beatmap_utils.py` holds the shared `.osu` parsing/writing code and data
structures used by every stage.

## Usage

### GUI

`gui.py` wraps `main.py` in a desktop window — a file picker for the song,
a field or checkbox for every argument below, and a Generate button:

```bash
pip install -r requirements.txt
python3 gui.py
```

It runs the same pipeline `main.py` does (there's no separate logic to
keep in sync), streams the pipeline's own console output into a log box
so you can watch each stage run, prints a statistics report on the
finished Insane difficulty once generation completes (see
`beatmap_stats.py` below), and — if "Open the finished map when done" is
checked — opens the resulting `.osz` (or, without `--osz`, the Insane
`.osu`) with whatever your OS has registered for that file type.

The Difficulties section lets you uncheck any of Easy/Normal/Hard/Insane
— every tier is still generated internally (an easier tier is always
derived from the one above it, so there's no way to skip one mid-spread),
an unchecked one is just deleted from the result afterward, including
being stripped back out of an already-built `.osz`.

Built with Tkinter, which ships with the Python standard library on
Windows and macOS installers; on Linux it's usually a separate distro
package (`sudo apt install python3-tk` on Debian/Ubuntu, `sudo dnf
install python3-tkinter` on Fedora).

### Command line

The easiest way to run the whole pipeline from a shell is `main.py`:

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

`main.py` also forwards `--spacing`, `--curviness`, `--stream-frequency`,
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
  curviness can drift from the `--curviness` baseline, how strongly the
  path wanders around the playfield, and how many times `--spacing`
  itself shifts over the course of the song, all together — low is tight
  and predictable, high is loose and varied. At `0`, spacing never
  changes at all (one constant multiplier the whole way through); higher
  values allow up to 3 shifts, each large enough to actually notice
  (roughly ±15-25%) without being jarring, and never more than a
  handful of changes in one song regardless. Passing `--angle-jitter`
  explicitly overrides temperature's value for that one knob only.
- `--angle-jitter DEGREES` (default: derived from `--temperature`, roughly
  `1`-`10`) — how much extra random wiggle gets added on top of each
  repeating motif's turn angle, for circles and slider curves alike.
  Turning this up gives more varied flow/angles without changing when
  anything is hit; turning it down makes the motif patterns read more
  rigidly.
- `--stream-frequency F` (default `0.5`) — how often a fast (quarter-beat
  -or-closer) run of notes becomes a deliberate *stream* — stacked in one
  spot, or spread along a locked-in straight line — rather than just
  following the ordinary motif-driven flow any other note would, one at a
  time with no forced overlap or fixed direction (`0` = never a stream,
  `1` = always one). Any run longer than 3 is split into consecutive
  bursts of at most 3 regardless of this setting — each its own
  independent unit with its own mode and its own entry/exit gap — so a
  chain of, say, 8 eighth-notes never reads as one long pile, one long
  line, or (at `0`) one undifferentiated wall of 8 individually-flowing
  notes either. Whichever bursts do become a stream still split roughly
  50/50 between stack and line (weighted by whether the burst is short
  enough to stack at all — see below); that mix isn't separately
  exposed, since "how often do I get a stream at all" was the genuinely
  useful knob here. A burst is only eligible for "stack" if its whole
  span (first member to last) is half a beat or less — piling more than
  that much elapsed time onto one spot is a real overlap the ranking
  criteria's Hard-difficulty rule forbids ("objects 1/2 of a beat apart or
  less must not fully overlap"); a burst failing that check is "line"
  instead whenever it streams. The single gap entering a burst, and the
  single gap leaving one (including between two consecutive bursts of the
  same long run, streaming or not), is also widened a bit past ordinary
  distance snap, so a burst reads as a clearly set-apart unit instead of
  bleeding into the normal flow — or the next burst — on either side.
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

### Beatmap statistics

`beatmap_stats.py` computes distribution statistics (not just single
numbers — full min/p25/median/p75/max/mean/stdev plus an ASCII histogram)
for any `.osu` file: delay between consecutive objects, on-screen jump
spacing, slider length and duration, turn-angle at each object, the
straight/curved/chain slider mix, and what fraction of consecutive pairs
are stacked:

```bash
python3 beatmap_stats.py "out/Song [Insane].osu"
```

Pass several paths to compare difficulties (or a real hand-mapped
beatmap) side by side. The GUI runs this automatically against the
Insane difficulty after every generation and prints the result in its
log box.

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
- **Difficulty settings per tier** — `make_easy.py`'s `TIER_TARGET` sets
  AR/OD/HP/CS to explicit per-tier values (taken from the same
  `example/keha_backstabber/` reference set, so Insane genuinely reads as
  the hardest difficulty — fast approach rate, tight timing window — and
  each tier down is a deliberate step easier rather than every tier
  clamping the same flat defaults into the middle of its own range),
  clamped into each tier's own document range (`TIER_SETTINGS`) as a
  safety net, not a relative shift from Insane's own settings.
  SliderMultiplier is deliberately *not*
  clamped, even though the document's Easy/Normal guideline says to avoid
  slider velocity above 1.3 — a real reference Easy difficulty
  (`example/keha_backstabber/`) uses a SliderMultiplier of 3.54, leaning
  on long, slow-*reading* sliders rather than a cramped multiplier; that
  concrete example took priority over the document's guideline here.
- **Drain time spread** — deletion never touches the first or last object,
  so `make_easy.py`'s assertion that each tier's first/last object time
  exactly matches Insane's should always hold trivially.
- **Objects never off-screen, snapping, timing overlaps** — enforced
  throughout (playfield margin, `slider_length_for_gap`'s rounded-gap
  derivation, `rounded_gap_ms`); see the pipeline stage docstrings. A
  deterministic final pass in `make_easy.py` (`enforce_min_gap`) also
  clears any remaining pair closer than a tier's own minimum gap, which
  otherwise reads as a real visual overlap at that tier's larger circle
  size even though the two objects' declared positions don't literally
  coincide.
- **Combo colours / hitsounds** — `beatmap_utils.default_metadata` sets
  three custom combo colours and every hittable edge gets an audible
  hitsound (never silent); `make_easy.py`'s `regularize_hitsounds` only
  ever *adds* a downbeat accent in a thinned measure, never removes an
  existing one, so a tier that thins broadly (Normal/Easy) can't end up
  with long silent-feeling stretches. Downbeats are also never deleted,
  which is what guarantees regularize_hitsounds always has something to
  accent every measure in the first place.
- **Known gap**: the document's Easy/Normal-tier note-density guideline
  ("mostly 1/1, 2/1, or slower") is closer than before but not fully met
  — `make_easy.py` deletes whole objects only, and a downbeat is never
  deleted, so a genuine downbeat-to-downbeat gap shorter than a full beat
  at a fast tempo can survive even in Easy (verified rare: 3 of ~260
  objects on the reference song). Spinner rules, skinning rules, and
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
