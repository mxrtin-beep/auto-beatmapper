# auto-beatmapper

Automatically generates a full [osu!](https://osu.ppy.sh/) beatmap set
(`.osu`) from an MP3, adapting *how dense* the map gets directly to the
song's own local loudness — quiet sections stay sparse, loud sections get
faster and busier — rather than a fixed rate reshaped after the fact.
Three stages, each producing its own `.osu` file so you can inspect (or
play) the map at every step.

## Pipeline

1. **`generate_base_beatmap_v2.py`** — analyzes the song to estimate its
   BPM and offset (the time of the first beat) — every integer BPM near a
   rough broadband guess is scored, at every possible phase, against how
   well a beat grid lines up with a low-frequency ("kick drum") onset
   envelope, giving a reliable BPM and a *close* offset; that offset is
   then refined the way a mapper sets it by eye in the editor's waveform
   view — a kick's amplitude rises fast and decays slowly, so the marker
   goes right at the peak — by finding the actual local amplitude peak of
   the raw (low-passed, same kick band) waveform near each grid beat and
   nudging the offset by the median correction. Pass `--bpm`/`--offset`
   (ms) to set either manually instead of auto-detecting it — any value
   works for `--offset`, since it's wrapped to the equivalent position
   within one beat (e.g. `-118` and `334` at 137 BPM name the same beat).

   Circles are then placed at a rate that tracks the song's own loudness,
   classified once per whole beat (never flickering between rates
   mid-beat, so a busy beat is busy *all the way through*):

   - Silent/very quiet -> no circles at all (nothing to click to).
   - Quiet             -> one circle per whole beat.
   - Normal            -> one circle per half beat.
   - Intense           -> one circle per quarter beat (a 16th note).
   - Climax            -> one circle per eighth beat (a 32nd note) — the
     densest tier there is, one subdivision finer than "intense".

   A single `--intensity` knob (0-1) shifts how much of the song reaches
   each rate; every rate stays reachable at any setting, just how *much*
   of the song lands in each one shifts. Climax isn't just a louder
   version of the next tier down — a whole run of consecutive loud beats
   becomes one burst anchored to its own measure's downbeat (not an
   isolated beat that can start on a musically weak position mid-measure),
   capped to one measure's span by default and gradually up to two
   measures at the highest `--intensity` settings, with at least one
   measure of cooldown (ordinary placement, not silence) forced before a
   fresh burst can start — so a long loud stretch reads as several
   distinct bursts, never one unbroken wall. Sparingly, in the louder half
   of the intensity range, an ordinary gap between two circles can also
   get a short embellishment — a 5- or 7-note chain at quarter-beat (16th
   note) spacing — fit strictly inside room that's already there, never
   crowding what's on either side. A fast (quarter-beat-or-closer) run
   also never sprawls past a full beat, and leaves a full two-quarter-beat
   gap before the next one picks back up, so back-to-back busy beats read
   as two distinct bursts rather than one continuous run shifted by a
   single slot.

2. **`add_sliders_v2.py`** — merges some adjacent circles into sliders
   (`--chain-probability` how often at all, `--slider-length-bias` how
   long when it does), including occasional repeating/"bounce" sliders
   (one path traveled back and forth) for a run that's already evenly
   spaced. A fast run in this pathway is never incidental — it's always a
   deliberate climax burst or embellishment chain — so it's always treated
   as one gesture: piled onto the exact same spot (0px apart), and the
   object immediately following it lands on that same spot too, once,
   before normal flow resumes.

   Positioning — distance snap (on-screen distance for a gap exactly
   matches what a slider spanning that time would travel), playfield
   bounds, a fast run reading as a deliberate stack or overlapping line
   rather than a zigzag blob, a small set of repeating turn-angle patterns
   keyed to each measure's own energy level, and a slow "wander" target
   that keeps the whole path migrating around the playfield instead of
   orbiting one local spot — is handled by re-running `apply_style.py`
   against the merged result.

   Hitsounds are assigned once per whole beat from local loudness and
   downbeat position (see below), and every difficulty tier — Insane plus
   whichever of Hard/Normal/Easy you ask for — gets its own real
   positioning pass: each tier is independently thinned (deleting objects
   only, never merging or reshaping — a lower tier has less business
   keeping a fast subdivision, thinned first and most often, then the
   slower ones progressively less) and then given its own `apply_style.py`
   run, with its own scaled-down jump distance (a lower tier's bigger
   circles need to travel less far to still feel comfortable), rather than
   a rescaled copy of Insane's own already-styled positions. Difficulty
   settings (HP/CS/OD/AR) are set per tier from the same
   `example/keha_backstabber/` reference set used throughout, so Insane
   genuinely reads as the hardest difficulty and each tier down is a
   deliberate step easier.

`beatmap_utils.py` holds the shared `.osu` parsing/writing code and data
structures used by every stage.

### Hitsounds

`assign_hitsounds` (in `add_variety.py`, shared code) decides one
hitsound per whole beat, not per object — checking a real community
beatmap set (`example/keha_backstabber/`) found a hitsound change always
lines up with a beat boundary, essentially never switching between two
objects that share one. A bigger accent (clap/finish) lines up with a
loud and/or downbeat moment; a long quiet stretch still gets an
occasional forced whistle often enough that it never reads as "no
hitsounds" to a checker.

The same reference set also showed a section's second or third pass — a
verse or chorus repeating — mostly reusing its *first* pass's exact
accent pattern, not just landing in the same coarse loudness bracket
independently each time (and often the same circle/slider layout too).
`find_repeating_measure_map` detects that kind of repeat (the same
windowed sequence of measure-loudness "buckets" recurring elsewhere in
the song) and, when a beat's measure repeats an earlier one, copies that
earlier measure's own corresponding beat's hitsound instead of
re-deriving it independently from that pass's own, merely similar energy.

`add_sliders_v2.py`'s own circle-vs-slider layout decisions reuse the
same repeat map, best-effort: a repeated measure's chain lengths and
plain/bounce choices copy an earlier occurrence's, falling back to a
fresh independent decision wherever this occurrence's own actual data
doesn't fit the replayed one (a chain crossing a measure boundary can
leave a repeat's own circles slightly out of alignment with the
original, for instance — the fallback just means that one decision rolls
fresh instead of forcing a bad fit). Pass `--no-reuse-layout` (or uncheck
"Reuse a repeated section's own layout" in the GUI) to revert to the
original, fully independent-per-run behavior.

`apply_style.py`'s own turn-angle motif selection (see Pipeline, above)
uses the same repeat map too: a measure that genuinely repeats an earlier
one (the stricter windowed-shingle match, not just this one measure's
own energy bucket coincidentally landing the same) plays its motif from
that earlier occurrence's own measure index — so the actual on-screen
*arrangement* (turn angles, not just which objects are circles vs.
sliders) reads as the same repeated shape too, not just similar.

## Usage

### GUI

`gui_v2.py` is a desktop window — a file picker for the song, a field or
dial for every knob below, and a Generate button:

```bash
pip install -r requirements.txt
python3 gui_v2.py
```

It streams the pipeline's own console output into a log box so you can
watch each stage run, and — if "Open the finished map when done" is
checked — opens the resulting `.osz` (or, without `--osz`, the Insane
`.osu`) with whatever your OS has registered for that file type.

Every knob — Intensity, Slider vs. circle mix, Slider length, Slider
curviness, Jump distance — is shown as a plain 0-1 dial that defaults to
the middle, but the middle doesn't have to mean the literal middle of the
underlying range: every dial is tuned so 0.5 already gives a
good-sounding result, then scales further toward either end from there.
A Difficulties section lets you check any of Hard/Normal/Easy alongside
the always-generated Insane.

Built with Tkinter, which ships with the Python standard library on
Windows and macOS installers; on Linux it's usually a separate distro
package (`sudo apt install python3-tk` on Debian/Ubuntu, `sudo dnf
install python3-tkinter` on Fedora).

### Command line

```bash
pip install -r requirements.txt

python3 generate_base_beatmap_v2.py song.mp3 \
    --output "out/Song (Circles).osu" --intensity 0.65 \
    --title "Song Title" --artist "Artist Name" --creator "Your Name"

python3 add_sliders_v2.py "out/Song (Circles).osu" song.mp3 \
    --output "out/Song [Insane].osu" \
    --hard-output "out/Song [Hard].osu" \
    --normal-output "out/Song [Normal].osu" \
    --easy-output "out/Song [Easy].osu" \
    --chain-probability 0.3 --slider-length-bias 0.4 --curviness 0.5 --spacing 1.9
```

`add_sliders_v2.py` also forwards `--curviness`, `--spacing`, and `--seed`
straight through to `apply_style.py` (see below) for each tier it
generates, at that tier's own scaled-down spacing.

To turn the resulting `.osu` file(s) into a `.osz` yourself: put them and
the source MP3 (renamed to match `AudioFilename` in the map) in the same
folder, zip it, and rename the zip's extension to `.osz` — or use
`build_osz.py`:

```bash
python3 build_osz.py song.mp3 "out/Song [Insane].osu" "out/Song [Hard].osu" \
    "out/Song [Normal].osu" "out/Song [Easy].osu" --output "out/Song.osz"
```

### Re-styling without changing the rhythm

`apply_style.py` never touches timing, object type, or object count — only
where things are placed. To get a different flow/angle pattern (a fresh
mix of stacks vs. overlapping lines, different slider curves, etc.)
without regenerating the beatmap's rhythm at all, re-run just that stage
with a new seed:

```bash
python3 apply_style.py "out/Song (Circles).osu" \
    --output "out/Song (Restyled).osu" --audio song.mp3 --seed 123
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
- `--stream-frequency F` (default `0.5`, but `add_sliders_v2.py` always
  forwards `1.0` — a fast run in this pathway is never incidental) — how
  often a fast (quarter-beat-or-closer) run of notes becomes a deliberate
  *stream* at all — stacked in one spot, or spread along a locked-in
  straight line — rather than just following the ordinary motif-driven
  flow any other note would, one at a time with no forced overlap or
  fixed direction (`0` = never a stream, `1` = always one). This only
  controls *whether* a run streams, not what it looks like when it does —
  see `--stack-probability` for that.
- `--stack-probability P` (default `1.0`) — of whichever bursts
  `--stream-frequency` already decided *are* a stream, the mix between
  piling into one stacked spot and spreading along a line (`0` = always
  line, `1` = always stack, whenever the burst is short enough to stack at
  all — see below). Has no say over whether a burst streams in the first
  place; that's `--stream-frequency`'s job alone. A burst is only eligible
  for "stack" if its whole span (first member to last) is half a beat or
  less — piling more than that much elapsed time onto one spot is a real
  overlap the ranking criteria's Hard-difficulty rule forbids ("objects
  1/2 of a beat apart or less must not fully overlap"); a burst failing
  that check is "line" instead whenever it streams. The single gap
  entering a burst, and the single gap leaving one, is also widened a bit
  past ordinary distance snap, so a burst reads as a clearly set-apart
  unit instead of bleeding into the normal flow — or the next burst — on
  either side; the object immediately following a stack also lands on
  that stack's own exact spot, once, before normal flow resumes.
- `--curviness C` (default `0.5`) — how curvy the map feels, `0`-`1`. `0`
  makes almost every slider a straight line; `1` makes almost every
  slider a pronounced curve, and makes the bow of every curved slider
  more pronounced too. This is a *theme*, not a flat per-slider coin
  flip: each measure's energy bucket gets its own curviness level offset
  from `--curviness` (seeded, so it's consistent across a run but varies
  section to section) — a section keeps reading as consistently curvy or
  consistently straight-and-bendy, the same way a repeating chorus reuses
  the same motif. Straight-vs-curved is also locked once per combo — the
  first slider in a combo rolls it, every later slider in that same combo
  just inherits the choice, so a combo never mixes a straight slider with
  a curved one back to back.

### Randomness / seeds

Both stages make a lot of small randomized choices (which eligible runs
become sliders, how the flow angle jitters) so running the pipeline twice
on the same song gives you a different-feeling map each time. Every run
prints the seed it used (`Using seed: 123456789`); pass `--seed
123456789` back in to reproduce that exact map again.

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
beatmap) side by side. The GUI's statistics-report checkbox runs
`beatmap_report.py` for a nicer PDF version of the same comparison.

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
  each tier down is a deliberate step easier), clamped into each tier's
  own document range (`TIER_SETTINGS`) as a safety net.
- **Objects never off-screen, snapping, timing overlaps** — enforced
  throughout (playfield margin, `slider_length_for_gap`'s rounded-gap
  derivation, a slider curve's actual rendered arc — not just its control
  points — checked against the playfield before it's ever used); see the
  pipeline stage docstrings. `add_sliders_v2.py`'s own merge step also
  rejects any circle-to-slider merge whose total span (or, for a bounce
  slider, any individual repeat leg) would fall under 125ms, matching the
  ranking criteria's minimum slider duration — checked once, at the one
  shared merge every tier's own thinning derives from, rather than only
  showing up as an error on whichever tier happened to still be carrying
  a too-short one.
- **Combo colours / hitsounds** — `beatmap_utils.default_metadata` sets
  three custom combo colours and every hittable edge gets an audible
  hitsound (never silent) via `assign_hitsounds` (see above), with a
  forced occasional whistle so a long quiet stretch never reads as "no
  hitsounds" to a checker.
- Spinner rules, skinning rules, and BPM-scaling nuances aren't
  addressed — this tool doesn't generate spinners or skin elements at all.

## Example output

`example/keha_backstabber/` holds a real community beatmap set (audio and
images stripped out) used as the format reference while building this
tool, and as the source of the analysis behind the hitsound/layout
repetition behavior described above.
