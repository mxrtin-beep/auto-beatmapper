# auto-beatmapper

Automatically generates a full [osu!](https://osu.ppy.sh/) beatmap set
(`.osz`) from an MP3 — pick a song, turn a few dials, get a map.

## Running it

The easiest way to run it is the GUI:

```bash
pip install -r requirements.txt
python3 gui_v2.py
```

Or, if you don't want to set up Python at all, grab the packaged build
from this repo's [Releases](../../releases) page and run that instead.

Either way you get a file picker for the song, a dial for every
parameter below, and a Generate button that streams the pipeline's
progress into a log box and (optionally) opens the finished map when
it's done.

## How a map gets made

Under the hood, generating a map is three steps, each building on the
last:

1. **Lay down circles.** The song's BPM and beat offset are detected
   automatically, then a circle is placed on a beat subdivision chosen by
   how loud/busy that moment of the song is — quiet parts get sparse,
   loud parts get dense bursts — so the map's rhythm tracks the track's
   own energy instead of a fixed pattern.
2. **Merge some circles into sliders.** Adjacent circles are combined
   into sliders (including the occasional back-and-forth "bounce"
   slider) to break up the rhythm and add some longer holds, rather than
   leaving everything as single taps.
3. **Style it.** Every object then gets moved around the playfield —
   jump distances, curve shapes, turn angles, streams vs. stacks — by a
   set of positioning rules, so the map actually flows and reads as
   intentional rather than a random scatter. This last step can also be
   re-run by itself to get a different look without touching the
   rhythm at all.

A lower difficulty (Hard/Normal/Easy) is generated the same way as
Insane, just thinned down and re-styled with gentler spacing, rather
than a rescaled copy of Insane.

## What you can configure

- **Intensity** — how much of the song reaches the busier circle
  densities.
- **Slider vs. circle mix** — how often adjacent circles get merged
  into sliders.
- **Slider length** — how long those sliders tend to be.
- **Slider curviness** — how straight vs. curved sliders are drawn.
- **Jump distance / spacing** — how far apart objects are placed on
  screen.
- **Creativity ("temperature")** — how tightly the styling sticks to a
  predictable pattern vs. how much it varies section to section.
- **Stream frequency / stack probability** — how often a fast run of
  notes becomes a deliberate stream, and whether that stream piles onto
  one spot or spreads along a line.
- **BPM / offset** — override the auto-detected beat grid manually.
- **Difficulties to generate** — Insane is always made; Hard/Normal/Easy
  are optional add-ons.
- **Seed** — see below.

## The statistics report

Checking the statistics-report option produces a PDF comparing the
generated difficulties (or a hand-mapped beatmap you point it at) side
by side — distributions, not just single numbers, for things like the
delay between objects, on-screen jump spacing, slider length/duration,
turn angles, and the straight/curved/chain slider mix. It also includes
a judgment page per difficulty, checking the map against osu!'s ranking
criteria (AR/OD/HP/CS ranges, off-screen objects, illegal overlaps,
slider velocity, stream length, and more) with a pass/warn/fail verdict
for each. It's a quick way to sanity-check that a map "feels" like a
real one and is close to rankable, rather than eyeballing it in the
editor.

You can also run it directly on an existing `.osu` or `.osz`:

```bash
python3 beatmap_report.py "MyMapset.osz"
```

which writes `output/MyMapset/report.pdf` by default (pass `--output`
to write somewhere else).

## The seed

Generating is randomized — which runs become sliders, how much the flow
jitters, and so on — so running the pipeline twice on the same song
normally gives two different-feeling maps. Every run prints the seed it
used; passing that same seed back in reproduces the exact same map again,
which is handy for re-generating a map you liked or for isolating the
effect of changing just one other parameter.
