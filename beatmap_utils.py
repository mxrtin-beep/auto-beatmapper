"""
Shared data structures and .osu file I/O helpers used by all three pipeline
stages (generate_base_beatmap.py, add_variety.py, apply_style.py).

Keeping this parsing/writing logic in one place means every stage reads and
writes *exactly* the same dialect of the .osu format, so a map can be piped
stage1 -> stage2 -> stage3 without any of them guessing at the others' output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --- osu! playfield & hit object type constants -----------------------------

PLAYFIELD_W = 512
PLAYFIELD_H = 384

HIT_CIRCLE = 1
HIT_SLIDER = 2
NEW_COMBO = 4
HIT_SPINNER = 8


# --- Timing -------------------------------------------------------------

@dataclass
class TimingPoint:
    """One line of the [TimingPoints] section.

    Only "uninherited" (red) timing points are produced by this pipeline —
    the maps generated here use a constant BPM, so a single red line at the
    map's offset is all that's needed.
    """

    time: float  # ms
    beat_length: float  # ms per beat
    meter: int = 4
    sample_set: int = 2  # Soft
    sample_index: int = 0
    volume: int = 80
    uninherited: bool = True
    effects: int = 0

    def to_line(self) -> str:
        return (
            f"{self.time:.0f},{self.beat_length:.12g},{self.meter},"
            f"{self.sample_set},{self.sample_index},{self.volume},"
            f"{1 if self.uninherited else 0},{self.effects}"
        )

    @staticmethod
    def from_line(line: str) -> "TimingPoint":
        parts = line.split(",")
        time, beat_length, meter, sample_set, sample_index, volume, uninherited = parts[:7]
        effects = parts[7] if len(parts) > 7 else "0"
        return TimingPoint(
            time=float(time),
            beat_length=float(beat_length),
            meter=int(meter),
            sample_set=int(sample_set),
            sample_index=int(sample_index),
            volume=int(volume),
            uninherited=bool(int(uninherited)),
            effects=int(effects),
        )


# --- Hit objects ----------------------------------------------------------

@dataclass
class HitObject:
    x: int
    y: int
    time: float  # ms
    is_new_combo: bool = False
    hitsound: int = 0

    # slider-only fields
    is_slider: bool = False
    curve_type: str = "L"  # L(inear) | P(erfect circle) | B(ezier)
    points: List[Tuple[int, int]] = field(default_factory=list)  # anchors after the head
    slides: int = 1  # 1 = no repeats
    length: float = 0.0  # pixel length of a single slide
    edge_hitsounds: Optional[List[int]] = None
    edge_samplesets: Optional[List[str]] = None

    def object_type(self) -> int:
        t = HIT_SLIDER if self.is_slider else HIT_CIRCLE
        if self.is_new_combo:
            t |= NEW_COMBO
        return t

    def duration_ms(self, beat_length_ms: float, slider_multiplier: float) -> float:
        """How long this object occupies the timeline (0 for circles)."""
        if not self.is_slider:
            return 0.0
        px_per_beat = slider_multiplier * 100.0
        ms_per_slide = (self.length / px_per_beat) * beat_length_ms
        return ms_per_slide * self.slides

    def end_time(self, beat_length_ms: float, slider_multiplier: float) -> float:
        return self.time + self.duration_ms(beat_length_ms, slider_multiplier)

    def to_line(self) -> str:
        obj_type = self.object_type()
        if not self.is_slider:
            return f"{self.x},{self.y},{self.time:.0f},{obj_type},{self.hitsound},0:0:0:0:"

        curve = self.curve_type + "|" + "|".join(f"{px}:{py}" for px, py in self.points)
        edge_h = self.edge_hitsounds or [0] * (self.slides + 1)
        edge_s = self.edge_samplesets or ["0:0"] * (self.slides + 1)
        return (
            f"{self.x},{self.y},{self.time:.0f},{obj_type},{self.hitsound},"
            f"{curve},{self.slides},{self.length:.14g},"
            f"{'|'.join(str(h) for h in edge_h)},{'|'.join(edge_s)},0:0:0:0:"
        )

    @staticmethod
    def from_line(line: str) -> "HitObject":
        parts = line.split(",")
        x, y, time, obj_type, hitsound = (int(parts[0]), int(parts[1]), float(parts[2]),
                                           int(parts[3]), int(parts[4]))
        is_new_combo = bool(obj_type & NEW_COMBO)
        is_slider = bool(obj_type & HIT_SLIDER)

        if not is_slider:
            return HitObject(x=x, y=y, time=time, is_new_combo=is_new_combo, hitsound=hitsound)

        curve_field = parts[5]
        curve_type, *point_strs = curve_field.split("|")
        points = []
        for p in point_strs:
            px, py = p.split(":")
            points.append((int(px), int(py)))
        slides = int(parts[6])
        length = float(parts[7])
        edge_hitsounds = [int(v) for v in parts[8].split("|")] if len(parts) > 8 and parts[8] else None
        edge_samplesets = parts[9].split("|") if len(parts) > 9 and parts[9] else None

        return HitObject(
            x=x, y=y, time=time, is_new_combo=is_new_combo, hitsound=hitsound,
            is_slider=True, curve_type=curve_type, points=points, slides=slides,
            length=length, edge_hitsounds=edge_hitsounds, edge_samplesets=edge_samplesets,
        )


# --- Whole-file container --------------------------------------------------

@dataclass
class Beatmap:
    general: Dict[str, str] = field(default_factory=dict)
    editor: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
    difficulty: Dict[str, str] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)
    timing_points: List[TimingPoint] = field(default_factory=list)
    colours: List[str] = field(default_factory=list)
    hit_objects: List[HitObject] = field(default_factory=list)

    # --- convenience accessors -------------------------------------------------
    @property
    def beat_length(self) -> float:
        """ms per beat, from the first (uninherited) timing point."""
        for tp in self.timing_points:
            if tp.uninherited:
                return tp.beat_length
        raise ValueError("Beatmap has no uninherited timing point")

    @property
    def bpm(self) -> float:
        return 60000.0 / self.beat_length

    @property
    def offset(self) -> float:
        for tp in self.timing_points:
            if tp.uninherited:
                return tp.time
        raise ValueError("Beatmap has no uninherited timing point")

    @property
    def slider_multiplier(self) -> float:
        return float(self.difficulty.get("SliderMultiplier", 1.4))


def default_metadata(title: str, artist: str, creator: str, version: str, audio_filename: str) -> Beatmap:
    """Build a Beatmap with sane defaults for every non-hit-object section."""
    bm = Beatmap()
    bm.general = {
        "AudioFilename": audio_filename,
        "AudioLeadIn": "0",
        "PreviewTime": "-1",
        "Countdown": "0",
        "SampleSet": "Soft",
        "StackLeniency": "0.7",
        "Mode": "0",
        "LetterboxInBreaks": "0",
        "WidescreenStoryboard": "0",
    }
    bm.editor = {
        "DistanceSpacing": "1",
        "BeatDivisor": "4",
        "GridSize": "32",
        "TimelineZoom": "2",
    }
    bm.metadata = {
        "Title": title,
        "TitleUnicode": title,
        "Artist": artist,
        "ArtistUnicode": artist,
        "Creator": creator,
        "Version": version,
        "Source": "",
        "Tags": "auto-generated auto-beatmapper",
        "BeatmapID": "0",
        "BeatmapSetID": "-1",
    }
    bm.difficulty = {
        "HPDrainRate": "5",
        "CircleSize": "4",
        "OverallDifficulty": "6",
        "ApproachRate": "8",
        "SliderMultiplier": "1.4",
        "SliderTickRate": "1",
    }
    bm.events = [
        "//Background and Video events",
        "//Break Periods",
        "//Storyboard Layer 0 (Background)",
        "//Storyboard Layer 1 (Fail)",
        "//Storyboard Layer 2 (Pass)",
        "//Storyboard Layer 3 (Foreground)",
        "//Storyboard Layer 4 (Overlay)",
        "//Storyboard Sound Samples",
    ]
    bm.colours = [
        "Combo1 : 255,128,64",
        "Combo2 : 128,192,255",
        "Combo3 : 255,220,80",
    ]
    return bm


def _write_kv_section(lines: List[str], name: str, kv: Dict[str, str]) -> None:
    lines.append(f"[{name}]")
    for k, v in kv.items():
        lines.append(f"{k}:{v}")
    lines.append("")


def write_osu(bm: Beatmap, path: str) -> None:
    lines: List[str] = ["osu file format v14", ""]
    _write_kv_section(lines, "General", bm.general)
    _write_kv_section(lines, "Editor", bm.editor)
    _write_kv_section(lines, "Metadata", bm.metadata)
    _write_kv_section(lines, "Difficulty", bm.difficulty)

    lines.append("[Events]")
    lines.extend(bm.events)
    lines.append("")

    lines.append("[TimingPoints]")
    for tp in sorted(bm.timing_points, key=lambda t: t.time):
        lines.append(tp.to_line())
    lines.append("")

    lines.append("[Colours]")
    lines.extend(bm.colours)
    lines.append("")

    lines.append("[HitObjects]")
    for ho in sorted(bm.hit_objects, key=lambda h: h.time):
        lines.append(ho.to_line())
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def read_osu(path: str) -> Beatmap:
    bm = Beatmap()
    section = None
    with open(path, "r", encoding="utf-8-sig") as f:
        raw_lines = [l.rstrip("\n") for l in f]

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            continue
        if section in ("General", "Editor", "Metadata", "Difficulty"):
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            target = {
                "General": bm.general, "Editor": bm.editor,
                "Metadata": bm.metadata, "Difficulty": bm.difficulty,
            }[section]
            target[key.strip()] = value.strip()
        elif section == "Events":
            bm.events.append(line)
        elif section == "TimingPoints":
            bm.timing_points.append(TimingPoint.from_line(stripped))
        elif section == "Colours":
            bm.colours.append(line)
        elif section == "HitObjects":
            bm.hit_objects.append(HitObject.from_line(stripped))

    return bm


# --- geometry helpers shared by later stages --------------------------------

def clamp_to_playfield(x: float, y: float, margin: int = 20) -> Tuple[int, int]:
    x = max(margin, min(PLAYFIELD_W - margin, x))
    y = max(margin, min(PLAYFIELD_H - margin, y))
    return int(round(x)), int(round(y))


def fix_time_overlaps(objects: List[HitObject], beat_length_ms: float, slider_multiplier: float,
                       min_gap_ms: float = 1.0) -> int:
    """Guarantee no slider's *on-disk* end time reaches into the next object's start.

    `HitObject.time` is written to the .osu file rounded to a whole
    millisecond (`to_line` uses `:.0f`), but a slider's rendered duration is
    reconstructed by osu! from its unrounded `length` — so two objects whose
    gap was computed to be exactly one subdivision apart can, after each
    object's start time is independently rounded, end up with the slider's
    rounded end landing at or past the next object's rounded start: a
    same-time-slot overlap that osu! (and any beatmap checker) flags as
    illegal, even though nothing here ever intended it. This shrinks the
    offending slider's `length` just enough to leave `min_gap_ms` of
    clearance — never moves any object's `time`, which would instead
    un-snap it from the beat grid. Mutates `objects` (sorted by time) in
    place; returns how many sliders were shrunk.
    """
    px_per_beat = slider_multiplier * 100.0
    fixed = 0
    for a, b in zip(objects, objects[1:]):
        if not a.is_slider:
            continue
        a_start = round(a.time)
        b_start = round(b.time)
        a_end = a_start + a.duration_ms(beat_length_ms, slider_multiplier)
        if a_end > b_start - min_gap_ms:
            allowed_duration = max(1.0, (b_start - a_start) - min_gap_ms)
            allowed_one_slide_ms = allowed_duration / max(1, a.slides)
            a.length = max(1.0, px_per_beat * (allowed_one_slide_ms / beat_length_ms))
            fixed += 1
    return fixed
