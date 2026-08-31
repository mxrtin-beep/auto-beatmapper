"""
Uniformity — how strongly a repeated song section (a verse's second
repeat, a chorus that comes back after a bridge) reuses the same
circle/slider, hitsound, and layout choices as its earlier occurrence(s),
instead of every note deciding independently.

`--uniformity` is 0-1:
  0   -> nothing here kicks in; every structural choice (slider-vs-circle,
         stream/stack mode, motif/curviness) is decided independently, the
         same way the pipeline already behaves without this parameter.
  1   -> the whole song is treated as a single section: every measure
         reuses the same choices as every other measure, at the same
         position within its own measure.
  in between -> the song is split into measure-groups, merged together
         whenever they sound alike (matched by a chroma self-similarity
         pass over the mix — the same kind of signal that identifies a
         returning verse or chorus, not just a fixed measure count), and
         each occurrence in a group reuses its group's pattern with
         probability `uniformity` at each individual decision point,
         otherwise decides independently. That's deliberate: a chorus's
         second repeat doesn't copy the first one 100% exactly, and a
         mid-value like 0.5 doesn't reset precisely every N measures --
         it's a tendency toward matching, not a hard rule.

Call sites don't need to know any of the above: `compute_pattern_groups`
gives a {measure_index: group_id} map (or None, when uniformity is 0 and
there's nothing to do), and `blended_choice` is a drop-in wrapper around a
structural decision that already exists as `indep_fn` -- it stays
byte-for-byte identical to the caller's own independent decision whenever
uniformity is 0 or no group applies.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Hashable, Optional, TypeVar

import numpy as np

T = TypeVar("T")


def compute_pattern_groups(audio_path: str, offset_ms: float, measure_length_ms: float,
                            track_end_ms: float, uniformity: float) -> Optional[Dict[int, int]]:
    """Return {measure_index: pattern_group_id}, or None when uniformity is
    0 (patterning disabled -- callers should fall back to their existing
    independent-randomness behavior entirely).

    Every measure starts out in a small local block -- a handful of
    consecutive measures, sized from `uniformity` (higher uniformity means
    longer blocks) -- and blocks are then merged together, however far
    apart they sit in the song, whenever their averaged chroma content is
    similar enough. That merge step is what lets a chorus reuse an earlier
    chorus's pattern instead of only ever matching its own immediate
    neighbors. The similarity threshold relaxes as uniformity rises, so
    near 1.0 almost every block ends up merged into one.
    """
    uniformity = max(0.0, min(1.0, uniformity))
    if uniformity <= 0.0:
        return None

    num_measures = max(1, int((track_end_ms - offset_ms) / measure_length_ms) + 1)
    if num_measures <= 1 or uniformity >= 0.999:
        # The whole song is short enough, or uniformity asks for it
        # outright, to just be one section -- no need for the audio
        # analysis below to arrive at the same answer.
        return {m: 0 for m in range(num_measures)}

    import librosa

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    frame_times_ms = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length) * 1000.0

    # One averaged, L2-normalized chroma vector per measure -- normalized
    # so the similarity comparison below is about harmonic content, not
    # loudness.
    measure_vecs = []
    for m in range(num_measures):
        start = offset_ms + m * measure_length_ms
        end = start + measure_length_ms
        mask = (frame_times_ms >= start) & (frame_times_ms < end)
        if not mask.any():
            measure_vecs.append(measure_vecs[-1] if measure_vecs else np.zeros(chroma.shape[0]))
            continue
        vec = chroma[:, mask].mean(axis=1)
        norm = np.linalg.norm(vec)
        measure_vecs.append(vec / norm if norm > 1e-9 else vec)

    # Local blocks: a new one starts every `block_size` measures. Higher
    # uniformity means longer blocks, so even a moderate setting already
    # groups a handful of consecutive measures before any cross-song
    # matching happens below.
    block_size = max(2, round(1 + uniformity * 7))  # ~4-5 measures at 0.5, ~8 near 1.0
    block_of_measure = [m // block_size for m in range(num_measures)]
    num_blocks = block_of_measure[-1] + 1
    block_vecs = [
        np.mean([measure_vecs[m] for m in range(num_measures) if block_of_measure[m] == b], axis=0)
        for b in range(num_blocks)
    ]

    # Union-find, merging any two blocks whose averaged chroma is similar
    # enough -- this is the "found a returning verse/chorus" step.
    threshold = 0.97 - 0.35 * uniformity  # ~0.79 at 0.5, ~0.62 near 1.0
    parent = list(range(num_blocks))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a in range(num_blocks):
        for b in range(a + 1, num_blocks):
            if float(np.dot(block_vecs[a], block_vecs[b])) >= threshold:
                union(a, b)

    # Remap union-find roots to small, contiguous ids.
    root_to_group: Dict[int, int] = {}
    group_of_block = []
    for b in range(num_blocks):
        root = find(b)
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group)
        group_of_block.append(root_to_group[root])

    return {m: group_of_block[block_of_measure[m]] for m in range(num_measures)}


def measure_index_for_time(time_ms: float, offset_ms: float, measure_length_ms: float) -> int:
    return int((time_ms - offset_ms) // measure_length_ms)


def blended_choice(seed: int, label: str, group_id: Optional[int], position_key: Hashable,
                    uniformity: float, group_fn: Callable[[random.Random], T],
                    indep_fn: Callable[[], T]) -> T:
    """Draw a structural decision, blended between `indep_fn` (the
    caller's existing, fully independent decision) and `group_fn` (a
    decision reused by every occurrence sharing this exact
    (label, group_id, position_key)).

    With probability `uniformity`, every call sharing that key resolves to
    the exact same value -- what lets a chorus's second repeat reuse its
    first repeat's circle/slider or stream/stack choice at the same beat
    position. Otherwise it falls back to `indep_fn`, exactly as if
    uniformity were 0. That "with probability uniformity" roll is itself
    stable per (label, group_id, position_key), not re-rolled on every
    call, so a given position in a pattern either matches its group or it
    doesn't -- it never flickers between the two for the same seed.
    """
    if group_id is None or uniformity <= 0.0:
        return indep_fn()
    key = f"{label}:{group_id}:{position_key}:{seed}"
    selector = random.Random(f"uniformity_select:{key}")
    if selector.random() < uniformity:
        group_rng = random.Random(f"uniformity_group:{key}")
        return group_fn(group_rng)
    return indep_fn()
