"""
Uniformity — how strongly a returning section (a verse's second repeat, a
chorus that comes back after a bridge) reuses its earlier circle/slider and
hitsound choices, on top of what apply_style.py's `find_repeating_measure_map`
already does unconditionally.

`find_repeating_measure_map` only ever catches an *exact* run of matching
energy buckets, `window` measures long — real repeats, but a narrow net. This
module adds a `uniformity` knob (0-1) on top of it:

  0   -> exactly `find_repeating_measure_map`'s own strict result, unchanged.
         This is what every call site already used unconditionally, so
         `uniformity=0` is a byte-for-byte no-op.
  1   -> also catches shorter, looser near-matches (a section that mostly,
         not exactly, sounds like an earlier one), and reuses that earlier
         section's pattern for all of them.
  in between -> the loosening (shorter window, more tolerated mismatches)
         phases in gradually, and even a near-match found this way is only
         *actually* reused with probability `uniformity` -- so a mid-value
         doesn't turn every similar-sounding passage into an exact repeat,
         just makes it more likely without guaranteeing it.

Callers that want a structural decision (which today's pipeline decides
independently, e.g. slider-vs-circle chaining) to participate the same way
can use `blended_choice` directly against a repeat map's canonical measure
id, without needing any of the above.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Hashable, Optional, TypeVar

T = TypeVar("T")


def fuzzy_repeat_map(measure_buckets: Dict[int, int], uniformity: float, seed: int,
                      base_window: int = 4) -> Dict[int, int]:
    """`apply_style.find_repeating_measure_map`, widened by `uniformity`.

    Always starts from that function's own strict, exact-match result
    (window=`base_window`) -- so at uniformity 0 this returns exactly what
    every call site already computed unconditionally before this knob
    existed, unchanged. Above 0, additionally looks for shorter windows
    (down to 2 measures) tolerating a growing number of mismatched buckets,
    and only actually applies a match found this way with probability
    `uniformity` -- each candidate's own coin flip, seeded stably off the
    two spans being compared, so it doesn't flicker between calls for the
    same seed.
    """
    uniformity = max(0.0, min(1.0, uniformity))
    from apply_style import find_repeating_measure_map  # local: avoids a circular import at module load

    result = find_repeating_measure_map(measure_buckets, window=base_window)
    if uniformity <= 0.0 or not measure_buckets:
        return result

    n = max(measure_buckets) + 1
    window = max(2, round(base_window - uniformity * (base_window - 2)))  # base_window at 0 -> 2 at 1
    tolerance = round(uniformity * (window - 1))  # 0 mismatches tolerated at low uniformity, up to window-1 at 1
    if n < window * 2:
        return result

    already_mapped = {m for m, v in result.items() if v != m}

    for a in range(n - window + 1):
        if all((a + k) in already_mapped for k in range(window)):
            continue
        sig_a = [measure_buckets.get(a + k, 0) for k in range(window)]
        for b in range(a + window, n - window + 1):
            if all((b + k) in already_mapped for k in range(window)):
                continue
            sig_b = [measure_buckets.get(b + k, 0) for k in range(window)]
            mismatches = sum(1 for x, y in zip(sig_a, sig_b) if x != y)
            if mismatches > tolerance:
                continue
            selector = random.Random(f"fuzzy_repeat:{a}:{b}:{seed}")
            if selector.random() >= uniformity:
                continue
            for k in range(window):
                idx = b + k
                if idx not in already_mapped:
                    result[idx] = a + k
                    already_mapped.add(idx)
    return result


def blended_choice(seed: int, label: str, group_id: Optional[Hashable], position_key: Hashable,
                    uniformity: float, group_fn: Callable[[random.Random], T],
                    indep_fn: Callable[[], T]) -> T:
    """Draw a structural decision, blended between `indep_fn` (a caller's
    existing, fully independent decision) and `group_fn` (a decision reused
    by every occurrence sharing this exact (label, group_id, position_key) --
    typically `group_id` is a repeat map's canonical measure index).

    With probability `uniformity`, every call sharing that key resolves to
    the exact same value -- what lets a chorus's second repeat reuse its
    first repeat's circle/slider choice at the same beat position.
    Otherwise it falls back to `indep_fn`, exactly as if uniformity were 0.
    That "with probability uniformity" roll is itself stable per
    (label, group_id, position_key), not re-rolled per call, so a given
    position in a pattern either matches its group or it doesn't -- it
    never flickers between the two for the same seed.
    """
    if group_id is None or uniformity <= 0.0:
        return indep_fn()
    key = f"{label}:{group_id}:{position_key}:{seed}"
    selector = random.Random(f"uniformity_select:{key}")
    if selector.random() < uniformity:
        group_rng = random.Random(f"uniformity_group:{key}")
        return group_fn(group_rng)
    return indep_fn()
