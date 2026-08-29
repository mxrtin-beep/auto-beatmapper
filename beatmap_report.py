#!/usr/bin/env python3
"""
Statistical report — a PDF of histograms and tables (rather than the plain
console dump beatmap_stats.py prints) comparing a generated beatmap's
difficulties against the matching difficulties of the bundled example map,
Ke$ha - Backstabber (Sped Up & Cut Ver.), the reference this project's
"does this feel like a real map" judgment is calibrated against.

Every timing-based distribution (delay between objects, slider duration) is
compared in *beats*, not milliseconds — two maps at different BPM cannot be
compared on raw millisecond gaps (a 200ms gap is a slow half-beat at 150 BPM
but a brisk full beat at 300 BPM), so beatmap_stats.py's delay_beats /
slider_beats fields (delay_ms and slider_length_px divided by the map's own
beat_length_ms) are what gets plotted, never the raw-ms fields. Spacing
(on-screen pixel distance) and turn angle are already scale-free and are
compared as-is.

Usage:
    python3 beatmap_report.py mymap.osu --tier insane --output report.pdf
    python3 beatmap_report.py mymap.osu --tier insane --against other.osu --output report.pdf
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")  # headless -- this module never opens a window
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from beatmap_stats import BeatmapStats, Distribution, compute_stats

EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example", "keha_backstabber")
EXAMPLE_LABEL = "Backstabber (reference)"

# (BeatmapStats field, axis label, unit) for every distribution plotted as
# a histogram. Only BPM-normalized/scale-free fields are ever included here
# -- see module docstring.
HIST_FIELDS = [
    ("delay_beats", "Delay between consecutive objects", "beats"),
    ("slider_beats", "Slider duration", "beats"),
    ("spacing_px", "Spacing (on-screen distance)", "px"),
    ("turn_angle_deg", "Turn angle", "°"),
]

GENERATED_COLOR = "#4c72b0"
REFERENCE_COLOR = "#dd8452"


def find_reference_osu(tier: str, example_dir: str = EXAMPLE_DIR) -> str | None:
    """Find the bundled example .osu file whose difficulty name best matches
    `tier` (case-insensitive substring match against the `[Difficulty]`
    bracket, e.g. "insane" matches "[Insane].osu" or "[Hero's Insane].osu").
    An exact bracket match (`[Insane]`) is always preferred over a fuzzy one
    (`[Hero's Insane]`), so the plain difficulty is picked over a guest diff
    sharing the same tier name.
    """
    candidates = sorted(glob.glob(os.path.join(example_dir, "*.osu")))
    if not candidates:
        return None
    exact = f"[{tier.capitalize()}].osu".lower()
    for path in candidates:
        if os.path.basename(path).lower().endswith(exact):
            return path
    tier_lower = tier.lower()
    for path in candidates:
        base = os.path.basename(path).lower()
        if f"[{tier_lower}" in base or f" {tier_lower}]" in base or f" {tier_lower}'s" in base:
            return path
    return None


def _plot_histogram_pair(ax, gen_dist: Distribution | None, ref_dist: Distribution | None,
                          gen_label: str, ref_label: str, title: str, unit: str) -> None:
    ax.set_title(title)
    if gen_dist is None and ref_dist is None:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    for dist, color, label in ((gen_dist, GENERATED_COLOR, gen_label), (ref_dist, REFERENCE_COLOR, ref_label)):
        if dist is None or not dist.histogram:
            continue
        centers = [(lo + hi) / 2.0 for lo, hi, _ in dist.histogram]
        widths = [(hi - lo) for lo, hi, _ in dist.histogram]
        total = sum(c for _, _, c in dist.histogram) or 1
        heights = [c / total for _, _, c in dist.histogram]  # normalized -- comparable across different note counts
        ax.bar(centers, heights, width=widths, align="center", color=color, alpha=0.55,
               edgecolor=color, label=f"{label} (median {dist.median:.2f}{unit})")

    ax.set_xlabel(unit)
    ax.set_ylabel("fraction of objects")
    ax.legend(fontsize=7, loc="upper right")


def _summary_rows(gen: BeatmapStats, ref: BeatmapStats | None) -> list[tuple[str, str, str]]:
    def fmt(stats: BeatmapStats | None, fn) -> str:
        return fn(stats) if stats is not None else "—"

    rows = [
        ("BPM", fmt(gen, lambda s: f"{s.bpm:.1f}"), fmt(ref, lambda s: f"{s.bpm:.1f}")),
        ("Objects", fmt(gen, lambda s: str(s.n_objects)), fmt(ref, lambda s: str(s.n_objects))),
        ("Circles", fmt(gen, lambda s: str(s.n_circles)), fmt(ref, lambda s: str(s.n_circles))),
        ("Sliders", fmt(gen, lambda s: str(s.n_sliders)), fmt(ref, lambda s: str(s.n_sliders))),
        ("Stacked pairs (<3px)", fmt(gen, lambda s: f"{100*s.stack_fraction:.1f}%"),
         fmt(ref, lambda s: f"{100*s.stack_fraction:.1f}%")),
    ]
    total_gen = gen.single_anchor_straight + gen.single_anchor_curved
    if total_gen:
        rows.append(("Straight sliders", f"{100*gen.single_anchor_straight/total_gen:.0f}%",
                      (f"{100*ref.single_anchor_straight/(ref.single_anchor_straight + ref.single_anchor_curved):.0f}%"
                       if ref is not None and (ref.single_anchor_straight + ref.single_anchor_curved) else "—")))
    for field, label, unit in HIST_FIELDS:
        gd: Distribution | None = getattr(gen, field)
        rd: Distribution | None = getattr(ref, field) if ref is not None else None
        rows.append((f"{label} (median)",
                      f"{gd.median:.2f}{unit}" if gd else "—",
                      f"{rd.median:.2f}{unit}" if rd else "—"))
    return rows


def render_report(tier_stats: dict[str, tuple[BeatmapStats, BeatmapStats | None]], output_pdf: str,
                   gen_label: str = "Generated") -> None:
    """Write a multi-page PDF: one summary table page, then one page of
    overlaid histograms per difficulty tier in `tier_stats` (tier name ->
    (generated stats, reference stats-or-None if no matching example diff
    was found))."""
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)) or ".", exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        # --- Summary table page ---
        n_tiers = len(tier_stats)
        fig, axes = plt.subplots(n_tiers, 1, figsize=(8.5, 2.6 * n_tiers + 1))
        if n_tiers == 1:
            axes = [axes]
        fig.suptitle(f"{gen_label} vs. {EXAMPLE_LABEL} — summary", fontsize=13, y=0.995)
        for ax, (tier, (gen, ref)) in zip(axes, tier_stats.items()):
            ax.axis("off")
            ref_col_label = os.path.basename(ref.source) if ref is not None else "(no matching reference diff)"
            rows = _summary_rows(gen, ref)
            table = ax.table(cellText=[[label, g, r] for label, g, r in rows],
                              colLabels=[tier.capitalize(), gen_label, "Backstabber"],
                              loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.3)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig)
        plt.close(fig)

        # --- One histogram page per tier ---
        for tier, (gen, ref) in tier_stats.items():
            fig, axes = plt.subplots(2, 2, figsize=(8.5, 8))
            fig.suptitle(f"{tier.capitalize()} — {gen_label} vs. {EXAMPLE_LABEL}"
                         + (" (BPM-normalized where applicable)"), fontsize=12)
            for ax, (field, label, unit) in zip(axes.flat, HIST_FIELDS):
                gen_dist = getattr(gen, field)
                ref_dist = getattr(ref, field) if ref is not None else None
                _plot_histogram_pair(ax, gen_dist, ref_dist, gen_label, "Backstabber", label, unit)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote statistics report: {output_pdf}")


def build_report_for_tiers(tier_paths: dict[str, str], output_pdf: str, gen_label: str = "Generated",
                            example_dir: str = EXAMPLE_DIR) -> None:
    """Convenience wrapper: `tier_paths` maps a tier name (e.g. "insane") to
    the generated .osu file for it; the matching Backstabber difficulty is
    located automatically for each. Tiers with no generated file are
    skipped silently (gui.py may only have generated a subset)."""
    tier_stats: dict[str, tuple[BeatmapStats, BeatmapStats | None]] = {}
    for tier, path in tier_paths.items():
        if not path or not os.path.isfile(path):
            continue
        gen_stats = compute_stats(path)
        ref_path = find_reference_osu(tier, example_dir)
        ref_stats = compute_stats(ref_path) if ref_path else None
        tier_stats[tier] = (gen_stats, ref_stats)
    if not tier_stats:
        raise RuntimeError("No generated .osu files found to report on.")
    render_report(tier_stats, output_pdf, gen_label=gen_label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PDF statistics report comparing a beatmap "
                                                   "against the bundled Backstabber example, normalized to BPM.")
    parser.add_argument("beatmap", help="Path to the generated .osu file to report on.")
    parser.add_argument("--tier", default="insane",
                         help="Difficulty name, used to pick the matching Backstabber difficulty to "
                              "compare against (default: insane).")
    parser.add_argument("--against", default=None,
                         help="Explicit .osu file to compare against instead of auto-picking a "
                              "Backstabber difficulty by --tier.")
    parser.add_argument("--output", required=True, help="Path to write the PDF report to.")
    parser.add_argument("--label", default="Generated", help="Label for the generated map in the report.")
    args = parser.parse_args()

    gen_stats = compute_stats(args.beatmap)
    if args.against:
        ref_stats = compute_stats(args.against)
    else:
        ref_path = find_reference_osu(args.tier)
        ref_stats = compute_stats(ref_path) if ref_path else None
        if ref_stats is None:
            print(f"Warning: no Backstabber difficulty matching {args.tier!r} found; "
                  f"report will show generated stats only.")

    render_report({args.tier: (gen_stats, ref_stats)}, args.output, gen_label=args.label)


if __name__ == "__main__":
    main()
