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
import datetime
import glob
import os

import matplotlib
matplotlib.use("Agg")  # headless -- this module never opens a window
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from beatmap_judge import Finding, PASS, WARN, FAIL, judge_beatmap
from beatmap_stats import BeatmapStats, Distribution, compute_stats
from beatmap_utils import extract_osz, guess_tier

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
    ("combo_length", "Combo length", "objects"),
    ("spacing_per_beat", "Spacing per beat (distance-snap)", "px/beat"),
]
# Grid shape the histogram page lays HIST_FIELDS out in -- kept alongside
# it so the two never drift apart (one entry added to HIST_FIELDS with no
# matching grid cell would just silently not get plotted).
HIST_GRID = (3, 2)  # (rows, cols)

# A small, consistent visual identity for the whole document -- one accent
# pair (generated vs. reference), one neutral ink/muted/rule/panel scale,
# and one font family, all defined once here rather than repeated (and
# drifting) across every page-building function below.
GENERATED_COLOR = "#4c72b0"
REFERENCE_COLOR = "#dd8452"
INK = "#1c1e26"
MUTED = "#6b7280"
RULE = "#d8dbe2"
PANEL = "#f4f5f7"
FONT_FAMILY = "DejaVu Sans"  # matplotlib's own default -- set explicitly so every page agrees

# Verdict colors for the judgment table -- a fail on a Rule reads as urgent
# (red), a guideline miss as advisory (amber), a pass as quiet confirmation
# (green), all pale enough that the black verdict/clause text stays legible.
VERDICT_COLOR = {PASS: "#2e7d4f", WARN: "#b8860b", FAIL: "#c0392b"}
VERDICT_ROW_TINT = {PASS: "#eef7f1", WARN: "#fdf6e8", FAIL: "#fbeceb"}

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "text.color": INK,
    "axes.edgecolor": RULE,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": RULE,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "axes.axisbelow": True,
})


def _add_footer(fig, page_num: int, total_pages: int, gen_label: str) -> None:
    """A consistent footer on every page: report identity on the left,
    page number on the right -- the same "this is one document, not a
    pile of loose charts" cue a real report's master page would give."""
    fig.text(0.06, 0.02, f"{gen_label} — Statistics Report", fontsize=7.5, color=MUTED, ha="left")
    fig.text(0.94, 0.02, f"{page_num} / {total_pages}", fontsize=7.5, color=MUTED, ha="right")


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def default_report_path(beatmap_path: str) -> str:
    """output/<beatmap name>/report.pdf -- a report gets its own subfolder
    named after the input file (extension stripped), matching the layout
    main.py already writes generated .osu/.osz files under."""
    name = os.path.splitext(os.path.basename(beatmap_path))[0]
    return os.path.join(OUTPUT_DIR, name, "report.pdf")


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
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, pad=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(RULE)

    if gen_dist is None and ref_dist is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes,
                 fontsize=10, color=MUTED, style="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    for dist, color, label in ((gen_dist, GENERATED_COLOR, gen_label), (ref_dist, REFERENCE_COLOR, ref_label)):
        if dist is None or not dist.histogram:
            continue
        centers = [(lo + hi) / 2.0 for lo, hi, _ in dist.histogram]
        widths = [(hi - lo) for lo, hi, _ in dist.histogram]
        total = sum(c for _, _, c in dist.histogram) or 1
        heights = [c / total for _, _, c in dist.histogram]  # normalized -- comparable across different note counts
        ax.bar(centers, heights, width=widths, align="center", color=color, alpha=0.6,
               edgecolor=color, linewidth=0.8, label=f"{label}  (median {dist.median:.2f}{unit})")

    ax.set_xlabel(unit, fontsize=9, color=MUTED)
    ax.set_ylabel("Fraction of objects", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8)
    legend = ax.legend(fontsize=7.5, loc="upper right", frameon=True, framealpha=0.9,
                        edgecolor=RULE, fancybox=False)
    legend.get_frame().set_linewidth(0.6)


def _summary_rows(gen: BeatmapStats, ref: BeatmapStats | None, has_ref: bool) -> list[tuple[str, ...]]:
    """One row per metric: (label, generated-value) if nothing is being
    compared against at all, or (label, generated-value, reference-value)
    when `has_ref` -- has_ref is passed in rather than inferred from
    `ref is not None` alone so a *mixed* report (some stages have a match,
    some don't -- e.g. a .osz tier with no corresponding Backstabber
    difficulty) still gets a reference column throughout, just "—" for the
    stages missing one, instead of the column itself appearing and
    disappearing page to page."""
    def fmt(stats: BeatmapStats | None, fn) -> str:
        return fn(stats) if stats is not None else "—"

    def row(label: str, gen_val: str, ref_val: str) -> tuple[str, ...]:
        return (label, gen_val, ref_val) if has_ref else (label, gen_val)

    rows = [
        row("BPM", fmt(gen, lambda s: f"{s.bpm:.1f}"), fmt(ref, lambda s: f"{s.bpm:.1f}")),
        row("Objects", fmt(gen, lambda s: str(s.n_objects)), fmt(ref, lambda s: str(s.n_objects))),
        row("Circles", fmt(gen, lambda s: str(s.n_circles)), fmt(ref, lambda s: str(s.n_circles))),
        row("Sliders", fmt(gen, lambda s: str(s.n_sliders)), fmt(ref, lambda s: str(s.n_sliders))),
        row("Stacked pairs (<3px)", fmt(gen, lambda s: f"{100*s.stack_fraction:.1f}%"),
            fmt(ref, lambda s: f"{100*s.stack_fraction:.1f}%")),
    ]
    total_gen = gen.single_anchor_straight + gen.single_anchor_curved
    if total_gen:
        rows.append(row("Straight sliders", f"{100*gen.single_anchor_straight/total_gen:.0f}%",
                         (f"{100*ref.single_anchor_straight/(ref.single_anchor_straight + ref.single_anchor_curved):.0f}%"
                          if ref is not None and (ref.single_anchor_straight + ref.single_anchor_curved) else "—")))
    for field, label, unit in HIST_FIELDS:
        gd: Distribution | None = getattr(gen, field)
        rd: Distribution | None = getattr(ref, field) if ref is not None else None
        rows.append(row(f"{label} (median)",
                         f"{gd.median:.2f}{unit}" if gd else "—",
                         f"{rd.median:.2f}{unit}" if rd else "—"))
    return rows


def _style_table(table, n_cols: int) -> None:
    """Header row in the accent color with white text, alternating body-row
    shading, and thin, consistent rules everywhere else — the difference
    between "a matplotlib table" and something that reads as a real report
    table at a glance."""
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(INK)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(PANEL if row % 2 == 0 else "white")
            cell.set_text_props(color=INK)


def _judgment_page(pdf, tier: str, findings: list[Finding], page_num: int, total_pages: int,
                    gen_label: str) -> None:
    """One page per stage: every ranking-criteria check decidable from the
    .osu file alone (see beatmap_judge.py), as a colored pass/warn/fail
    table, with an overall verdict line summarizing the tally."""
    n_pass = sum(1 for f in findings if f.verdict == PASS)
    n_warn = sum(1 for f in findings if f.verdict == WARN)
    n_fail = sum(1 for f in findings if f.verdict == FAIL)

    n_rows = max(1, len(findings))
    fig_height = min(11.0, 0.42 * n_rows + 2.2)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.axis("off")

    fig.suptitle(f"{tier.capitalize()} — Ranking Criteria Judgment", fontsize=15, fontweight="bold",
                 color=INK, y=(fig_height - 0.32) / fig_height)
    if n_fail:
        verdict_text, verdict_color = f"{n_fail} rule failure(s) — not ready to submit", VERDICT_COLOR[FAIL]
    elif n_warn:
        verdict_text, verdict_color = f"{n_warn} guideline warning(s) — worth a look", VERDICT_COLOR[WARN]
    else:
        verdict_text, verdict_color = "All checks pass", VERDICT_COLOR[PASS]
    fig.text(0.5, (fig_height - 0.6) / fig_height, verdict_text, fontsize=11, fontweight="bold",
              color=verdict_color, ha="center")
    fig.text(0.5, (fig_height - 0.82) / fig_height,
              f"{n_pass} pass · {n_warn} warn · {n_fail} fail  (of {len(findings)} checks decidable from the .osu file)",
              fontsize=8.5, color=MUTED, ha="center")

    if findings:
        cell_text = [[f.kind, f.clause, f.verdict, f.detail] for f in findings]
        table = ax.table(cellText=cell_text, colLabels=["Type", "Clause", "Verdict", "Detail"],
                          loc="upper center", cellLoc="left",
                          colWidths=[0.1, 0.22, 0.1, 0.58],
                          bbox=(0, 0, 1, (fig_height - 1.15) / fig_height))
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(RULE)
            cell.set_linewidth(0.6)
            cell.PAD = 0.02
            if row == 0:
                cell.set_facecolor(INK)
                cell.set_text_props(color="white", fontweight="bold")
                continue
            finding = findings[row - 1]
            cell.set_facecolor(VERDICT_ROW_TINT[finding.verdict])
            cell.set_text_props(color=INK, wrap=True)
            if col == 2:
                cell.set_text_props(color=VERDICT_COLOR[finding.verdict], fontweight="bold", wrap=True)
    else:
        fig.text(0.5, 0.5, "No checks applicable at this tier.", fontsize=10, color=MUTED,
                  ha="center", style="italic")

    fig.text(0.06, 0.02, "Checks cover geometry, timing, and [Difficulty] settings only -- anything "
                          "tied to the music itself (musical timing of cues, hitsounds, difficulty "
                          "spikes vs. song intensity) is outside what a file alone can judge.",
              fontsize=6.5, color=MUTED, ha="left", style="italic", wrap=True)
    _add_footer(fig, page_num, total_pages, gen_label)
    pdf.savefig(fig)
    plt.close(fig)


def _cover_page(pdf, gen_label: str, stage_names: list[str], ref_label: str | None) -> None:
    """A proper title page — the same reason a real report doesn't open
    straight on page one of data: it names what the document is, what
    it's measuring against (if anything -- `ref_label` is None for a
    report with no comparison at all), and when it was generated, before
    any chart asks the reader to already know that context."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.62, "Statistics Report", fontsize=26, fontweight="bold",
              color=INK, ha="center")
    fig.text(0.5, 0.565, gen_label, fontsize=16, color=MUTED, ha="center")

    fig.add_artist(plt.Line2D([0.15, 0.85], [0.52, 0.52], color=RULE, linewidth=1, transform=fig.transFigure))

    sections_y = 0.47
    if ref_label is not None:
        fig.text(0.15, 0.47, "Compared against", fontsize=9, color=MUTED)
        fig.text(0.15, 0.44, ref_label, fontsize=12, color=INK, fontweight="bold")
        sections_y = 0.395
    fig.text(0.15, sections_y, "Sections in this report", fontsize=9, color=MUTED)
    for i, name in enumerate(stage_names):
        fig.text(0.17, sections_y - 0.03 - i * 0.028, f"•  {name.capitalize()}", fontsize=10.5, color=INK)

    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.15, 0.10, f"Generated {generated_at}", fontsize=8.5, color=MUTED)
    fig.text(0.15, 0.075, "Timing distributions are compared in beats, normalized to each map's own BPM.",
             fontsize=8, color=MUTED, style="italic")

    # Legend swatches, matching every histogram page exactly -- just the
    # generated map's own color if there's nothing to compare against.
    swatches = [(GENERATED_COLOR, gen_label)]
    if ref_label is not None:
        swatches.append((REFERENCE_COLOR, ref_label))
    for i, (color, label) in enumerate(swatches):
        y = 0.20 - i * 0.032
        fig.add_artist(plt.Rectangle((0.15, y), 0.02, 0.016, color=color, alpha=0.7,
                                      transform=fig.transFigure, clip_on=False))
        fig.text(0.185, y + 0.002, label, fontsize=9, color=INK)

    pdf.savefig(fig)
    plt.close(fig)


def render_report(tier_stats: dict[str, tuple[BeatmapStats, BeatmapStats | None]], output_pdf: str,
                   gen_label: str = "Generated", ref_label: str = EXAMPLE_LABEL) -> None:
    """Write a multi-page PDF: a title page, one summary table page, then
    one page of overlaid histograms per stage in `tier_stats` (stage name
    -> (generated stats, reference stats-or-None if no matching example
    diff was found)). "Stage" here is whatever key the caller passes —
    a difficulty tier for the main pipeline, or any other label (e.g. a
    Base Map v2 pathway stage) — render_report itself never looks it up
    against the example set; that's build_report_for_tiers' job, for
    callers that do mean an actual difficulty tier.

    `ref_label` names whatever `tier_stats`' reference stats actually are
    (defaults to Backstabber, but a caller comparing against something
    else -- e.g. gui_v2.py's ReportWindow -- passes that beatmap's own
    name instead). If *no* stage has a reference at all, the whole
    reference column/legend/swatch is left out of every page rather than
    showing a "Backstabber" (or whatever) column full of "—"; this is
    decided once, from whether any stage has a reference, not per stage,
    so the column doesn't appear and disappear page to page in a mixed
    report (see _summary_rows for why that matters there specifically).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)) or ".", exist_ok=True)
    has_ref = any(ref is not None for _, ref in tier_stats.values())
    total_pages = 2 + 2 * len(tier_stats)  # cover + summary + (histogram + judgment) per stage
    with PdfPages(output_pdf) as pdf:
        _cover_page(pdf, gen_label, list(tier_stats.keys()), ref_label if has_ref else None)

        # --- Summary table page ---
        n_tiers = len(tier_stats)
        fig_height = 2.6 * n_tiers + 1.3
        fig, axes = plt.subplots(n_tiers, 1, figsize=(8.5, fig_height))
        if n_tiers == 1:
            axes = [axes]
        # Title/subtitle placed at a constant *inch* offset from the top
        # edge, not a fixed fraction of the figure height -- fig_height
        # itself varies with n_tiers, so a fixed fraction (e.g. y=0.98)
        # lands at a different absolute distance from the top on every
        # page, close enough on a short (single-tier) page to overlap.
        fig.suptitle("Summary", fontsize=16, fontweight="bold", color=INK,
                     y=(fig_height - 0.32) / fig_height)
        subtitle = f"{gen_label}  vs.  {ref_label}" if has_ref else gen_label
        fig.text(0.5, (fig_height - 0.62) / fig_height, subtitle, fontsize=10, color=MUTED, ha="center")
        col_labels = ["Metric", gen_label, ref_label] if has_ref else ["Metric", gen_label]
        for ax, (tier, (gen, ref)) in zip(axes, tier_stats.items()):
            ax.axis("off")
            ax.set_title(tier.capitalize(), fontsize=11, fontweight="bold", color=INK, loc="left", pad=8)
            rows = _summary_rows(gen, ref, has_ref)
            table = ax.table(cellText=[list(r) for r in rows], colLabels=col_labels,
                              loc="center", cellLoc="center", bbox=(0, 0, 1, 1))
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            _style_table(table, len(col_labels))
        fig.tight_layout(rect=(0.02, 0.04, 0.98, (fig_height - 1.0) / fig_height))
        _add_footer(fig, 2, total_pages, gen_label)
        pdf.savefig(fig)
        plt.close(fig)

        # --- One histogram page, then one judgment page, per stage ---
        page_num = 3
        rows, cols = HIST_GRID
        for tier, (gen, ref) in tier_stats.items():
            fig, axes = plt.subplots(rows, cols, figsize=(8.5, 3.0 * rows + 2.0))
            fig.suptitle(tier.capitalize(), fontsize=15, fontweight="bold", color=INK, y=0.975)
            subtitle = f"{gen_label}  vs.  {ref_label}  ·  normalized to BPM" if has_ref \
                else f"{gen_label}  ·  normalized to BPM"
            fig.text(0.5, 0.965, subtitle, fontsize=9.5, color=MUTED, ha="center")
            for ax, (field, label, unit) in zip(axes.flat, HIST_FIELDS):
                gen_dist = getattr(gen, field)
                ref_dist = getattr(ref, field) if ref is not None else None
                _plot_histogram_pair(ax, gen_dist, ref_dist, gen_label, ref_label, label, unit)
            fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.94))
            _add_footer(fig, page_num, total_pages, gen_label)
            pdf.savefig(fig)
            plt.close(fig)
            page_num += 1

            findings = judge_beatmap(gen.source, tier)
            _judgment_page(pdf, tier, findings, page_num, total_pages, gen_label)
            page_num += 1

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
    parser.add_argument("beatmap", help="Path to the generated .osu file to report on, or a packaged .osz "
                                          "-- every difficulty inside whose Version: names a recognized "
                                          "tier (easy/normal/hard/insane/expert) is reported on.")
    parser.add_argument("--tier", default=None,
                         help="Difficulty name, used to pick the matching Backstabber difficulty to "
                              "compare against for a single .osu input (default: insane). For a .osz "
                              "input, restricts the report to just this one recognized tier instead of "
                              "every tier found inside it.")
    parser.add_argument("--against", default=None,
                         help="Explicit .osu file to compare against instead of auto-picking a "
                              "Backstabber difficulty by --tier. Only valid for a single .osu input.")
    parser.add_argument("--output", default=None,
                         help="Path to write the PDF report to. Defaults to "
                              "output/<beatmap name>/report.pdf, named after the input file.")
    parser.add_argument("--label", default="Generated", help="Label for the generated map in the report.")
    args = parser.parse_args()

    output_pdf = args.output or default_report_path(args.beatmap)

    if args.beatmap.lower().endswith(".osz"):
        if args.against:
            raise SystemExit("--against is only valid for a single .osu input, not a .osz package.")
        osu_paths = extract_osz(args.beatmap)
        tier_paths: dict[str, str] = {}
        for path in osu_paths:
            tier = guess_tier(path)
            if tier is not None and tier not in tier_paths:  # first match wins over a same-tier guest diff
                tier_paths[tier] = path
        if args.tier:
            if args.tier.lower() not in tier_paths:
                raise SystemExit(f"No difficulty matching --tier {args.tier!r} found inside {args.beatmap}. "
                                  f"Found: {', '.join(sorted(tier_paths)) or '(none recognized)'}")
            tier_paths = {args.tier.lower(): tier_paths[args.tier.lower()]}
        if not tier_paths:
            raise SystemExit(f"No recognizable difficulty names (easy/normal/hard/insane/expert) found "
                              f"inside {args.beatmap}.")
        build_report_for_tiers(tier_paths, output_pdf, gen_label=args.label)
        return

    tier = args.tier or "insane"
    gen_stats = compute_stats(args.beatmap)
    if args.against:
        ref_stats = compute_stats(args.against)
        ref_label = os.path.splitext(os.path.basename(args.against))[0]
    else:
        ref_path = find_reference_osu(tier)
        ref_stats = compute_stats(ref_path) if ref_path else None
        ref_label = EXAMPLE_LABEL
        if ref_stats is None:
            print(f"Warning: no Backstabber difficulty matching {tier!r} found; "
                  f"report will show generated stats only.")

    render_report({tier: (gen_stats, ref_stats)}, output_pdf, gen_label=args.label, ref_label=ref_label)


if __name__ == "__main__":
    main()
