#!/usr/bin/env python3
"""
Desktop GUI for the Base Map v2 pathway — generate_base_beatmap_v2.py plus
add_sliders_v2.py, in one window.

This is a deliberately separate app from gui.py, not a tab or checkbox
inside it: the v2 pathway is an independent, standalone alternative to the
main pipeline (see generate_base_beatmap_v2.py's own module docstring),
and it exposes a smaller, pathway-specific set of knobs:

  * Intensity          -- generate_base_beatmap_v2.py's --intensity: how
                           much of the song ends up on half/quarter-beat
                           spacing (and measure-anchored eighth-note
                           bursts) versus staying whole-beat dominant.
  * Slider vs. circle mix -- add_sliders_v2.py's --chain-probability: how
                           often an eligible run of circles actually
                           becomes a slider at all, versus staying plain
                           circles.
  * Slider length       -- add_sliders_v2.py's --slider-length-bias: of
                            whichever runs do become sliders, fewer/longer
                            vs. more/shorter ones.
  * Slider curviness    -- forwarded to add_sliders_v2.py's --curviness
                            (the same knob apply_style.py's own
                            --curviness is, since add_sliders_v2.py
                            re-runs apply_style.py for positioning).
  * Jump distance       -- forwarded to apply_style.py's own --spacing,
                            the same knob that already produces
                            accurately-spaced, error-free output for the
                            main pipeline.

Reuses gui.py's dark theme (_configure_style) and color constants directly
so the two apps look like part of the same project, without duplicating
~150 lines of ttk style setup.

Usage:
    python3 gui_v2.py
"""

from __future__ import annotations

import os
import queue
import random
import shutil
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import add_sliders_v2
import beatmap_report
import generate_base_beatmap_v2
from background_style import apply_background, extract_combo_colors
from beatmap_stats import BeatmapStats, compute_stats
from beatmap_utils import extract_osz, guess_tier
from build_osz import build_osz
from gui import BG, BG_ENTRY, FONT_MONO, PAD_INNER, PAD_OUTER, TextRedirector, _bind_click_to_position, \
    _configure_style, _open_path

# Version names for the three stages, mirroring the main pipeline's own
# Base/Variety/Styled naming.
CIRCLES_VERSION = "Auto Base v2 (Circles)"
SLIDERS_VERSION = "Auto Base v2 (Sliders)"
STYLED_VERSION = "Insane"


@dataclass
class SliderParam:
    """A style knob shown as a plain 0-1 dial, thumb defaulting to the
    middle — but the middle doesn't have to *mean* the middle of the
    underlying range. `actual_lo`/`actual_mid`/`actual_hi` are the real
    values forwarded to the pipeline at display 0 / 0.5 / 1, piecewise-
    linear between them (see to_actual) — every knob's tuned default
    (e.g. Intensity's own 0.65) shows up as a plain, unremarkable 0.5 on
    the dial instead of an oddly-specific number, while the dial's two
    ends can still open the underlying range all the way out."""
    flag: str
    label: str
    description: str
    actual_lo: float
    actual_mid: float
    actual_hi: float

    def to_actual(self, display: float) -> float:
        display = max(0.0, min(1.0, display))
        if display <= 0.5:
            return self.actual_lo + (self.actual_mid - self.actual_lo) * (display / 0.5)
        return self.actual_mid + (self.actual_hi - self.actual_mid) * ((display - 0.5) / 0.5)


SLIDER_PARAMS = [
    SliderParam("--intensity", "Intensity",
                "How much of the song ends up on faster subdivisions. 0 = whole-beat "
                "dominant throughout. 1 = much more of it on half/quarter-beat spacing, "
                "including denser measure-anchored bursts.",
                0.0, 0.65, 1.0),
    SliderParam("--chain-probability", "Slider vs. circle mix",
                "How often an eligible run of adjacent circles actually becomes a slider, "
                "versus staying plain circles. 0 = always circles. 1 = every eligible "
                "run becomes a slider.",
                0.0, 0.3, 1.0),
    SliderParam("--slider-length-bias", "Slider length",
                "Of whichever runs do become sliders: how long they tend to run. "
                "0 = more, shorter/choppier sliders. 1 = fewer, longer sliders.",
                0.0, 0.4, 1.0),
    SliderParam("--curviness", "Slider curviness",
                "How curved slider paths look. 0 = mostly straight lines. "
                "1 = pronounced arcs.",
                0.0, 0.5, 1.0),
    SliderParam("--spacing", "Jump distance",
                "How far apart notes are placed for a given time gap between them. "
                "0 = tight, close together. 1 = wide, dramatic jumps.",
                0.5, 1.9, 2.5),
    SliderParam("--uniformity", "Pattern uniformity",
                "How much the circle/slider layout repeats across measures with the same "
                "note density. 0 = independent every measure. 1 = one fixed pattern "
                "per density.",
                0.0, 0.5, 1.0),
]


@dataclass
class EntryParam:
    flag: str
    label: str
    description: str


TIMING_PARAMS = [
    EntryParam("--bpm", "BPM override",
               "Force a specific tempo instead of detecting it from the song automatically. "
               "Leave blank to auto-detect."),
    EntryParam("--offset", "Beat offset override (ms)",
               "Force the time of the very first beat, in milliseconds, instead of detecting "
               "it automatically. Leave blank to auto-detect."),
]


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Auto Beatmapper — Base Map v2")
        root.geometry("640x680")
        root.minsize(520, 480)
        _configure_style(root)

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.result_error: Exception | None = None
        self.result_path: str | None = None
        self.slider_vars: dict[str, tk.DoubleVar] = {}
        self.entry_vars: dict[str, tk.StringVar] = {}
        self._slider_params_by_flag = {p.flag: p for p in SLIDER_PARAMS}

        outer = ttk.Frame(root)
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, background=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        form = ttk.Frame(canvas)
        form_id = canvas.create_window((0, 0), window=form, anchor="nw")
        form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(form_id, width=e.width))

        def on_mousewheel(event: tk.Event) -> None:
            delta = -1 if event.num == 5 or event.delta < 0 else 1
            canvas.yview_scroll(-delta, "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)
        canvas.bind_all("<Button-4>", on_mousewheel)
        canvas.bind_all("<Button-5>", on_mousewheel)

        # --- Song section ---
        song_panel = self._panel(form, "Song")
        ttk.Label(song_panel, text="Song file", style="Heading.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=PAD_INNER, pady=(PAD_INNER, 2))
        self.audio_var = tk.StringVar()
        self._file_row(song_panel, 1, self.audio_var, self._browse_audio)

        ttk.Label(song_panel, text="Output folder", style="Heading.TLabel").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=PAD_INNER, pady=(14, 2))
        self.outdir_var = tk.StringVar(value="output")
        self._file_row(song_panel, 3, self.outdir_var, self._browse_outdir, last=True)
        song_panel.columnconfigure(0, weight=1)

        # --- Background section ---
        background_panel = self._panel(form, "Background")
        ttk.Label(background_panel, text="Background image", style="Heading.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=PAD_INNER, pady=(PAD_INNER, 2))
        ttk.Label(background_panel,
                  text="Optional. Its most common colors also become the map's combo colors. "
                       "Leave blank for no background and the default colors.",
                  style="Hint.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", padx=PAD_INNER)
        self.background_var = tk.StringVar()
        self._file_row(background_panel, 2, self.background_var, self._browse_background, last=True)
        background_panel.columnconfigure(0, weight=1)

        # --- Song info section ---
        meta_panel = self._panel(form, "Song info")
        self.title_var = tk.StringVar()
        self.artist_var = tk.StringVar(value="Unknown Artist")
        self.creator_var = tk.StringVar(value="auto-beatmapper")
        meta_fields = (
            ("Title", "Leave blank to use the audio file's name.", self.title_var),
            ("Artist", "Who performed the song.", self.artist_var),
            ("Creator", "Your mapper name, credited in the beatmap.", self.creator_var),
        )
        for i, (label, desc, var) in enumerate(meta_fields):
            self._labeled_entry(meta_panel, i, label, desc, var, first=(i == 0), last=(i == len(meta_fields) - 1))
        meta_panel.columnconfigure(1, weight=1)

        # --- Style section ---
        style_panel = self._panel(form, "Style")
        for i, p in enumerate(SLIDER_PARAMS):
            self._slider_row(style_panel, i, p, first=(i == 0))
        style_panel.columnconfigure(0, weight=1)

        # --- Timing section ---
        timing_panel = self._panel(form, "Timing")
        for i, p in enumerate(TIMING_PARAMS):
            var = tk.StringVar()
            self.entry_vars[p.flag] = var
            self._labeled_entry(timing_panel, i, p.label, p.description, var,
                                 first=(i == 0), last=(i == len(TIMING_PARAMS) - 1))
        timing_panel.columnconfigure(1, weight=1)

        # --- Difficulties section ---
        diff_panel = self._panel(form, "Difficulties")
        ttk.Label(diff_panel,
                  text="Insane is always generated. Checking any of Hard/Normal/Easy thins it down "
                       "and gives that tier its own real positioning pass (its own jump distance and "
                       "Difficulty settings -- bigger circles, lower HP drain, lower approach rate), "
                       "not just a rescaled copy of Insane's.", style="Hint.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", padx=PAD_INNER, pady=(PAD_INNER, 6))
        self.difficulty_vars: dict[str, tk.BooleanVar] = {}
        for i, tier in enumerate(("Hard", "Normal", "Easy")):
            var = tk.BooleanVar(value=False)
            self.difficulty_vars[tier] = var
            ttk.Checkbutton(diff_panel, text=tier, variable=var).grid(
                row=1, column=i, sticky="w", padx=(PAD_INNER if i == 0 else 8, 8), pady=(0, PAD_INNER))
        diff_panel.columnconfigure(3, weight=1)

        # --- Options ---
        opt_panel = self._panel(form, "Options")
        self.osz_var = tk.BooleanVar(value=True)
        self.auto_open_var = tk.BooleanVar(value=True)
        self.keep_intermediate_var = tk.BooleanVar(value=False)
        self.report_var = tk.BooleanVar(value=False)
        options = (
            (self.osz_var, "Package as .osz (ready to import into osu!)"),
            (self.auto_open_var, "Open the finished map when done"),
            (self.keep_intermediate_var, "Keep intermediate stages too (Circles and Sliders, "
                                          "alongside the final Styled map)"),
            (self.report_var, "Generate a statistics report (PDF, plotted against Backstabber)"),
        )
        # ttk::checkbutton has no -wraplength option on every platform/Tk
        # build (it raised TclError: unknown option "-wraplength" on
        # Windows/Tk 8.6+, even though ttk::label supports it just fine) --
        # so a checkbox's own label just has to stay short enough to never
        # need wrapping in the first place; anything that needs more
        # explanation gets a separate wrapping Hint.TLabel underneath it
        # instead (ttk::label's -wraplength does work).
        for i, (var, label) in enumerate(options):
            ttk.Checkbutton(opt_panel, text=label, variable=var).grid(
                row=i, column=0, sticky="w", padx=PAD_INNER,
                pady=(PAD_INNER if i == 0 else 6, PAD_INNER if i == len(options) - 1 else 0))
        opt_panel.columnconfigure(0, weight=1)

        # --- Generate button + log ---
        bottom = ttk.Frame(root)
        bottom.pack(fill="both", padx=PAD_OUTER, pady=(6, PAD_OUTER))

        button_row = ttk.Frame(bottom, style="Panel.TFrame")
        button_row.pack(fill="x", pady=(0, 8))
        button_row.columnconfigure(0, weight=1)
        self.generate_button = ttk.Button(button_row, text="Generate", command=self._on_generate)
        self.generate_button.grid(row=0, column=0, sticky="ew")
        # Separate tool, not a Generate option -- runs the statistics
        # report against whatever beatmap you point it at, independent of
        # (and without re-running) this window's own generation pipeline.
        ttk.Button(button_row, text="Statistics Report...", style="Secondary.TButton",
                   command=self._open_report_window).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        log_frame = ttk.Frame(bottom, style="Panel.TFrame")
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, height=14, state="disabled", wrap="word",
                                 background=BG_ENTRY, foreground="#e6e6ec", insertbackground="#e6e6ec",
                                 font=FONT_MONO, relief="flat", padx=10, pady=10)
        self.log_text.pack(fill="both", expand=True)

        self.root.after(100, self._drain_log_queue)

    # --- Layout helpers (same patterns as gui.py) ---

    def _panel(self, parent: tk.Widget, title: str) -> ttk.Labelframe:
        panel = ttk.Labelframe(parent, text=title)
        panel.pack(fill="x", expand=True, padx=PAD_OUTER, pady=(0, PAD_OUTER))
        return panel

    def _file_row(self, parent: tk.Widget, row: int, var: tk.StringVar, browse_cmd, last: bool = False) -> None:
        row_frame = ttk.Frame(parent, style="Panel.TFrame")
        row_frame.grid(row=row, column=0, columnspan=2, sticky="ew",
                        padx=PAD_INNER, pady=(4, PAD_INNER if last else 4))
        row_frame.columnconfigure(0, weight=1)
        ttk.Entry(row_frame, textvariable=var).grid(row=0, column=0, sticky="ew")
        ttk.Button(row_frame, text="Browse...", style="Secondary.TButton",
                   command=browse_cmd).grid(row=0, column=1, padx=(8, 0))

    def _labeled_entry(self, parent: tk.Widget, row: int, label: str, desc: str,
                        var: tk.StringVar, first: bool = False, last: bool = False) -> None:
        ttk.Label(parent, text=label, style="Heading.TLabel").grid(
            row=row * 3, column=0, columnspan=2, sticky="w",
            padx=PAD_INNER, pady=(PAD_INNER if first else 14, 2))
        ttk.Label(parent, text=desc, style="Hint.TLabel").grid(
            row=row * 3 + 1, column=0, columnspan=2, sticky="w", padx=PAD_INNER)
        ttk.Entry(parent, textvariable=var).grid(
            row=row * 3 + 2, column=0, columnspan=2, sticky="ew",
            padx=PAD_INNER, pady=(4, PAD_INNER if last else 0))

    def _slider_row(self, parent: tk.Widget, row: int, p: SliderParam, first: bool = False) -> None:
        # Every dial here is a plain 0-1 scale, thumb defaulting to the
        # middle -- see SliderParam's own docstring for why that middle
        # doesn't have to be the middle of the real underlying range.
        var = tk.DoubleVar(value=0.5)
        self.slider_vars[p.flag] = var
        base_row = row * 4
        ttk.Label(parent, text=p.label, style="Heading.TLabel").grid(
            row=base_row, column=0, sticky="w", padx=PAD_INNER, pady=(PAD_INNER if first else 14, 2))
        value_label = ttk.Label(parent, style="Value.TLabel", width=6, anchor="e")
        value_label.grid(row=base_row, column=1, sticky="e", padx=PAD_INNER)

        def on_change(_evt=None, var=var, value_label=value_label) -> None:
            value_label.configure(text=f"{var.get():.2f}")

        on_change()
        ttk.Label(parent, text=p.description, style="Hint.TLabel").grid(
            row=base_row + 1, column=0, columnspan=2, sticky="w", padx=PAD_INNER)
        scale = ttk.Scale(parent, from_=0.0, to=1.0, orient="horizontal", variable=var, command=on_change)
        scale.grid(row=base_row + 2, column=0, columnspan=2, sticky="ew", padx=PAD_INNER, pady=(6, 4))
        _bind_click_to_position(scale, var, 0.0, 1.0, on_change)

    # --- File pickers ---

    def _browse_audio(self) -> None:
        path = filedialog.askopenfilename(title="Choose a song",
                                           filetypes=[("Audio files", "*.mp3 *.wav *.ogg"), ("All files", "*.*")])
        if path:
            self.audio_var.set(path)

    def _browse_outdir(self) -> None:
        path = filedialog.askdirectory(title="Choose an output folder")
        if path:
            self.outdir_var.set(path)

    def _browse_background(self) -> None:
        path = filedialog.askopenfilename(title="Choose a background image",
                                           filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"),
                                                      ("All files", "*.*")])
        if path:
            self.background_var.set(path)

    def _open_report_window(self) -> None:
        ReportWindow(self.root)

    # --- Log ---

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                self._append_log(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log_queue)

    # --- Generation ---

    def _on_generate(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        audio = self.audio_var.get().strip()
        if not audio:
            messagebox.showerror("Missing song", "Choose an MP3 file first.")
            return
        if not os.path.isfile(audio):
            messagebox.showerror("File not found", f"Can't find:\n{audio}")
            return

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.generate_button.configure(state="disabled", text="Generating...")

        self.worker = threading.Thread(target=self._run, args=(audio,), daemon=True)
        self.worker.start()
        self.root.after(200, self._poll_worker)

    def _actual(self, flag: str) -> float:
        """The real value forwarded to the pipeline for `flag`'s dial,
        mapped from its current 0-1 display position — see SliderParam.to_actual."""
        return self._slider_params_by_flag[flag].to_actual(self.slider_vars[flag].get())

    def _run(self, audio: str) -> None:
        old_stdout, old_stderr, old_argv = sys.stdout, sys.stderr, sys.argv
        redirector = TextRedirector(self.log_queue)
        sys.stdout = redirector
        sys.stderr = redirector
        self.result_error = None
        self.result_path = None
        try:
            title = self.title_var.get().strip() or os.path.splitext(os.path.basename(audio))[0]
            outdir = self.outdir_var.get().strip() or "output"
            os.makedirs(outdir, exist_ok=True)
            keep_intermediate = self.keep_intermediate_var.get()
            seed = random.SystemRandom().randrange(2**32)

            # Copied into outdir (like a difficulty file, not a cleaned-up
            # intermediate) so it's right there for a loose-file user, not
            # just inside a .osz -- same reasoning as the audio itself.
            background_path = None
            background_colours = None
            background = self.background_var.get().strip()
            if background:
                if not os.path.isfile(background):
                    raise RuntimeError(f"Background image not found: {background}")
                background_path = os.path.join(outdir, os.path.basename(background))
                if os.path.abspath(background_path) != os.path.abspath(background):
                    shutil.copyfile(background, background_path)
                background_colours = extract_combo_colors(background_path)

            circles_path = os.path.join(outdir, f"{title} [{CIRCLES_VERSION}].osu")
            sliders_path = os.path.join(outdir, f"{title} [{SLIDERS_VERSION}].osu")
            styled_path = os.path.join(outdir, f"{title} [{STYLED_VERSION}].osu")

            base_argv = [audio, "--output", circles_path, "--title", title,
                         "--artist", self.artist_var.get().strip() or "Unknown Artist",
                         "--creator", self.creator_var.get().strip() or "auto-beatmapper",
                         "--version", CIRCLES_VERSION,
                         "--intensity", f"{self._actual('--intensity'):.3f}",
                         "--seed", str(seed)]
            for p in TIMING_PARAMS:
                value = self.entry_vars[p.flag].get().strip()
                if value:
                    base_argv += [p.flag, value]
            sys.argv = ["generate_base_beatmap_v2.py"] + base_argv
            generate_base_beatmap_v2.main()

            # Difficulty spread: Insane (--output, always) plus whichever of
            # Hard/Normal/Easy are checked, each with its own
            # thin-then-apply_style.py pass (add_sliders_v2.py's build_tier)
            # rather than derived from Insane's own already-styled positions
            # the way make_easy.py's spread works for the main pipeline --
            # every tier gets its own real apply_style.py output (and its
            # own, scaled-down --spacing for the easier tiers), instead of
            # a lower tier inheriting Insane's tighter jump distance on top
            # of its own much bigger circles.
            tier_paths: dict[str, str] = {}
            sliders_argv = [circles_path, audio, "--output", styled_path, "--version", STYLED_VERSION,
                             "--chain-probability", f"{self._actual('--chain-probability'):.3f}",
                             "--slider-length-bias", f"{self._actual('--slider-length-bias'):.3f}",
                             "--curviness", f"{self._actual('--curviness'):.3f}",
                             "--spacing", f"{self._actual('--spacing'):.3f}",
                             "--uniformity", f"{self._actual('--uniformity'):.3f}",
                             "--seed", str(seed)]
            # The merged-but-unstyled "Sliders" stage is only ever worth
            # writing out when the user actually wants to inspect it --
            # otherwise it's exactly what add_sliders_v2.py already treats
            # it as: a hidden working file, cleaned up on its own once
            # apply_style.py is done reading it.
            if keep_intermediate:
                sliders_argv += ["--merged-output", sliders_path, "--merged-version", SLIDERS_VERSION]
            for tier in ("Hard", "Normal", "Easy"):
                if not self.difficulty_vars[tier].get():
                    continue
                tier_path = os.path.join(outdir, f"{title} [{tier}].osu")
                sliders_argv += [f"--{tier.lower()}-output", tier_path]
                tier_paths[tier] = tier_path
            sys.argv = ["add_sliders_v2.py"] + sliders_argv
            add_sliders_v2.main()

            if self.report_var.get():
                # Compared against Backstabber's Insane -- the closest
                # thing to a formal "tier" this pathway has right now,
                # since it doesn't yet derive a difficulty spread of its
                # own (see gui_v2.py's own module docstring). For a report
                # against something else, use the separate "Statistics
                # Report..." tool instead (ReportWindow, below).
                try:
                    gen_stats = compute_stats(styled_path)
                    ref_path = beatmap_report.find_reference_osu("insane")
                    ref_stats = compute_stats(ref_path) if ref_path else None
                    report_path = os.path.join(outdir, f"{title} (Base v2 Statistics Report).pdf")
                    beatmap_report.render_report({"styled": (gen_stats, ref_stats)},
                                                  report_path, gen_label=title)
                    self.log_queue.put(f"Wrote statistics report: {report_path}\n")
                except Exception as e:  # noqa: BLE001 - the report is a bonus, never fail generation over it
                    self.log_queue.put(f"\n(couldn't generate statistics report: {e})\n")

            # Circles/Sliders are worth keeping around at all only when the
            # user actually checked the box -- otherwise they're the same
            # kind of internal working file main.py's own Base/Variety are
            # without --keep-intermediate-files, cleaned up once the thing
            # that's actually meant to be played (Styled) exists. The
            # derived difficulties (if any were checked) are always kept.
            all_paths = [circles_path, sliders_path, styled_path] if keep_intermediate else [styled_path]
            all_paths += list(tier_paths.values())
            if not keep_intermediate:
                os.remove(circles_path)

            if background_path is not None:
                background_filename = os.path.basename(background_path)
                for path in all_paths:
                    apply_background(path, background_filename, background_colours)

            if self.osz_var.get():
                osz_path = os.path.join(outdir, f"{title} (Base v2).osz")
                build_osz(all_paths, audio, osz_path,
                          extra_files=[background_path] if background_path else None)
                self.log_queue.put(f"Packaged {osz_path}\n")
                for path in all_paths:
                    os.remove(path)
                self.result_path = osz_path
            else:
                self.result_path = styled_path
        except SystemExit as e:
            if e.code not in (None, 0):
                self.result_error = RuntimeError(f"Exited with code {e.code}")
        except Exception as e:  # noqa: BLE001 - surfaced to the user as-is
            self.result_error = e
        finally:
            sys.stdout, sys.stderr, sys.argv = old_stdout, old_stderr, old_argv

    def _poll_worker(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.root.after(200, self._poll_worker)
            return
        self.generate_button.configure(state="normal", text="Generate")
        if self.result_error is not None:
            messagebox.showerror("Generation failed", str(self.result_error))
            return
        if self.auto_open_var.get() and self.result_path:
            _open_path(self.result_path)


class ReportWindow:
    """Standalone "Statistics Report..." tool window: generate a report PDF
    for any beatmap you point it at, compared against Backstabber, an
    explicit reference beatmap, or nothing at all -- entirely independent
    of App's own Generate pipeline (no beatmap has to have just been
    generated, and generating one doesn't touch this window's state)."""

    def __init__(self, master: tk.Tk) -> None:
        self.win = tk.Toplevel(master)
        self.win.title("Auto Beatmapper — Statistics Report")
        # A plain Toplevel starts on Tk's own default (light) background --
        # _configure_style() only ever themed `master`'s own root window,
        # so without this the window's raw background shows through as a
        # pale border around every dark ttk panel, instead of matching the
        # rest of the app.
        self.win.configure(bg=BG)
        self.win.geometry("620x760")
        self.win.minsize(520, 520)
        self.win.transient(master)

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.result_error: Exception | None = None
        self.result_path: str | None = None

        outer = ttk.Frame(self.win)
        outer.pack(fill="both", expand=True, padx=PAD_OUTER, pady=PAD_OUTER)

        # --- Beatmap section ---
        song_panel = self._panel(outer, "Beatmap to report on")
        ttk.Label(song_panel, text="Beatmap file (.osu or .osz)", style="Heading.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=PAD_INNER, pady=(PAD_INNER, 2))
        ttk.Label(song_panel,
                  text="A .osz reports on every difficulty inside it whose name is a recognized "
                       "tier (Easy/Normal/Hard/Insane/Expert).", style="Hint.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=PAD_INNER)
        self.song_var = tk.StringVar()
        self._file_row(song_panel, 2, self.song_var, self._browse_song, last=True)
        song_panel.columnconfigure(0, weight=1)

        # --- Compare against section ---
        compare_panel = self._panel(outer, "Compare against")
        ttk.Label(compare_panel,
                  text="Optional. Pick a specific beatmap to compare against, check Backstabber, "
                       "or leave both off to generate the report with no comparison at all.",
                  style="Hint.TLabel").grid(row=0, column=0, columnspan=2, sticky="w",
                                             padx=PAD_INNER, pady=(PAD_INNER, 6))
        self.compare_backstabber_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(compare_panel, text="Compare against Backstabber (bundled example)",
                         variable=self.compare_backstabber_var).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=PAD_INNER, pady=(0, 8))
        ttk.Label(compare_panel, text="Or compare against a specific .osu/.osz",
                  style="Heading.TLabel").grid(row=2, column=0, columnspan=2, sticky="w",
                                                 padx=PAD_INNER, pady=(0, 2))
        self.compare_song_var = tk.StringVar()
        self._file_row(compare_panel, 3, self.compare_song_var, self._browse_compare_song, last=True)
        compare_panel.columnconfigure(0, weight=1)

        # --- Output ---
        out_panel = self._panel(outer, "Output")
        ttk.Label(out_panel, text="Output PDF (optional)", style="Heading.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=PAD_INNER, pady=(PAD_INNER, 2))
        ttk.Label(out_panel, text="Leave blank to write to output/<beatmap name>/report.pdf.",
                  style="Hint.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", padx=PAD_INNER)
        self.output_var = tk.StringVar()
        self._file_row(out_panel, 2, self.output_var, self._browse_output, last=False)
        self.auto_open_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(out_panel, text="Open the report when done", variable=self.auto_open_var).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=PAD_INNER, pady=(4, PAD_INNER))
        out_panel.columnconfigure(0, weight=1)

        # --- Generate button + log ---
        self.generate_button = ttk.Button(outer, text="Generate Report", command=self._on_generate)
        self.generate_button.pack(fill="x", pady=(0, 8))

        log_frame = ttk.Frame(outer, style="Panel.TFrame")
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, height=8, state="disabled", wrap="word",
                                 background=BG_ENTRY, foreground="#e6e6ec", insertbackground="#e6e6ec",
                                 font=FONT_MONO, relief="flat", padx=10, pady=10)
        self.log_text.pack(fill="both", expand=True)

        self.win.after(100, self._drain_log_queue)

    # --- Layout helpers (same patterns as App) ---

    def _panel(self, parent: tk.Widget, title: str) -> ttk.Labelframe:
        # Unlike App's own _panel, this window has no scrollable canvas
        # behind it -- expand=True here would let these fixed-size panels
        # eat all the Toplevel's height, pushing the Generate button and
        # log below the visible window instead of just taking their
        # natural height and leaving the rest to the log frame below.
        panel = ttk.Labelframe(parent, text=title)
        panel.pack(fill="x", pady=(0, PAD_OUTER))
        return panel

    def _file_row(self, parent: tk.Widget, row: int, var: tk.StringVar, browse_cmd, last: bool = False) -> None:
        row_frame = ttk.Frame(parent, style="Panel.TFrame")
        row_frame.grid(row=row, column=0, columnspan=2, sticky="ew",
                        padx=PAD_INNER, pady=(4, PAD_INNER if last else 4))
        row_frame.columnconfigure(0, weight=1)
        ttk.Entry(row_frame, textvariable=var).grid(row=0, column=0, sticky="ew")
        ttk.Button(row_frame, text="Browse...", style="Secondary.TButton",
                   command=browse_cmd).grid(row=0, column=1, padx=(8, 0))

    # --- File pickers ---

    def _browse_song(self) -> None:
        path = filedialog.askopenfilename(title="Choose a beatmap to report on",
                                           filetypes=[("osu! beatmaps", "*.osu *.osz"), ("All files", "*.*")])
        if path:
            self.song_var.set(path)

    def _browse_compare_song(self) -> None:
        path = filedialog.askopenfilename(title="Choose a beatmap to compare against",
                                           filetypes=[("osu! beatmaps", "*.osu *.osz"), ("All files", "*.*")])
        if path:
            self.compare_song_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Save report as", defaultextension=".pdf",
                                             filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if path:
            self.output_var.set(path)

    # --- Log ---

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                self._append_log(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        self.win.after(100, self._drain_log_queue)

    # --- Generation ---

    def _on_generate(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        song = self.song_var.get().strip()
        if not song:
            messagebox.showerror("Missing beatmap", "Choose a .osu or .osz file first.")
            return
        if not os.path.isfile(song):
            messagebox.showerror("File not found", f"Can't find:\n{song}")
            return

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.generate_button.configure(state="disabled", text="Generating...")

        self.worker = threading.Thread(target=self._run, args=(song,), daemon=True)
        self.worker.start()
        self.win.after(200, self._poll_worker)

    def _resolve_compare_stats(self, tier: str) -> BeatmapStats | None:
        """Which reference stats (if any) to plot the report against, in
        priority order: an explicitly chosen song beats the Backstabber
        checkbox, and neither being set just means no comparison -- the
        report still generates fine with only the reported-on map's own
        stats (render_report treats a None ref as "no data" per field)."""
        custom = self.compare_song_var.get().strip()
        if custom:
            if not os.path.isfile(custom):
                raise RuntimeError(f"Compare-against file not found: {custom}")
            if custom.lower().endswith(".osz"):
                osu_paths = extract_osz(custom)
                match = next((p for p in osu_paths if guess_tier(p) == tier), None)
                match = match or (osu_paths[0] if osu_paths else None)
                if match is None:
                    raise RuntimeError(f"No .osu difficulty found inside {custom}")
                return compute_stats(match)
            return compute_stats(custom)
        if self.compare_backstabber_var.get():
            ref_path = beatmap_report.find_reference_osu(tier)
            return compute_stats(ref_path) if ref_path else None
        return None

    def _run(self, song: str) -> None:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        redirector = TextRedirector(self.log_queue)
        sys.stdout = redirector
        sys.stderr = redirector
        self.result_error = None
        self.result_path = None
        try:
            output_pdf = self.output_var.get().strip() or beatmap_report.default_report_path(song)
            label = os.path.splitext(os.path.basename(song))[0]
            # Names whatever _resolve_compare_stats actually compares
            # against, for the summary-table column header and cover page
            # -- only shown at all if some stage ends up with a reference
            # (render_report itself decides that, from tier_stats).
            custom = self.compare_song_var.get().strip()
            ref_label = os.path.splitext(os.path.basename(custom))[0] if custom else "Backstabber"

            if song.lower().endswith(".osz"):
                osu_paths = extract_osz(song)
                tier_paths: dict[str, str] = {}
                for path in osu_paths:
                    tier = guess_tier(path)
                    if tier is not None and tier not in tier_paths:  # first match wins over a guest diff
                        tier_paths[tier] = path
                if not tier_paths:
                    raise RuntimeError(f"No recognizable difficulty names (Easy/Normal/Hard/Insane/Expert) "
                                        f"found inside {song}.")
                # Built directly (rather than via beatmap_report.build_report_for_tiers, which always
                # compares against Backstabber) so each tier gets this window's own comparison choice.
                tier_stats = {tier: (compute_stats(path), self._resolve_compare_stats(tier))
                               for tier, path in tier_paths.items()}
                beatmap_report.render_report(tier_stats, output_pdf, gen_label=label, ref_label=ref_label)
            else:
                tier = guess_tier(song) or "insane"
                gen_stats = compute_stats(song)
                ref_stats = self._resolve_compare_stats(tier)
                beatmap_report.render_report({tier: (gen_stats, ref_stats)}, output_pdf,
                                              gen_label=label, ref_label=ref_label)

            self.log_queue.put(f"Wrote statistics report: {output_pdf}\n")
            self.result_path = output_pdf
        except Exception as e:  # noqa: BLE001 - surfaced to the user as-is
            self.result_error = e
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def _poll_worker(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.win.after(200, self._poll_worker)
            return
        self.generate_button.configure(state="normal", text="Generate Report")
        if self.result_error is not None:
            messagebox.showerror("Report failed", str(self.result_error))
            return
        if self.auto_open_var.get() and self.result_path:
            _open_path(self.result_path)


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
