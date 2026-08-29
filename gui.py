#!/usr/bin/env python3
"""
Desktop GUI for the pipeline — the same thing `main.py` does from the
shell, but with a file picker for the song, a form for every main.py
argument (in plain language, not `--flag` form), and a button instead of
a command line.

Requires no extra dependency beyond what the pipeline itself already
needs — Tkinter ships with the Python standard library on Windows and
macOS installers. On Linux it's usually a separate distro package (e.g.
`sudo apt install python3-tk` on Debian/Ubuntu, `sudo dnf install
python3-tkinter` on Fedora) since distros split it out from the core
interpreter.

Usage:
    python3 gui.py
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import zipfile
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import beatmap_stats
import main as pipeline_main

# --- Color palette (a dark theme instead of Tk's stock gray) ---
BG = "#1c1e26"
BG_PANEL = "#242732"
BG_ENTRY = "#2c2f3c"
FG = "#e6e6ec"
FG_MUTED = "#9498a8"
ACCENT = "#7c9cff"
ACCENT_ACTIVE = "#9db2ff"
BORDER = "#3a3e4d"

FONT_LABEL = ("Segoe UI", 10, "bold")
FONT_HINT = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 9)

PAD_OUTER = 18   # panel-to-window and section-to-section spacing
PAD_INNER = 16   # content-to-panel-border spacing (labels sat flush against the border before)

DIFFICULTIES = ("Easy", "Normal", "Hard", "Insane")


@dataclass
class SliderParam:
    """A bounded numeric style knob, shown as a slider with its live
    value, a plain-language label, and a phrase-long description of what
    it does and its range."""
    flag: str
    label: str
    description: str
    lo: float
    hi: float
    default: float


SLIDER_PARAMS = [
    SliderParam("--spacing", "Jump distance",
                "How far apart notes are placed for a given time gap between them. "
                "Min 0.5 (tight, close together), max 2.5 (wide, dramatic jumps). Default 1.3.",
                0.5, 2.5, 1.3),
    SliderParam("--curviness", "Slider curviness",
                "How curved slider paths look, from mostly straight lines to pronounced "
                "arcs. Min 0 (straight), max 1 (very curved). Default 0.5.",
                0.0, 1.0, 0.5),
    SliderParam("--stream-frequency", "Stream frequency",
                "How often a fast run of notes (on quarter or eighth beats) becomes a "
                "deliberate stream — either piled into one stacked spot or spread along a "
                "straight line — instead of just following the normal flow like any other "
                "note. Min 0 (never a stream; a run is still capped at 3 notes in a row, but "
                "they're never piled up or locked onto one line), max 1 (always a stream). "
                "Default 0.5.",
                0.0, 1.0, 0.5),
    SliderParam("--temperature", "Creativity",
                "How much variation the whole map has — turn angles, slider curviness, how "
                "far the pattern wanders around the screen, and how many times the jump "
                "distance shifts over the course of the song. Min 0 (tight and repetitive, "
                "spacing never changes), max 1 (loose and varied). Default 0.5.",
                0.0, 1.0, 0.5),
]


@dataclass
class EntryParam:
    """An optional, unbounded (or integer) numeric override, shown as a
    plain text field left blank to use the pipeline's own default."""
    flag: str
    label: str
    description: str


STYLE_ENTRY_PARAMS = [
    EntryParam("--angle-jitter", "Slider angles",
               "The map repeats a handful of fixed turn shapes (e.g. always turning 90 "
               "degrees traces a square, 60 traces a hexagon) so patterns stay recognizable. "
               "This adds a small random +/- degrees on top of every one of those turns, so "
               "the shape isn't perfectly identical each time it repeats. Since the shapes "
               "themselves turn 60-144 degrees, the default (roughly 1-10) is a subtle, "
               "hand-drawn-feeling wobble; try 20-45 for an obviously looser, less crisp "
               "shape, or higher still to mostly break the shape apart. Leave blank to "
               "derive it from Creativity automatically."),
]

TIMING_PARAMS = [
    EntryParam("--bpm", "BPM override",
               "Force a specific tempo instead of detecting it from the song automatically. "
               "Leave blank to auto-detect."),
    EntryParam("--offset", "Beat offset override (ms)",
               "Force the time of the very first beat, in milliseconds, instead of detecting "
               "it automatically. Leave blank to auto-detect."),
]

SEED_PARAM = EntryParam("--seed", "Random seed",
                         "A fixed whole number so re-running with the same song and settings "
                         "produces the exact same map. Leave blank for a different result every time.")


class TextRedirector:
    """A writable stream that pushes text onto a thread-safe queue instead
    of printing directly — the pipeline runs on a background thread (so the
    window stays responsive), but Tkinter widgets may only be touched from
    the main thread, so the log box drains this queue on a timer instead."""

    def __init__(self, line_queue: "queue.Queue[str]") -> None:
        self.line_queue = line_queue

    def write(self, text: str) -> None:
        if text:
            self.line_queue.put(text)

    def flush(self) -> None:
        pass


def _configure_style(root: tk.Tk) -> None:
    root.configure(bg=BG)
    style = ttk.Style(root)
    # 'clam' is the only stock theme that actually honors custom colors on
    # every widget below — the default themes ignore most background/
    # foreground overrides on Windows and macOS.
    style.theme_use("clam")

    style.configure(".", background=BG, foreground=FG, font=FONT_HINT)
    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=BG_PANEL)
    style.configure("TLabel", background=BG, foreground=FG, font=FONT_HINT)
    style.configure("Panel.TLabel", background=BG_PANEL, foreground=FG)
    style.configure("Heading.TLabel", background=BG_PANEL, foreground=FG, font=FONT_LABEL)
    style.configure("Hint.TLabel", background=BG_PANEL, foreground=FG_MUTED,
                     font=FONT_HINT, wraplength=460, justify="left")
    style.configure("Value.TLabel", background=BG_PANEL, foreground=ACCENT, font=FONT_LABEL)

    style.configure("TLabelframe", background=BG_PANEL, bordercolor=BORDER,
                     relief="solid", borderwidth=1, labelmargins=(10, 6, 10, 10))
    # `padding` here is what actually gives the section title breathing
    # room from the frame's border — the border is drawn snug against
    # whatever the label widget's own bounding box is, so without this the
    # text sits with its first letter touching the corner.
    style.configure("TLabelframe.Label", background=BG_PANEL, foreground=ACCENT, font=FONT_LABEL,
                     padding=(4, 6, 4, 6))

    style.configure("TEntry", fieldbackground=BG_ENTRY, foreground=FG,
                     bordercolor=BORDER, insertcolor=FG, padding=6)
    style.map("TEntry", fieldbackground=[("disabled", BG_PANEL)])

    style.configure("TButton", background=ACCENT, foreground="#101218",
                     font=FONT_LABEL, padding=8, borderwidth=0)
    style.map("TButton", background=[("active", ACCENT_ACTIVE), ("disabled", BORDER)],
              foreground=[("disabled", FG_MUTED)])

    style.configure("Secondary.TButton", background=BG_ENTRY, foreground=FG,
                     font=FONT_HINT, padding=6, borderwidth=1)
    style.map("Secondary.TButton", background=[("active", BORDER)])

    # A plain ttk Checkbutton's "indicator" (the little box) renders as a
    # bare glyph that reads as an X rather than a checkbox on several
    # platforms unless these are set explicitly — this gives it an actual
    # square that fills in the accent color when checked.
    style.configure("TCheckbutton", background=BG_PANEL, foreground=FG, font=FONT_HINT,
                     indicatorsize=16, indicatormargin=8, indicatorrelief="flat",
                     indicatordiameter=14)
    style.map("TCheckbutton", background=[("active", BG_PANEL)],
              indicatorbackground=[("selected", ACCENT), ("!selected", BG_ENTRY)],
              indicatorforeground=[("selected", "#101218"), ("!selected", BG_ENTRY)])

    style.configure("TScale", background=BG_PANEL, troughcolor=BG_ENTRY)
    style.configure("TSeparator", background=BORDER)
    style.configure("Vertical.TScrollbar", background=BG_ENTRY, troughcolor=BG,
                     bordercolor=BG, arrowcolor=FG_MUTED)


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Auto Beatmapper")
        root.geometry("700x860")
        root.minsize(560, 500)
        _configure_style(root)

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.result_error: Exception | None = None
        self.result_paths: list[str] = []
        self.slider_vars: dict[str, tk.DoubleVar] = {}
        self.entry_vars: dict[str, tk.StringVar] = {}
        self.difficulty_vars: dict[str, tk.BooleanVar] = {}

        # --- Scrollable form area (the form is long — sections stack) ---
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

        canvas.bind_all("<MouseWheel>", on_mousewheel)   # Windows / macOS
        canvas.bind_all("<Button-4>", on_mousewheel)      # Linux scroll up
        canvas.bind_all("<Button-5>", on_mousewheel)      # Linux scroll down

        # --- Song section ---
        song_panel = self._panel(form, "Song")
        self._section_heading(song_panel, 0, "Song file",
                               "The MP3 (or WAV/OGG) to build a beatmap from.")
        self.audio_var = tk.StringVar()
        self._file_row(song_panel, 2, self.audio_var, self._browse_audio)

        self._section_heading(song_panel, 3, "Output folder",
                               "Where the finished files are written. Every song lands in "
                               "the same flat folder.", first=False)
        self.outdir_var = tk.StringVar(value="output")
        self._file_row(song_panel, 5, self.outdir_var, self._browse_outdir, last=True)
        song_panel.columnconfigure(0, weight=1)

        # --- Song info section ---
        meta_panel = self._panel(form, "Song info")
        self.title_var = tk.StringVar()
        self.artist_var = tk.StringVar(value="Unknown Artist")
        self.creator_var = tk.StringVar(value="auto-beatmapper")
        meta_fields = (
            ("Title", "Song title. Leave blank to use the audio file's name.", self.title_var),
            ("Artist", "Who performed the song.", self.artist_var),
            ("Creator", "Your mapper name, credited in the beatmap.", self.creator_var),
        )
        for i, (label, desc, var) in enumerate(meta_fields):
            self._labeled_entry(meta_panel, i, label, desc, var,
                                 first=(i == 0), last=(i == len(meta_fields) - 1))
        meta_panel.columnconfigure(1, weight=1)

        # --- Style section (sliders, plus the one freeform angle field) ---
        style_panel = self._panel(form, "Style")
        for i, p in enumerate(SLIDER_PARAMS):
            self._slider_row(style_panel, i, p, first=(i == 0))
        # An arbitrarily large row offset for whatever comes after the
        # sliders — grid rows with no widget in them take up no space, so
        # this only needs to sort after every row _slider_row used (it
        # doesn't need to be contiguous with them).
        entry_row_base = 1000
        for j, p in enumerate(STYLE_ENTRY_PARAMS):
            var = tk.StringVar()
            self.entry_vars[p.flag] = var
            self._labeled_entry(style_panel, entry_row_base + j, p.label, p.description, var,
                                 first=False, last=True)
        style_panel.columnconfigure(0, weight=1)

        # --- Timing section ---
        timing_panel = self._panel(form, "Timing")
        for i, p in enumerate(TIMING_PARAMS):
            var = tk.StringVar()
            self.entry_vars[p.flag] = var
            self._labeled_entry(timing_panel, i, p.label, p.description, var,
                                 first=(i == 0), last=(i == len(TIMING_PARAMS) - 1))
        timing_panel.columnconfigure(1, weight=1)

        # --- Seed section ---
        seed_panel = self._panel(form, "Seed")
        seed_var = tk.StringVar()
        self.entry_vars[SEED_PARAM.flag] = seed_var
        self._labeled_entry(seed_panel, 0, SEED_PARAM.label, SEED_PARAM.description, seed_var,
                             first=True, last=True)
        seed_panel.columnconfigure(1, weight=1)

        # --- Difficulties section ---
        diff_panel = self._panel(form, "Difficulties")
        ttk.Label(diff_panel, text="All four are always generated; unchecked ones are simply "
                                    "left out of the result.", style="Hint.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", padx=PAD_INNER, pady=(PAD_INNER, 6))
        for i, tier in enumerate(DIFFICULTIES):
            var = tk.BooleanVar(value=True)
            self.difficulty_vars[tier] = var
            ttk.Checkbutton(diff_panel, text=tier, variable=var).grid(
                row=1, column=i, sticky="w", padx=(PAD_INNER if i == 0 else 8, 8),
                pady=(0, PAD_INNER))
        diff_panel.columnconfigure(len(DIFFICULTIES), weight=1)

        # --- Options section (checkboxes only, no explanatory text) ---
        opt_panel = self._panel(form, "Options")
        self.osz_var = tk.BooleanVar(value=True)
        self.keep_osu_var = tk.BooleanVar(value=False)
        self.auto_open_var = tk.BooleanVar(value=True)
        options = (
            (self.osz_var, "Package as .osz (ready to import into osu!)"),
            (self.keep_osu_var, "Keep loose .osu files too"),
            (self.auto_open_var, "Open the finished map when done"),
        )
        for i, (var, label) in enumerate(options):
            ttk.Checkbutton(opt_panel, text=label, variable=var).grid(
                row=i, column=0, sticky="w", padx=PAD_INNER,
                pady=(PAD_INNER if i == 0 else 6, PAD_INNER if i == len(options) - 1 else 0))
        opt_panel.columnconfigure(0, weight=1)

        # --- Generate button + log (outside the scroll area, always visible) ---
        bottom = ttk.Frame(root)
        bottom.pack(fill="both", padx=PAD_OUTER, pady=(6, PAD_OUTER))
        self.generate_button = ttk.Button(bottom, text="Generate", command=self._on_generate)
        self.generate_button.pack(fill="x", pady=(0, 8))

        log_frame = ttk.Frame(bottom, style="Panel.TFrame")
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, height=10, state="disabled", wrap="word",
                                 background=BG_ENTRY, foreground=FG, insertbackground=FG,
                                 font=FONT_MONO, relief="flat", padx=10, pady=10)
        self.log_text.pack(fill="both", expand=True)

        self.root.after(100, self._drain_log_queue)

    # --- Layout helpers ---

    def _panel(self, parent: tk.Widget, title: str) -> ttk.Labelframe:
        panel = ttk.Labelframe(parent, text=title)
        panel.pack(fill="x", expand=True, padx=PAD_OUTER, pady=(0, PAD_OUTER))
        return panel

    def _section_heading(self, parent: tk.Widget, row: int, label: str, desc: str,
                          first: bool = True) -> None:
        ttk.Label(parent, text=label, style="Heading.TLabel").grid(
            row=row, column=0, columnspan=2, sticky="w",
            padx=PAD_INNER, pady=(PAD_INNER if first else 14, 2))
        ttk.Label(parent, text=desc, style="Hint.TLabel").grid(
            row=row + 1, column=0, columnspan=2, sticky="w", padx=PAD_INNER)

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
        var = tk.DoubleVar(value=p.default)
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
        scale = ttk.Scale(parent, from_=p.lo, to=p.hi, orient="horizontal",
                           variable=var, command=on_change)
        scale.grid(row=base_row + 2, column=0, columnspan=2, sticky="ew", padx=PAD_INNER, pady=(6, 0))
        range_frame = ttk.Frame(parent, style="Panel.TFrame")
        range_frame.grid(row=base_row + 3, column=0, columnspan=2, sticky="ew", padx=PAD_INNER, pady=(0, 4))
        range_frame.columnconfigure(1, weight=1)
        ttk.Label(range_frame, text=f"min {p.lo:g}", style="Hint.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(range_frame, text=f"max {p.hi:g}", style="Hint.TLabel").grid(row=0, column=2, sticky="e")

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

    def _build_argv(self) -> list[str] | None:
        audio = self.audio_var.get().strip()
        if not audio:
            messagebox.showerror("Missing song", "Choose an MP3 file first.")
            return None
        if not os.path.isfile(audio):
            messagebox.showerror("File not found", f"Can't find:\n{audio}")
            return None
        if not any(var.get() for var in self.difficulty_vars.values()):
            messagebox.showerror("No difficulties selected", "Check at least one difficulty to generate.")
            return None

        argv = [audio, "--outdir", self.outdir_var.get().strip() or "output",
                "--artist", self.artist_var.get().strip() or "Unknown Artist",
                "--creator", self.creator_var.get().strip() or "auto-beatmapper"]

        if self.title_var.get().strip():
            argv += ["--title", self.title_var.get().strip()]

        for p in SLIDER_PARAMS:
            argv += [p.flag, f"{self.slider_vars[p.flag].get():.3f}"]

        for p in (*STYLE_ENTRY_PARAMS, *TIMING_PARAMS, SEED_PARAM):
            value = self.entry_vars[p.flag].get().strip()
            if value:
                try:
                    float(value)
                except ValueError:
                    messagebox.showerror("Invalid value", f"{p.label} must be a number, got: {value!r}")
                    return None
                argv += [p.flag, value]

        # Difficulties are always all generated (--no-spread is never
        # passed) — an unchecked one is deleted afterward instead, since
        # make_easy.py derives each easier tier from the one above it, so
        # skipping a tier mid-spread isn't something the pipeline itself
        # supports doing any other way.
        if self.osz_var.get():
            argv.append("--osz")
            # Always keep the loose files at the main.py level, regardless
            # of the user's own choice — _run_pipeline needs the loose
            # Insane .osu to still exist afterward to compute stats on,
            # and honors the user's real preference itself once that's
            # done (see the keep_osu_var check there).
            argv.append("--keep-osu-files")
        elif self.keep_osu_var.get():
            argv.append("--keep-osu-files")
        return argv

    def _on_generate(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        argv = self._build_argv()
        if argv is None:
            return

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.generate_button.configure(state="disabled", text="Generating...")

        self.worker = threading.Thread(target=self._run_pipeline, args=(argv,), daemon=True)
        self.worker.start()
        self.root.after(200, self._poll_worker)

    def _run_pipeline(self, argv: list[str]) -> None:
        old_stdout, old_stderr, old_argv = sys.stdout, sys.stderr, sys.argv
        redirector = TextRedirector(self.log_queue)
        sys.stdout = redirector
        sys.stderr = redirector
        sys.argv = ["main.py"] + argv
        self.result_error = None
        self.result_paths = []
        try:
            title = None
            for i, a in enumerate(argv):
                if a == "--title":
                    title = argv[i + 1]
            if title is None:
                title = os.path.splitext(os.path.basename(argv[0]))[0]
            outdir = "output"
            for i, a in enumerate(argv):
                if a == "--outdir":
                    outdir = argv[i + 1]

            pipeline_main.main()

            insane_path = os.path.join(outdir, f"{title} [Insane].osu")
            if os.path.isfile(insane_path):
                try:
                    stats = beatmap_stats.compute_stats(insane_path)
                    self.log_queue.put("\n" + stats.format() + "\n")
                except Exception as e:  # noqa: BLE001 - stats are a bonus, never fail generation over them
                    self.log_queue.put(f"\n(couldn't compute stats: {e})\n")

            excluded = [t for t, v in self.difficulty_vars.items() if not v.get()]
            osz_path = os.path.join(outdir, f"{title}.osz")
            if excluded:
                if os.path.isfile(osz_path):
                    _remove_difficulties_from_osz(osz_path, title, excluded)
                for tier in excluded:
                    loose_path = os.path.join(outdir, f"{title} [{tier}].osu")
                    if os.path.isfile(loose_path):
                        os.remove(loose_path)

            # main.py was always told --keep-osu-files when packaging (see
            # _build_argv) so the Insane .osu above was guaranteed to still
            # exist to compute stats from — now that that's done, honor the
            # user's actual preference the way main.py itself would have.
            if self.osz_var.get() and not self.keep_osu_var.get() and os.path.isfile(osz_path):
                for tier in DIFFICULTIES:
                    if tier in excluded:
                        continue
                    loose_path = os.path.join(outdir, f"{title} [{tier}].osu")
                    if os.path.isfile(loose_path):
                        os.remove(loose_path)

            if os.path.isfile(osz_path):
                self.result_paths = [osz_path]
            else:
                self.result_paths = [os.path.join(outdir, f"{title} [{tier}].osu")
                                      for tier in ("Insane", "Hard", "Normal", "Easy")
                                      if tier not in excluded
                                      and os.path.isfile(os.path.join(outdir, f"{title} [{tier}].osu"))]
        except SystemExit as e:
            if e.code not in (None, 0):
                self.result_error = RuntimeError(f"Pipeline exited with code {e.code}")
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
        if self.auto_open_var.get() and self.result_paths:
            _open_path(self.result_paths[0])


def _remove_difficulties_from_osz(osz_path: str, title: str, excluded_tiers: list[str]) -> None:
    """Rewrite the .osz without the .osu entries for `excluded_tiers` — a
    .osz is just a zip file, and build_osz.py names each entry by its own
    basename (`{title} [{Tier}].osu`), so this is a plain filter-and-
    recompress rather than anything beatmap-specific. Every other entry
    (the audio file, any images) is carried over untouched."""
    to_remove = {f"{title} [{tier}].osu" for tier in excluded_tiers}
    tmp_path = osz_path + ".tmp"
    with zipfile.ZipFile(osz_path, "r") as zin, zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename in to_remove:
                continue
            zout.writestr(item, zin.read(item.filename))
    os.replace(tmp_path, osz_path)


def _open_path(path: str) -> None:
    """Open a finished file with whatever the OS considers its default
    handler — a .osz with osu! itself (which imports it on open), a loose
    .osu file with whatever's registered for it."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except OSError as e:
        messagebox.showwarning("Couldn't open file", f"Generated {path}, but couldn't open it automatically:\n{e}")


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
