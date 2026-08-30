#!/usr/bin/env python3
"""
Desktop GUI for the Base Map v2 pathway — generate_base_beatmap_v2.py plus
add_sliders_v2.py, in one window.

This is a deliberately separate app from gui.py, not a tab or checkbox
inside it: the v2 pathway is an independent, standalone alternative to the
main pipeline (see generate_base_beatmap_v2.py's own module docstring),
and it exposes a completely different, much smaller set of knobs — no
Difficulties spread, no streams/stacks, no statistics report, just three
sliders:

  * Intensity       -- generate_base_beatmap_v2.py's --intensity: how much
                        of the song ends up on half/quarter-beat spacing
                        (and measure-anchored eighth-note bursts) versus
                        staying whole-beat dominant.
  * Slider length    -- add_sliders_v2.py's --slider-length-bias: fewer/
                         longer sliders vs. more/shorter ones, when
                         circles get merged into sliders.
  * Slider curviness -- forwarded to add_sliders_v2.py's --curviness (the
                         same knob apply_style.py's own --curviness is,
                         since add_sliders_v2.py re-runs apply_style.py to
                         do the actual positioning).

Reuses gui.py's dark theme (_configure_style) and color constants directly
so the two apps look like part of the same project, without duplicating
~150 lines of ttk style setup.

Usage:
    python3 gui_v2.py
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import add_sliders_v2
import generate_base_beatmap_v2
from build_osz import build_osz
from gui import BG, BG_ENTRY, FONT_MONO, PAD_INNER, PAD_OUTER, TextRedirector, _configure_style, _open_path

# Version names for the three stages, mirroring the main pipeline's own
# Base/Variety/Styled naming.
CIRCLES_VERSION = "Auto Base v2 (Circles)"
SLIDERS_VERSION = "Auto Base v2 (Sliders)"
STYLED_VERSION = "Auto Base v2 (Styled)"


@dataclass
class SliderParam:
    flag: str
    label: str
    description: str
    lo: float
    hi: float
    default: float


SLIDER_PARAMS = [
    SliderParam("--intensity", "Intensity",
                "How much of the song ends up on faster subdivisions. Min 0 (whole-beat "
                "dominant throughout), max 1 (much more of it on half/quarter-beat spacing, "
                "including denser measure-anchored bursts). Default 0.5.",
                0.0, 1.0, 0.5),
    SliderParam("--slider-length-bias", "Slider length",
                "How long a merged slider tends to run. Min 0 (more, shorter/choppier "
                "sliders), max 1 (fewer, longer sliders). Default 0.5.",
                0.0, 1.0, 0.5),
    SliderParam("--curviness", "Slider curviness",
                "How curved slider paths look, from mostly straight lines to pronounced "
                "arcs. Min 0 (straight), max 1 (very curved). Default 0.5.",
                0.0, 1.0, 0.5),
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

        # --- Style section (the only three knobs this pathway has) ---
        style_panel = self._panel(form, "Style")
        for i, p in enumerate(SLIDER_PARAMS):
            self._slider_row(style_panel, i, p, first=(i == 0))
        style_panel.columnconfigure(0, weight=1)

        # --- Options ---
        opt_panel = self._panel(form, "Options")
        self.osz_var = tk.BooleanVar(value=True)
        self.auto_open_var = tk.BooleanVar(value=True)
        self.keep_intermediate_var = tk.BooleanVar(value=False)
        options = (
            (self.osz_var, "Package as .osz (ready to import into osu!)"),
            (self.auto_open_var, "Open the finished map when done"),
            (self.keep_intermediate_var, "Keep intermediate stages too (Circles and Sliders, "
                                          "alongside the final Styled map)"),
        )
        for i, (var, label) in enumerate(options):
            ttk.Checkbutton(opt_panel, text=label, variable=var).grid(
                row=i, column=0, sticky="w", padx=PAD_INNER,
                pady=(PAD_INNER if i == 0 else 6, PAD_INNER if i == len(options) - 1 else 0))
        opt_panel.columnconfigure(0, weight=1)

        # --- Generate button + log ---
        bottom = ttk.Frame(root)
        bottom.pack(fill="both", padx=PAD_OUTER, pady=(6, PAD_OUTER))
        self.generate_button = ttk.Button(bottom, text="Generate", command=self._on_generate)
        self.generate_button.pack(fill="x", pady=(0, 8))

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
        scale = ttk.Scale(parent, from_=p.lo, to=p.hi, orient="horizontal", variable=var, command=on_change)
        scale.grid(row=base_row + 2, column=0, columnspan=2, sticky="ew", padx=PAD_INNER, pady=(6, 4))

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

            circles_path = os.path.join(outdir, f"{title} [{CIRCLES_VERSION}].osu")
            sliders_path = os.path.join(outdir, f"{title} [{SLIDERS_VERSION}].osu")
            styled_path = os.path.join(outdir, f"{title} [{STYLED_VERSION}].osu")

            base_argv = [audio, "--output", circles_path, "--title", title,
                         "--artist", self.artist_var.get().strip() or "Unknown Artist",
                         "--creator", self.creator_var.get().strip() or "auto-beatmapper",
                         "--version", CIRCLES_VERSION,
                         "--intensity", f"{self.slider_vars['--intensity'].get():.3f}"]
            sys.argv = ["generate_base_beatmap_v2.py"] + base_argv
            generate_base_beatmap_v2.main()

            sliders_argv = [circles_path, audio, "--output", styled_path, "--version", STYLED_VERSION,
                             "--slider-length-bias", f"{self.slider_vars['--slider-length-bias'].get():.3f}",
                             "--curviness", f"{self.slider_vars['--curviness'].get():.3f}"]
            # The merged-but-unstyled "Sliders" stage is only ever worth
            # writing out when the user actually wants to inspect it --
            # otherwise it's exactly what add_sliders_v2.py already treats
            # it as: a hidden working file, cleaned up on its own once
            # apply_style.py is done reading it.
            if keep_intermediate:
                sliders_argv += ["--merged-output", sliders_path, "--merged-version", SLIDERS_VERSION]
            sys.argv = ["add_sliders_v2.py"] + sliders_argv
            add_sliders_v2.main()

            # Circles/Sliders are worth keeping around at all only when the
            # user actually checked the box -- otherwise they're the same
            # kind of internal working file main.py's own Base/Variety are
            # without --keep-intermediate-files, cleaned up once the thing
            # that's actually meant to be played (Styled) exists.
            all_paths = [circles_path, sliders_path, styled_path] if keep_intermediate else [styled_path]
            if not keep_intermediate:
                os.remove(circles_path)

            if self.osz_var.get():
                osz_path = os.path.join(outdir, f"{title} (Base v2).osz")
                build_osz(all_paths, audio, osz_path)
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


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
