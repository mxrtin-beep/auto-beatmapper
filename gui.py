#!/usr/bin/env python3
"""
Desktop GUI for the pipeline — the same thing `main.py` does from the
shell, but with a file picker for the song, a form for every main.py
argument, and a button instead of a command line.

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
from tkinter import filedialog, messagebox, ttk

import main as pipeline_main


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


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Auto Beatmapper")
        root.geometry("640x720")

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.result_error: Exception | None = None
        self.result_paths: list[str] = []

        pad = {"padx": 8, "pady": 4}
        frame = ttk.Frame(root)
        frame.pack(fill="both", expand=True, padx=8, pady=8)
        frame.columnconfigure(1, weight=1)

        row = 0

        # --- Song file ---
        ttk.Label(frame, text="Song (MP3)").grid(row=row, column=0, sticky="w", **pad)
        self.audio_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.audio_var).grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(frame, text="Browse...", command=self._browse_audio).grid(row=row, column=2, **pad)
        row += 1

        # --- Output directory ---
        ttk.Label(frame, text="Output folder").grid(row=row, column=0, sticky="w", **pad)
        self.outdir_var = tk.StringVar(value="output")
        ttk.Entry(frame, textvariable=self.outdir_var).grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(frame, text="Browse...", command=self._browse_outdir).grid(row=row, column=2, **pad)
        row += 1

        # --- Metadata ---
        self.title_var = tk.StringVar()
        self.artist_var = tk.StringVar(value="Unknown Artist")
        self.creator_var = tk.StringVar(value="auto-beatmapper")
        for label, var in (("Title (optional)", self.title_var),
                            ("Artist", self.artist_var),
                            ("Creator", self.creator_var)):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", **pad)
            ttk.Entry(frame, textvariable=var).grid(row=row, column=1, columnspan=2, sticky="ew", **pad)
            row += 1

        ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

        # --- Style knobs (all optional -> forwarded only if the user changes them) ---
        self.spacing_var = tk.StringVar()
        self.curviness_var = tk.StringVar()
        self.stack_probability_var = tk.StringVar()
        self.angle_jitter_var = tk.StringVar()
        self.temperature_var = tk.StringVar()
        for label, var, hint in (
            ("--spacing", self.spacing_var, "jump distance multiplier (default 1.3)"),
            ("--curviness", self.curviness_var, "0-1 slider curviness (default 0.5)"),
            ("--stack-probability", self.stack_probability_var, "0-1 stack vs. line mix (default 0.5)"),
            ("--angle-jitter", self.angle_jitter_var, "degrees (default from --temperature)"),
            ("--temperature", self.temperature_var, "0-1 creative vs. structured (default 0.5)"),
        ):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", **pad)
            ttk.Entry(frame, textvariable=var, width=10).grid(row=row, column=1, sticky="w", **pad)
            ttk.Label(frame, text=hint, foreground="#666").grid(row=row, column=2, sticky="w", **pad)
            row += 1

        ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

        # --- BPM / offset overrides ---
        self.bpm_var = tk.StringVar()
        self.offset_var = tk.StringVar()
        ttk.Label(frame, text="--bpm (optional)").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.bpm_var, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frame, text="manual BPM instead of auto-detect", foreground="#666").grid(row=row, column=2, sticky="w", **pad)
        row += 1
        ttk.Label(frame, text="--offset ms (optional)").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.offset_var, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frame, text="manual offset instead of auto-detect", foreground="#666").grid(row=row, column=2, sticky="w", **pad)
        row += 1

        self.seed_var = tk.StringVar()
        ttk.Label(frame, text="--seed (optional)").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.seed_var, width=10).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(frame, text="fixed seed to reproduce a run", foreground="#666").grid(row=row, column=2, sticky="w", **pad)
        row += 1

        ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1

        # --- Checkboxes ---
        self.spread_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Generate full spread (Easy/Normal/Hard/Insane) — uncheck for Insane only",
                        variable=self.spread_var).grid(row=row, column=0, columnspan=3, sticky="w", **pad)
        row += 1

        self.osz_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Package as .osz (ready to import into osu!)",
                        variable=self.osz_var).grid(row=row, column=0, columnspan=3, sticky="w", **pad)
        row += 1

        self.keep_osu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Keep loose .osu files after building the .osz",
                        variable=self.keep_osu_var).grid(row=row, column=0, columnspan=3, sticky="w", **pad)
        row += 1

        self.auto_open_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Open the finished map when done",
                        variable=self.auto_open_var).grid(row=row, column=0, columnspan=3, sticky="w", **pad)
        row += 1

        # --- Generate button ---
        self.generate_button = ttk.Button(frame, text="Generate", command=self._on_generate)
        self.generate_button.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        row += 1

        # --- Log output ---
        self.log_text = tk.Text(frame, height=16, state="disabled", wrap="word")
        self.log_text.grid(row=row, column=0, columnspan=3, sticky="nsew", **pad)
        frame.rowconfigure(row, weight=1)

        self.root.after(100, self._drain_log_queue)

    # --- UI helpers ---

    def _browse_audio(self) -> None:
        path = filedialog.askopenfilename(title="Choose a song",
                                           filetypes=[("Audio files", "*.mp3 *.wav *.ogg"), ("All files", "*.*")])
        if path:
            self.audio_var.set(path)

    def _browse_outdir(self) -> None:
        path = filedialog.askdirectory(title="Choose an output folder")
        if path:
            self.outdir_var.set(path)

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

        argv = [audio, "--outdir", self.outdir_var.get().strip() or "output",
                "--artist", self.artist_var.get().strip() or "Unknown Artist",
                "--creator", self.creator_var.get().strip() or "auto-beatmapper"]

        if self.title_var.get().strip():
            argv += ["--title", self.title_var.get().strip()]

        for flag, var in (("--spacing", self.spacing_var), ("--curviness", self.curviness_var),
                           ("--stack-probability", self.stack_probability_var),
                           ("--angle-jitter", self.angle_jitter_var),
                           ("--temperature", self.temperature_var),
                           ("--bpm", self.bpm_var), ("--offset", self.offset_var),
                           ("--seed", self.seed_var)):
            value = var.get().strip()
            if value:
                try:
                    float(value)
                except ValueError:
                    messagebox.showerror("Invalid value", f"{flag} must be a number, got: {value!r}")
                    return None
                argv += [flag, value]

        if not self.spread_var.get():
            argv.append("--no-spread")
        if self.osz_var.get():
            argv.append("--osz")
        if self.keep_osu_var.get():
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
            osz_path = os.path.join(outdir, f"{title}.osz")
            if os.path.isfile(osz_path):
                self.result_paths = [osz_path]
            else:
                self.result_paths = [os.path.join(outdir, f"{title} [{tier}].osu")
                                      for tier in ("Insane", "Hard", "Normal", "Easy")
                                      if os.path.isfile(os.path.join(outdir, f"{title} [{tier}].osu"))]
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
