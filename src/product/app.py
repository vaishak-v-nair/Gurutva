"""Gurutva, as a window instead of a command line.

The engine and the CLI were both real, and both unusable by the person this
is actually for. A geoscientist with a laptop and a CSV had to install
Python, obtain a private repository, open a terminal, and type a line with
six flags in it. The realistic number of them who would do that is zero, so
the product did not exist for them.

This is the missing rung. Pick a file, fill three boxes, press one button,
and the report opens in your browser. tkinter only, which ships with Python,
because a tool aimed at people on locked-down work laptops must not need a
package install to start.

Deliberately NOT hidden behind friendly defaults: the noise floor, the prior
amplitude and the correlation length are still typed by the user, because
every one of them is a statement about their site that the software has no
business inventing. The labels explain what each one means in their language;
they do not guess it for them.

Run:  py -3 -m src.product.app
"""

import queue
import sys
import threading
import traceback
import webbrowser
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, ttk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIELDS = {
    "Gravity (mGal)": "gravity",
    "Magnetics (nT)": "mag",
}

HINTS = {
    "noise": ("How repeatable is your instrument?",
              "The spread you get re-reading the same station.\n"
              "Gravity: often 0.02-0.1 mGal.   Magnetics: often 1-5 nT.\n"
              "We will not guess this: a tool that invents a noise floor is\n"
              "inventing its own passing grade."),
    "prior": ("How much does the rock vary here?",
              "Gravity: density contrast in g/cc. A basin against basement is\n"
              "about 0.25. Magnetics: susceptibility in SI, often very small.\n"
              "Take it from your own logs or your own report."),
    "corr": ("Over what distance does it change?",
             "The scale your geology varies on, in metres. A few hundred is\n"
             "usual. Rock is connected; treating each cell as a stranger\n"
             "cannot reproduce the signal a real body makes."),
}


class App:
    def __init__(self, root):
        self.root = root
        root.title("Gurutva - what does your survey actually support?")
        root.minsize(720, 560)
        self.q = queue.Queue()
        self.survey = tk.StringVar(value="")
        self.field = tk.StringVar(value="Gravity (mGal)")
        self.noise = tk.StringVar(value="")
        self.prior = tk.StringVar(value="")
        self.corr = tk.StringVar(value="400")
        self.busy = False
        self._build()
        self.root.after(120, self._drain)

    # ------------------------------------------------------------- layout
    def _build(self):
        pad = dict(padx=16, pady=6)
        head = ttk.Frame(self.root)
        head.pack(fill="x", **pad)
        ttk.Label(head, text="Gurutva",
                  font=("Georgia", 20)).pack(side="left")
        ttk.Label(head, text="  how much of that picture is real?",
                  foreground="#5d6670").pack(side="left", pady=(8, 0))

        box = ttk.LabelFrame(self.root, text=" 1. Your survey ")
        box.pack(fill="x", **pad)
        row = ttk.Frame(box)
        row.pack(fill="x", padx=12, pady=10)
        ttk.Button(row, text="Choose a CSV...",
                   command=self.pick).pack(side="left")
        ttk.Button(row, text="Use the example",
                   command=self.example).pack(side="left", padx=(8, 0))
        self.file_lbl = ttk.Label(row, text="no file chosen",
                                  foreground="#5d6670")
        self.file_lbl.pack(side="left", padx=(12, 0))
        ttk.Label(box, text="Four columns with a header row:  x,y,z,gz\n"
                            "Position in metres (or lon,lat in degrees), "
                            "elevation in metres, then your reading.",
                  foreground="#5d6670", justify="left").pack(
            anchor="w", padx=12, pady=(0, 10))

        box2 = ttk.LabelFrame(self.root, text=" 2. What only you can tell us ")
        box2.pack(fill="both", expand=True, **pad)
        f = ttk.Frame(box2)
        f.pack(fill="x", padx=12, pady=10)
        ttk.Label(f, text="Measurement type").grid(row=0, column=0, sticky="w")
        ttk.Combobox(f, textvariable=self.field, values=list(FIELDS),
                     state="readonly", width=18).grid(row=0, column=1,
                                                      sticky="w", pady=4)
        for i, (key, var) in enumerate((("noise", self.noise),
                                        ("prior", self.prior),
                                        ("corr", self.corr)), start=1):
            title, why = HINTS[key]
            ttk.Label(f, text=title).grid(row=i * 2, column=0, sticky="w",
                                          pady=(10, 0))
            ttk.Entry(f, textvariable=var, width=20).grid(
                row=i * 2, column=1, sticky="w", pady=(10, 0))
            ttk.Label(f, text=why, foreground="#5d6670",
                      justify="left").grid(row=i * 2 + 1, column=0,
                                           columnspan=2, sticky="w")

        bar = ttk.Frame(self.root)
        bar.pack(fill="x", **pad)
        self.go = ttk.Button(bar, text="Run", command=self.run)
        self.go.pack(side="left")
        self.status = ttk.Label(bar, text="", foreground="#5d6670")
        self.status.pack(side="left", padx=12)

        self.log = tk.Text(self.root, height=11, wrap="word",
                           font=("Consolas", 9), relief="flat",
                           background="#f4f6f8")
        self.log.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        self.say("Pick a file, or press 'Use the example' to see it work.")

    # -------------------------------------------------------------- helpers
    def say(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def pick(self):
        p = filedialog.askopenfilename(
            title="Choose your survey", initialdir=str(ROOT / "examples"),
            filetypes=[("Survey data", "*.csv *.txt"), ("All files", "*.*")])
        if p:
            self.survey.set(p)
            self.file_lbl.config(text=Path(p).name, foreground="#12151a")

    def example(self):
        p = ROOT / "examples" / "demo_survey.csv"
        self.survey.set(str(p))
        self.file_lbl.config(text=p.name, foreground="#12151a")
        self.field.set("Gravity (mGal)")
        self.noise.set("0.05")
        self.prior.set("0.10")
        self.corr.set("400")
        self.say("Loaded the example: 120 gravity stations over a buried body.")
        self.say("Press Run.")

    def _number(self, var, name):
        try:
            v = float(var.get())
        except ValueError:
            raise ValueError(f"{name} needs to be a number, not "
                             f"{var.get()!r}.")
        if v <= 0:
            raise ValueError(f"{name} has to be greater than zero.")
        return v

    # ------------------------------------------------------------------ run
    def run(self):
        if self.busy:
            return
        try:
            if not self.survey.get():
                raise ValueError("Choose a survey file first.")
            noise = self._number(self.noise, "The instrument repeatability")
            prior = self._number(self.prior, "The rock variation")
            corr = self._number(self.corr, "The distance scale")
        except ValueError as e:
            self.say("")
            self.say("STOPPED: " + str(e))
            return

        out = Path(self.survey.get()).with_name(
            Path(self.survey.get()).stem + "_gurutva.html")
        args = ["report", "--survey", self.survey.get(),
                "--field", FIELDS[self.field.get()],
                "--noise", str(noise), "--prior-sd", str(prior),
                "--corr-len", str(corr), "--out", str(out)]
        self.busy = True
        self.go.config(state="disabled")
        self.status.config(text="working, this takes about a minute...")
        self.log.delete("1.0", "end")
        self.say("Running on " + Path(self.survey.get()).name + " ...")
        threading.Thread(target=self._work, args=(args, out),
                         daemon=True).start()

    def _work(self, args, out):
        """The CLI, called in a thread, with its printing captured.

        Same code path as the command line: there is no second, friendlier
        implementation that could drift away from the tested one.
        """
        import contextlib
        import io

        from src.product import cli

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                cli.main(args)
            self.q.put(("done", buf.getvalue(), out))
        except SystemExit as e:
            self.q.put(("refused", buf.getvalue() + "\n" + str(e), None))
        except Exception:
            self.q.put(("error", buf.getvalue() + "\n" + traceback.format_exc(),
                        None))

    def _drain(self):
        try:
            kind, text, out = self.q.get_nowait()
        except queue.Empty:
            self.root.after(120, self._drain)
            return
        self.busy = False
        self.go.config(state="normal")
        self.status.config(text="")
        for line in text.strip().splitlines():
            self.say(line)
        if kind == "done":
            self.say("")
            self.say("Report saved next to your file:")
            self.say("  " + str(out))
            webbrowser.open(out.as_uri())
        elif kind == "refused":
            self.say("")
            self.say("Gurutva refused to answer. The reason is above, and it "
                     "is the point of the tool: it will not give you a number "
                     "it cannot stand behind.")
        else:
            self.say("")
            self.say("Something broke that should not have. The trace above "
                     "is the bug report.")
        self.root.after(120, self._drain)


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
