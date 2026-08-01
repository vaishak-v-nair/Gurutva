"""Package Gurutva into one zip a visitor can actually download and run.

The page told people to double-click Gurutva. Nothing existed for them to
double-click: the launcher lived only in a private repository. That is an
instruction a visitor cannot follow, which is worse than no instruction.

This builds the smallest set of files that actually runs the window, tests
that the result works from a clean directory with nothing else present, and
emits it as a zip the landing page embeds directly. No server, no account,
no repository access.

What is deliberately NOT included: src/smoothness.py and src/mesh.py, which
pull in SimPEG. The only caller is prior.build(), the TreeMesh path, and it
imports them inside the function rather than at module level. The window uses
build_regular and never touches it. Including them would turn a 200 KB
download into a several-hundred-megabyte dependency for a code path nobody in
this bundle can reach.

Run: py -3 web/make_bundle.py
"""

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = [
    "src/__init__.py",
    "src/prism_forward.py",
    "src/mag_forward.py",
    "src/posterior.py",
    "src/gurutva_core/__init__.py",
    "src/gurutva_core/gates.py",
    "src/gurutva_core/lensing.py",
    "src/gurutva_core/astro.py",
    "src/product/__init__.py",
    "src/product/app.py",
    "src/product/cli.py",
    "src/product/prior.py",
    "src/product/report.py",
    "src/product/survey.py",
    "src/product/verdict.py",
    "src/product/joint.py",
    "examples/demo_survey.csv",
    "examples/demo_magnetics.csv",
    "examples/demo_lease_block.csv",
    "Gurutva.bat",
    "gurutva",
    "USAGE.md",
    "LICENSE",
]

FIRST_RUN = """GURUTVA - start here
=====================

WHAT YOU NEED FIRST
-------------------
Python 3.10 or newer, with three packages:

    pip install numpy scipy matplotlib

That is the honest requirement. If you do not have Python, install it from
python.org first (tick "Add Python to PATH" on Windows).

TO START
--------
Windows:   double-click  Gurutva.bat
Mac/Linux: run           ./gurutva

A window opens. Press "Use the example" and then "Run" to watch it work on a
survey that ships with it. Then choose your own CSV.

YOUR DATA
---------
Four columns with a header row:

    x,y,z,gz
    331000,4263000,1650.2,-0.084

x,y  position in metres (or lon,lat in degrees - detected automatically)
z    station elevation in metres
gz   your gravity reading in mGal, or total field in nT for magnetics

WHAT IT ASKS YOU FOR, AND WHY IT WILL NOT GUESS
-----------------------------------------------
Three numbers, all statements about YOUR ground:

  how repeatable your instrument is
  how much the rock varies at your site
  over what distance it changes

A tool that invents a noise floor is inventing its own passing grade. The
answer it gives you is only as defensible as those three numbers, so you
type them and the report prints them back.

WHAT YOU GET
------------
An HTML report saved next to your data. It says one of three things:

  NOT CLAIMED   a check failed. The numbers are diagnostics, not decisions.
  PROVISIONAL   every check passed, but nothing was compared against a known
                right answer, because real data has none.
  CLAIMABLE     every check passed AND it was verified against a known answer.

If it refuses, that is the product working, not breaking.

Nothing is uploaded. It runs entirely on your machine.

vaishak.v.nair.dev@gmail.com
"""


def build():
    missing = [f for f in FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"missing from the repo: {missing}")

    out = ROOT / "web" / "gurutva.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in FILES:
            z.write(ROOT / f, f"gurutva/{f}")
        z.writestr("gurutva/START-HERE.txt", FIRST_RUN)
    print(f"bundle: {out}  ({out.stat().st_size / 1024:.0f} KB, "
          f"{len(FILES) + 1} files)")
    return out


def verify(zip_path):
    """Unpack into an empty directory and run it there, with nothing else
    present. A bundle that only works inside the repo it came from is not a
    bundle."""
    tmp = Path(tempfile.mkdtemp(prefix="gurutva_bundle_"))
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        work = tmp / "gurutva"
        rep = work / "out.html"
        r = subprocess.run(
            [sys.executable, "-m", "src.product.cli", "report",
             "--survey", str(work / "examples" / "demo_survey.csv"),
             "--noise", "0.05", "--prior-sd", "0.10", "--corr-len", "400",
             "--cell", "300", "--depth", "1200", "--samples", "120",
             "--out", str(rep)],
            cwd=work, capture_output=True, text=True, timeout=900)
        ok = rep.exists() and rep.stat().st_size > 2000
        print(f"clean-room run: exit {r.returncode}, report written: {ok}")
        if not ok:
            print("--- stdout ---\n" + r.stdout[-1500:])
            print("--- stderr ---\n" + r.stderr[-1500:])
            raise SystemExit("the bundle does not run on its own")
        # and the window must at least import with nothing else around
        r2 = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0,'.');"
             "import src.product.app as a; print('window imports ok')"],
            cwd=work, capture_output=True, text=True, timeout=300)
        print("window import:", r2.stdout.strip() or r2.stderr.strip()[-200:])
        if r2.returncode != 0:
            raise SystemExit("the window does not import from the bundle")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    p = build()
    verify(p)
    print("web/build_interactive.py encodes this zip directly.")
