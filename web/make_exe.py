"""Freeze Gurutva into one file, then prove it runs where nothing else is.

The zip removed the repository from the visitor's path. It did not remove
Python: START-HERE.txt still opens with "Python 3.10 or newer, with three
packages", which on a managed corporate laptop is a request the person cannot
grant themselves. This removes that too. One file, double-click, no install.

What is deliberately kept out: SimPEG, discretize, torch and sbi. The only
route to them is `prior.build`'s TreeMesh path, imported inside the function
and never reached from the window, which uses `build_regular`. Excluding them
is the same decision make_bundle.py already made, for the same reason and with
the same risk if it is ever wrong — so the clean-room run below is what checks
it, not the reasoning.

Verification here is the point of the file, not a courtesy at the end of it.
A frozen binary that has only ever run inside its own build tree has not been
tested; every path it resolves is still the developer's. So it is copied to an
empty directory and run there:

  --where   does it see its own bundled examples, from a strange directory
  report    does the engine actually produce a report, end to end

Run: py -3 web/make_exe.py
"""

import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "src" / "product" / "exe_main.py"
NAME = "Gurutva"

# Carried inside the binary so "Use the example" works on a machine that has
# never seen this repository.
DATA = [
    ("examples/demo_survey.csv", "examples"),
    ("examples/demo_magnetics.csv", "examples"),
    ("examples/demo_lease_block.csv", "examples"),
    ("USAGE.md", "."),
    ("LICENSE", "."),
]

# Everything reachable only through the TreeMesh prior, which the window
# cannot reach. Left in, the download goes from tens of megabytes to over a
# gigabyte for a code path nobody holding this file can run.
#
# The second block was measured, not guessed. The first build came out at
# 163 MB; reading the archive back showed 85 MB of it was llvmlite, pyarrow,
# PIL, h5py, cryptography and the Jupyter stack — none of which appears in a
# single import statement anywhere the product can reach. They arrive through
# optional-dependency hooks in this environment. The clean-room run below is
# what proves the exclusion is safe; nothing here is safe by argument.
EXCLUDE = [
    "simpeg", "SimPEG", "discretize", "torch", "sbi", "pandas",
    "src.smoothness", "src.mesh", "pytest", "notebook",
    "PyQt5", "PyQt6", "PySide2", "PySide6", "wx",
    # measured dead weight
    "numba", "llvmlite", "pyarrow", "dask", "xarray", "h5py", "seaborn",
    "cryptography", "zmq", "tornado", "fsspec", "git", "pygments",
    "IPython", "ipykernel", "ipywidgets", "jupyter_client", "jupyter_core",
    "matplotlib_inline", "sqlite3", "setuptools",
    "pkg_resources", "docutils", "lxml", "yaml",
]
# PIL was on that list for one build. matplotlib.colors imports it at module
# level, so the binary built, started, found its own files, and then died the
# moment it was asked to do the one thing it exists for. Nothing short of
# running it would have found that.


def splash(work):
    """The only thing that can speak while the archive is still unpacking.

    Drawn here rather than committed as a binary, so what is on it is
    readable in the same diff as the decision to show it. It says the one
    true thing a first run needs: this takes a few seconds, once.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    work.mkdir(parents=True, exist_ok=True)
    out = work / "splash.png"
    fig = plt.figure(figsize=(4.4, 1.9), dpi=100)
    fig.patch.set_facecolor("#faf8f4")
    fig.text(0.5, 0.62, "Gurutva", ha="center", va="center",
             fontsize=30, family="Georgia", color="#12151a")
    fig.text(0.5, 0.40, "gravity, with error bars", ha="center", va="center",
             fontsize=10, family="Georgia", color="#5d6670")
    # Measured, not softened: 17s the first time, ~11s after, because a
    # one-file build unpacks 67 MB on every launch. An earlier draft of this
    # line said "a few seconds", which was a claim this project would have
    # retracted if anyone else had written it.
    fig.text(0.5, 0.16, "unpacking — about ten seconds, every launch",
             ha="center", va="center", fontsize=8, color="#8a929b")
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# Bump this with the release tag. It feeds both the Windows version resource
# and the string a frozen build stamps on every report it writes, so a stale
# value here means a report that names the wrong build.
VERSION = (0, 1, 2, 0)

VERSION_RESOURCE = """
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={v}, prodvers={v},
    mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'Vaishak V Nair'),
      StringStruct('FileDescription',
                   'Gurutva - posterior uncertainty for gravity and '
                   'magnetic surveys'),
      StringStruct('FileVersion', '{s}'),
      StringStruct('InternalName', 'Gurutva'),
      StringStruct('LegalCopyright',
                   'Copyright 2026 Vaishak V Nair. Apache-2.0.'),
      StringStruct('OriginalFilename', 'Gurutva.exe'),
      StringStruct('ProductName', 'Gurutva'),
      StringStruct('ProductVersion', '{s}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""


def version_file(work):
    """Give the binary a name Windows can show.

    An unsigned executable with no version resource makes SmartScreen's
    "More info" panel read `App: Gurutva.exe / Publisher: Unknown publisher`,
    which is the worst possible framing for a stranger who has just been told
    this is a real product. It does not remove the warning — only a
    code-signing certificate does that — but it fills in the name, the
    description and the copyright, so the panel describes something rather
    than nothing.
    """
    work.mkdir(parents=True, exist_ok=True)
    out = work / "version.txt"
    s = ".".join(str(n) for n in VERSION)
    out.write_text(VERSION_RESOURCE.replace("{v}", str(VERSION))
                   .replace("{s}", s), encoding="utf-8")
    return out


def stamp_version():
    """Write the version the frozen build will report as its own.

    The released binary stamped every report `gurutva` with no commit,
    because `_version()` shells out to git and a frozen build has neither a
    repository nor git. A report that cannot name the code that produced it
    is exactly what this project says an auditable document is not.

    Written into the package so PyInstaller picks it up as an ordinary
    module; gitignored so a working tree never carries one, and `_version()`
    only trusts it when frozen.
    """
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    v = ".".join(str(n) for n in VERSION[:3])
    label = f"{v}+{sha or 'nogit'}" + ("-dirty" if dirty else "")
    out = ROOT / "src" / "product" / "_build_version.py"
    out.write_text(
        '"""Written by web/make_exe.py at build time. Not in git."""\n\n'
        f'BUILD_VERSION = "{label}"\n', encoding="utf-8")
    print(f"stamped: {label}")
    return label


def build(onedir=False):
    """One file, or one folder.

    One file is the thing people ask for and the thing that is easy to hand
    over. It also pays its whole size in unpacking on *every* launch, because
    a one-file build extracts itself into a temporary directory each time it
    starts — not once. Which of those two costs matters more is a measurement,
    so both are built and both are timed.
    """
    dist = ROOT / "dist" / ("onedir" if onedir else "onefile")
    work = ROOT / "build" / ("pyinstaller-dir" if onedir else "pyinstaller")
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm",
           "--onedir" if onedir else "--onefile",
           "--windowed", "--name", NAME,
           "--distpath", str(dist), "--workpath", str(work),
           "--specpath", str(work),
           "--version-file", str(version_file(work)),
           "--hidden-import", "src.product._build_version",
           "--paths", str(ROOT)]
    stamp_version()
    if not onedir:                     # a folder build has nothing to unpack
        cmd += ["--splash", str(splash(work))]
    for src, dest in DATA:
        cmd += ["--add-data", f"{ROOT / src}{os_sep()}{dest}"]
    for mod in EXCLUDE:
        cmd += ["--exclude-module", mod]
    # Imported inside functions, which the analyser does find — named here so
    # a refactor that moves them cannot silently empty the binary.
    for mod in ("src.product.cli", "src.product.app", "src.product.report",
                "src.product.verdict", "src.product.survey",
                "src.product.prior", "src.product.joint"):
        cmd += ["--hidden-import", mod]
    cmd.append(str(ENTRY))

    print("building...")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit("PyInstaller failed")
    payload = (dist / NAME) if onedir else dist    # folder, or just the file
    exe = payload / f"{NAME}.exe"
    if not exe.exists():                           # not Windows
        exe = payload / NAME
    size = sum(f.stat().st_size for f in payload.rglob("*") if f.is_file())
    print(f"built in {time.perf_counter() - t0:.0f}s: {exe} "
          f"({size / 1e6:.1f} MB{' in the folder' if onedir else ''})")
    return payload, exe


def os_sep():
    return ";" if sys.platform == "win32" else ":"


def verify(payload, exe):
    """Run it where nothing else is: no repo, no src/, no examples/."""
    tmp = Path(tempfile.mkdtemp(prefix="gurutva_exe_"))
    try:
        if payload.is_dir():
            shutil.copytree(payload, tmp / payload.name)
            here = tmp / payload.name / exe.name
        else:
            here = tmp / exe.name
            shutil.copy2(exe, here)

        t0 = time.perf_counter()
        r = subprocess.run([str(here), "--where"], cwd=tmp,
                           capture_output=True, text=True, timeout=600)
        cold = time.perf_counter() - t0
        report = (tmp / "gurutva-where.log")
        text = report.read_text(encoding="utf-8") if report.exists() else ""
        print(f"--- --where (cold start {cold:.1f}s, exit {r.returncode}) ---")
        print(text or r.stdout or r.stderr[-1000:])
        if r.returncode != 0:
            raise SystemExit("the binary cannot find its own bundled files")

        # A stranger's own CSV, sitting in their own folder.
        shutil.copy2(ROOT / "examples" / "demo_survey.csv", tmp / "survey.csv")
        out = tmp / "survey_gurutva.html"
        t0 = time.perf_counter()
        r = subprocess.run(
            [str(here), "report", "--survey", str(tmp / "survey.csv"),
             "--noise", "0.05", "--prior-sd", "0.10", "--corr-len", "400",
             "--cell", "300", "--depth", "1200", "--samples", "120",
             "--out", str(out)],
            cwd=tmp, capture_output=True, text=True, timeout=1800)
        secs = time.perf_counter() - t0
        log = out.with_suffix(".log")
        ok = out.exists() and out.stat().st_size > 2000
        print(f"--- clean-room report ({secs:.0f}s, exit {r.returncode}, "
              f"written: {ok}) ---")
        if log.exists():
            print(log.read_text(encoding="utf-8")[-1500:])
        if not ok:
            print(r.stdout[-1500:], r.stderr[-1500:])
            raise SystemExit("the binary does not produce a report on its own")
        print(f"report: {out.stat().st_size / 1024:.0f} KB")

        # What the launch actually costs, three times, after the first.
        # `--where` does the same startup work the window does and then
        # exits, so it times the part the user is staring at.
        laps = []
        for _ in range(3):
            t0 = time.perf_counter()
            subprocess.run([str(here), "--where"], cwd=tmp,
                           capture_output=True, timeout=600)
            laps.append(time.perf_counter() - t0)
        print("startup after the first: " +
              ", ".join(f"{s:.1f}s" for s in laps))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


FIRST_RUN = """GURUTVA - start here
=====================

Double-click  Gurutva.exe

That is the whole installation. No Python, no pip, no admin rights, nothing
to install. Everything it needs is inside this folder.

Press "Use the example" and then "Run" to watch it work on a survey that
ships with it. Then choose your own CSV.

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

WINDOWS WILL WARN YOU
---------------------
This file is not code-signed, so SmartScreen will say "unrecognised app".
That is a statement about a certificate nobody has bought yet, not about the
file. "More info" then "Run anyway" if you trust where you got it - and if
you do not, that is the correct instinct.

vaishak.v.nair.dev@gmail.com
"""


def package(payload):
    """Zip the folder build, the way it will actually be handed over.

    Zipping is not a formality here: the one-file build pays ~11 seconds of
    self-unpacking on EVERY launch, and the folder build pays none. The
    download is roughly the same size either way, because the one file is
    the same content already compressed.
    """
    out = ROOT / "web" / "Gurutva-windows.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(payload.rglob("*")):
            if f.is_file():
                z.write(f, Path(NAME) / f.relative_to(payload))
        z.writestr(f"{NAME}/START-HERE.txt", FIRST_RUN)
    print(f"package: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def verify_package(zip_path):
    """Unzip somewhere empty and run what came out — not what built it."""
    tmp = Path(tempfile.mkdtemp(prefix="gurutva_zip_"))
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        exe = tmp / NAME / f"{NAME}.exe"
        r = subprocess.run([str(exe), "--where"], cwd=tmp,
                           capture_output=True, text=True, timeout=600)
        print(f"unzipped and run: exit {r.returncode}")
        if r.returncode != 0:
            raise SystemExit("the zip does not run once unzipped")
        assert (tmp / NAME / "START-HERE.txt").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["onefile", "onedir"]
    for kind in which:
        print(f"\n================ {kind} ================")
        payload, exe = build(onedir=(kind == "onedir"))
        verify(payload, exe)
        if kind == "onedir":
            verify_package(package(payload))
    print("\nThe window itself still needs a screen: run the built exe.")
