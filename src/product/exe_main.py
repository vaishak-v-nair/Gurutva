"""The entry point for the single file, when there is no Python on the machine.

The zip was the rung below this one: 61 KB, but it still asked a stranger to
install Python 3.10 and three packages before anything happened. On a locked
down work laptop — the exact machine this tool is aimed at — that is the end
of the road. This is the rung that removes it.

Two behaviours, one binary:

  Gurutva.exe                      opens the window
  Gurutva.exe report --survey ...  runs the command line, writing what it
                                   would have printed to a log beside the
                                   report

The second one exists because the first one cannot be verified without a pair
of human hands. A frozen binary that has never been run outside the tree it
was built in is not a product, and the only honest way to prove it runs is to
run it somewhere else, with nothing else present. That needs a mode with no
window in it.

There is still no second implementation of anything. Both paths end in
`cli.main`, the same function the 155 tests exercise.
"""

import io
import sys
from pathlib import Path


def close_splash():
    """Take the splash down the moment there is something behind it.

    A one-file build unpacks itself before a single line of this runs, and on
    a first run — cold cache, antivirus reading every byte — that is seconds
    of a double-click doing visibly nothing. Someone who has just been told
    this is a product will double-click again, and start a second unpack.
    The splash is the only thing that can speak during that window, because
    Python is not running yet when it appears.
    """
    try:
        import pyi_splash                     # injected only in frozen builds
    except ImportError:
        return
    try:
        pyi_splash.close()
    except Exception:                          # never let cosmetics kill a run
        pass


def root():
    """Where the bundled data lives — the repo, or the unpacked archive.

    In a PyInstaller one-file build the modules and the data are unpacked
    into a temporary directory, and `sys._MEIPASS` names it. `examples/`
    and `LICENSE` are added there at build time, so the window's "Use the
    example" button finds the same file it finds from a source checkout.
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


def _beside_binary(name):
    """Next to the .exe when frozen; next to whatever is running otherwise.

    `sys.executable` is the binary in a frozen build and the *interpreter* in
    a source checkout — where it points at the Python installation, which on
    the kind of managed laptop this tool is aimed at is read-only.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).with_name(name)
    return Path.cwd() / name


def _log_path(argv):
    """Beside --out if given, otherwise beside the binary."""
    for i, a in enumerate(argv):
        if a == "--out" and i + 1 < len(argv):
            return Path(argv[i + 1]).with_suffix(".log")
    return _beside_binary("gurutva.log")


def headless(argv):
    """Run the command line with its printing captured.

    A windowed build has no stdout at all — `sys.stdout` is None and the
    first `print` in the engine would take the whole run down with an
    AttributeError that names neither the cause nor the fix. So the output
    is collected and written to a file, and the exit code still means what
    it meant on the command line: 0 ran, 2 refused, 1 broke.
    """
    from src.product import cli

    buf = io.StringIO()
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = buf
    code = 0
    try:
        cli.main(argv)
    except SystemExit as e:                     # a gate said no
        code = e.code if isinstance(e.code, int) else 2
    except Exception:                           # a bug, and it gets reported
        import traceback
        traceback.print_exc(file=buf)
        code = 1
    finally:
        sys.stdout, sys.stderr = real_out, real_err

    text = buf.getvalue()
    try:
        _log_path(argv).write_text(text, encoding="utf-8")
    except OSError:
        pass
    if real_out is not None:                    # a console build, or a pipe
        real_out.write(text)
    return code


def where():
    """What the binary can actually see, from wherever it was double-clicked.

    The window's "Use the example" button reads `examples/demo_survey.csv`
    relative to the app module, not to this one. Inside a frozen build those
    are two different pieces of path arithmetic, so the answer is printed
    rather than assumed — by the build's own verification, and by anyone
    whose laptop puts the file somewhere unexpected.
    """
    from src.product import app

    lines = [
        f"frozen: {bool(getattr(sys, 'frozen', False))}",
        f"executable: {sys.executable}",
        f"root: {root()}",
        f"app root: {app.ROOT}",
    ]
    for rel in ("examples/demo_survey.csv", "examples/demo_magnetics.csv",
                "examples/demo_lease_block.csv", "LICENSE"):
        p = app.ROOT / rel
        lines.append(f"{'found  ' if p.exists() else 'MISSING'} {rel}")

    # The licence travels inside the binary, and a download that disagrees
    # with the repository about its own terms is worse than no download. The
    # bundled files are compressed, so nothing outside the running process
    # can read this — the binary has to say it itself.
    lic = app.ROOT / "LICENSE"
    if lic.exists():
        first = next((ln.strip() for ln in
                      lic.read_text(encoding="utf-8", errors="replace")
                      .splitlines() if ln.strip()), "")
        lines.append(f"licence: {first}")
    text = "\n".join(lines)
    if sys.stdout is not None:
        print(text)
    try:
        _beside_binary("gurutva-where.log").write_text(text, encoding="utf-8")
    except OSError:
        pass
    return 0 if all("MISSING" not in ln for ln in lines) else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    sys.path.insert(0, str(root()))
    if argv:                                   # nothing to look at: no splash
        close_splash()
        return where() if argv[0] == "--where" else headless(argv)
    from src.product import app
    app.main(on_ready=close_splash)
    return 0


if __name__ == "__main__":
    sys.exit(main())
