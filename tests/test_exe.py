"""Gates on the single file — the rung that removes Python from the ask.

The zip removed the repository from a stranger's path. It did not remove
"install Python 3.10 and three packages", which on a managed work laptop is
not a step the person is allowed to take. `exe_main` is what a frozen build
starts at, and everything here is about the two things that are different
inside a frozen build and cannot be noticed from a source checkout:

  where its files are     `sys._MEIPASS`, not the repository
  where its printing goes nowhere at all — a windowed build has no stdout,
                          so the first print in the engine would otherwise
                          take the run down with an AttributeError

The end-to-end proof that the built binary works lives in web/make_exe.py,
which runs it in an empty directory. These are the fast checks that keep the
logic honest between builds.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.product import exe_main as X                       # noqa: E402


# ------------------------------------------------------------------ paths

def test_root_is_the_repo_when_not_frozen():
    assert X.root() == ROOT
    assert (X.root() / "examples" / "demo_survey.csv").exists()


def test_root_follows_meipass_when_frozen(monkeypatch, tmp_path):
    """A frozen build unpacks somewhere new on every launch. Reading the
    repository path out of __file__ there would find a directory that does
    not exist on the user's machine at all."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert X.root() == tmp_path


def test_the_log_lands_beside_the_report(tmp_path):
    out = tmp_path / "survey_gurutva.html"
    assert X._log_path(["report", "--out", str(out)]) == out.with_suffix(".log")


def test_without_out_the_log_lands_beside_the_binary(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert X._log_path(["report"]) == tmp_path / "gurutva.log"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert X._log_path(["report"]).parent == Path(sys.executable).parent


def test_nothing_is_written_into_the_python_installation(monkeypatch,
                                                         tmp_path):
    """A read-only Python install is normal on a managed machine, and a tool
    that writes its diagnostics there fails for exactly the user it is for."""
    monkeypatch.chdir(tmp_path)
    X.where()
    assert (tmp_path / "gurutva-where.log").exists()
    assert not Path(sys.executable).with_name("gurutva-where.log").exists()


# --------------------------------------------------------------- headless

def test_headless_writes_what_the_cli_printed(monkeypatch, tmp_path):
    """A windowed build has no stdout. If the engine's printing is not
    captured, the user's only record of the run is that a file appeared."""
    out = tmp_path / "r.html"

    def fake_main(argv):
        print("[PASS] adequacy: the only gate that tests reality")

    monkeypatch.setattr("src.product.cli.main", fake_main)
    code = X.headless(["report", "--out", str(out)])
    assert code == 0
    assert "adequacy" in out.with_suffix(".log").read_text(encoding="utf-8")


def test_a_refusal_keeps_its_exit_code(monkeypatch, tmp_path):
    """A gate saying no is the product working. It must not look like a
    crash to whatever launched it."""
    out = tmp_path / "r.html"

    def refuse(argv):
        print("licensing: FAIL")
        raise SystemExit(2)

    monkeypatch.setattr("src.product.cli.main", refuse)
    assert X.headless(["report", "--out", str(out)]) == 2
    assert "licensing" in out.with_suffix(".log").read_text(encoding="utf-8")


def test_a_crash_is_reported_rather_than_swallowed(monkeypatch, tmp_path):
    out = tmp_path / "r.html"

    def boom(argv):
        raise ValueError("mesh went sideways")

    monkeypatch.setattr("src.product.cli.main", boom)
    assert X.headless(["report", "--out", str(out)]) == 1
    log = out.with_suffix(".log").read_text(encoding="utf-8")
    assert "mesh went sideways" in log and "Traceback" in log


def test_stdout_is_put_back_afterwards(monkeypatch, tmp_path):
    """Redirecting sys.stdout and failing to restore it would silence
    everything that ran next, including the window's own reporting."""
    before = sys.stdout
    monkeypatch.setattr("src.product.cli.main",
                        lambda argv: (_ for _ in ()).throw(RuntimeError("x")))
    X.headless(["report", "--out", str(tmp_path / "r.html")])
    assert sys.stdout is before


# ------------------------------------------------------------- dispatch

def test_no_arguments_opens_the_window(monkeypatch):
    seen = {}
    monkeypatch.setattr("src.product.app.main",
                        lambda on_ready=None: seen.update(ready=on_ready))
    assert X.main([]) == 0
    assert seen["ready"] is X.close_splash


def test_arguments_run_the_engine_and_never_open_a_window(monkeypatch):
    monkeypatch.setattr("src.product.app.main",
                        lambda on_ready=None: pytest.fail("opened a window"))
    monkeypatch.setattr(X, "headless", lambda argv: 7)
    assert X.main(["report", "--survey", "x.csv"]) == 7


def test_where_reports_a_missing_bundled_file(monkeypatch, tmp_path):
    """The window's 'Use the example' button reads examples/ relative to the
    app module. Inside a frozen build that is different path arithmetic from
    this module's, so it is checked rather than assumed."""
    from src.product import app as A
    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    assert X.where() == 1
    assert "MISSING" in (tmp_path / "gurutva-where.log").read_text(
        encoding="utf-8")


def test_where_passes_against_the_real_tree():
    assert X.where() == 0


def test_close_splash_is_harmless_outside_a_frozen_build():
    X.close_splash()                    # no pyi_splash module: must not raise


# ----------------------------------------------------------- the window

def test_the_window_signals_ready_before_it_blocks(monkeypatch):
    """on_ready fires after the window is up and BEFORE mainloop, or the
    splash would sit on top of the thing it was covering for."""
    tk = pytest.importorskip("tkinter")
    from src.product import app as A

    order = []
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    root.destroy()

    monkeypatch.setattr(A.tk, "Tk", lambda: _FakeRoot(order))
    monkeypatch.setattr(A, "App", lambda root: order.append("built"))
    monkeypatch.setattr(A.ttk, "Style", lambda: _NoStyle())
    A.main(on_ready=lambda: order.append("ready"))
    assert order == ["built", "update", "ready", "mainloop"]


class _FakeRoot:
    def __init__(self, order):
        self.order = order

    def update(self):
        self.order.append("update")

    def mainloop(self):
        self.order.append("mainloop")


class _NoStyle:
    def theme_use(self, name):
        pass


# --------------------------------------------------- the build manifest

def test_everything_the_build_carries_actually_exists():
    """A build that silently drops examples/ still starts, still opens, and
    fails only when someone presses 'Use the example'."""
    sys.path.insert(0, str(ROOT / "web"))
    import make_exe

    for src, _dest in make_exe.DATA:
        assert (ROOT / src).exists(), f"make_exe carries a missing file: {src}"


def test_the_bundled_examples_match_what_the_window_offers():
    sys.path.insert(0, str(ROOT / "web"))
    import make_exe

    carried = {s for s, _ in make_exe.DATA if s.startswith("examples/")}
    assert "examples/demo_survey.csv" in carried, (
        "the window's 'Use the example' button reads demo_survey.csv")
