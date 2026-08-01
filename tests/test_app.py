"""Gates on the window, which is the only surface most users will ever see.

The engine and the CLI were both real and both unusable by the person this is
for: a geoscientist with a laptop and a CSV had to install Python, obtain a
private repo, open a terminal, and type six flags. These tests cover the rung
that fixes that -- especially the refusals, because a first-time user meets
those far more often than the happy path.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

tk = pytest.importorskip("tkinter")


@pytest.fixture(scope="module")
def _root():
    """ONE Tk root for the whole module.

    Creating and destroying a root per test was intermittently failing with
    "no display available" -- the same test skipped on one run and passed on
    the next, which is worse than either outcome because it hides real
    breakage behind a coin flip.
    """
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def app(_root):
    from src.product import app as A
    for child in _root.winfo_children():
        child.destroy()
    return A.App(_root)


def _last(a):
    return a.log.get("1.0", "end").strip().splitlines()[-1]


def test_the_example_button_fills_everything_in(app):
    """A first-time user must be able to see it work before owning any data."""
    app.example()
    assert Path(app.survey.get()).exists()
    assert float(app.noise.get()) > 0
    assert float(app.prior.get()) > 0
    assert float(app.corr.get()) > 0


def test_a_missing_file_is_refused_in_plain_words(app):
    app.run()
    assert "Choose a survey file first" in _last(app)


def test_text_where_a_number_belongs_names_the_field(app):
    """'invalid literal for float()' is not an error message for a geologist."""
    app.example()
    app.noise.set("about 0.05")
    app.run()
    msg = _last(app)
    assert "repeatability" in msg and "needs to be a number" in msg


def test_zero_and_negative_are_refused(app):
    app.example()
    for var in (app.noise, app.prior, app.corr):
        old = var.get()
        for bad in ("0", "-2"):
            var.set(bad)
            app.log.delete("1.0", "end")
            app.run()
            assert "greater than zero" in _last(app), f"{bad} slipped through"
        var.set(old)


def test_nothing_is_guessed_for_the_user(app):
    """The three declared numbers start EMPTY. Defaulting them would be the
    software inventing a statement about someone else's site."""
    assert app.noise.get() == ""
    assert app.prior.get() == ""


def test_the_worker_uses_the_same_code_path_as_the_cli(app, tmp_path):
    """No second, friendlier implementation that could drift from the tested
    one: the window calls cli.main and captures its printing."""
    import inspect
    from src.product import app as A
    src = inspect.getsource(A.App._work)
    assert "cli.main(args)" in src


def test_a_refusal_is_reported_as_a_refusal_not_a_crash(app, tmp_path):
    """When a gate or a guard says no, the user must be told that is the
    point of the tool, not that it broke."""
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c,d" + chr(10) + "1,2,3,4" + chr(10), encoding="utf-8")
    app.survey.set(str(bad))
    app.noise.set("0.05")
    app.prior.set("0.1")
    app.corr.set("400")
    args = ["report", "--survey", str(bad), "--field", "gravity",
            "--noise", "0.05", "--prior-sd", "0.1", "--corr-len", "400",
            "--out", str(tmp_path / "o.html")]
    app._work(args, tmp_path / "o.html")
    kind, text, _ = app.q.get()
    assert kind == "refused", f"got {kind}"
    assert "missing column" in text
