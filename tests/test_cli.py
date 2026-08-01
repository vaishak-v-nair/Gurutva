"""The product's front door. If this breaks, no customer can use anything."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.product import cli


def _survey(tmp_path, name="s.csv"):
    rng = np.random.default_rng(3)
    gx, gy = np.meshgrid(np.arange(8) * 250.0, np.arange(5) * 250.0,
                         indexing="ij")
    x, y = gx.ravel() + 331000, gy.ravel() + 4263000
    z = np.full(x.size, 1650.0)
    g = (0.1 * np.exp(-((x - x.mean()) ** 2 + (y - y.mean()) ** 2) / 4e5)
         + rng.normal(0, 0.02, x.size))
    p = tmp_path / name
    np.savetxt(p, np.column_stack([x, y, z, g]), delimiter=",",
               header="x,y,z,gz", comments="", fmt="%.3f")
    return str(p)


BASE = ["--noise", "0.02", "--prior-sd", "0.1", "--corr-len", "400",
        "--cell", "300", "--depth", "900", "--samples", "120"]


def test_report_mode_produces_a_report(tmp_path):
    out = tmp_path / "r.html"
    cli.main(["report", "--survey", _survey(tmp_path)] + BASE
             + ["--out", str(out)])
    assert out.exists() and out.stat().st_size > 2000
    assert out.with_suffix(".json").exists()
    html = out.read_text(encoding="utf-8")
    assert "excess mass" in html
    assert "CLAIMABLE" in html or "NOT CLAIMED" in html


def test_selftest_actually_runs_the_recovery_gate(tmp_path):
    """On real data gate 5 cannot run. selftest exists so that it can."""
    out = tmp_path / "c.html"
    cli.main(["selftest", "--survey", _survey(tmp_path)] + BASE
             + ["--body-depth", "400", "--body-radius", "300",
                "--out", str(out)])
    html = out.read_text(encoding="utf-8")
    assert "recovery" in html
    assert "RECOVERY UNTESTED" not in html, "selftest must actually test it"


def test_report_mode_states_that_recovery_was_untested(tmp_path):
    """Silence must never read as success on real data."""
    out = tmp_path / "r2.html"
    cli.main(["report", "--survey", _survey(tmp_path)] + BASE
             + ["--out", str(out)])
    html = out.read_text(encoding="utf-8")
    if "NOT CLAIMED" not in html:
        assert "RECOVERY UNTESTED" in html


def test_bad_csv_is_refused_with_a_useful_message(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        cli.main(["report", "--survey", str(p), "--noise", "0.02",
                  "--prior-sd", "0.1", "--corr-len", "400"])
    assert "missing column" in str(e.value)


def test_too_few_stations_is_refused(tmp_path):
    p = tmp_path / "tiny.csv"
    p.write_text("x,y,z,gz\n0,0,0,1\n1,1,0,1\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        cli.main(["report", "--survey", str(p), "--noise", "0.02",
                  "--prior-sd", "0.1", "--corr-len", "400"])
    assert "Too few" in str(e.value)


def test_oversized_mesh_is_refused_not_attempted(tmp_path):
    """Refuse loudly rather than swap a laptop for an hour."""
    with pytest.raises(SystemExit) as e:
        cli.main(["report", "--survey", _survey(tmp_path), "--noise", "0.02",
                  "--prior-sd", "0.1", "--corr-len", "400", "--cell", "20",
                  "--depth", "3000"])
    assert "limit" in str(e.value)


def test_noise_is_mandatory():
    """A tool that invents a noise floor is inventing its own passing grade."""
    with pytest.raises(SystemExit):
        cli.main(["report", "--survey", "x.csv", "--prior-sd", "0.1",
                  "--corr-len", "400"])
