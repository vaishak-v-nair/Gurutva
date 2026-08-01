"""The product's front door. If this breaks, no customer can use anything.

The second half of this file is an adversarial pass made on 2026-08-05:
twelve malformed or degenerate surveys were pushed through the CLI the way a
real user would hit them. Four crashed with tracebacks and one misreported
its cause. Each is now a refusal written in the user's language, pinned here
so it stays that way.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.product import cli, survey as SV

BASE = ["--noise", "0.02", "--prior-sd", "0.1", "--corr-len", "400",
        "--cell", "300", "--depth", "900", "--samples", "120"]
NL = chr(10)


def _grid(n_x=8, n_y=5, sep=",", z=1650.0):
    """Rows of a small 2-D survey, joined with `sep`."""
    out = []
    for i in range(n_x):
        for j in range(n_y):
            out.append(sep.join([f"{331000 + i * 250}", f"{4263000 + j * 250}",
                                 f"{z}", "0.01"]))
    return out


def _write(tmp_path, name, header, rows, bom=False):
    p = tmp_path / name
    text = header + NL + NL.join(rows) + NL
    p.write_text(("﻿" if bom else "") + text, encoding="utf-8")
    return str(p)


def _survey(tmp_path, name="s.csv"):
    """A well-formed survey with real signal, for the happy paths."""
    rng = np.random.default_rng(3)
    gx, gy = np.meshgrid(np.arange(8) * 250.0, np.arange(5) * 250.0,
                         indexing="ij")
    x, y = gx.ravel() + 331000, gy.ravel() + 4263000
    g = (0.1 * np.exp(-((x - x.mean()) ** 2 + (y - y.mean()) ** 2) / 4e5)
         + rng.normal(0, 0.02, x.size))
    p = tmp_path / name
    np.savetxt(p, np.column_stack([x, y, np.full(x.size, 1650.0), g]),
               delimiter=",", header="x,y,z,gz", comments="", fmt="%.3f")
    return str(p)


# ----------------------------------------------------------- the happy path
def test_report_mode_produces_a_report(tmp_path):
    out = tmp_path / "r.html"
    cli.main(["report", "--survey", _survey(tmp_path)] + BASE
             + ["--out", str(out)])
    assert out.exists() and out.stat().st_size > 2000
    assert out.with_suffix(".json").exists()
    html = out.read_text(encoding="utf-8")
    assert "excess mass" in html
    assert any(w in html for w in ("CLAIMABLE", "PROVISIONAL", "NOT CLAIMED"))


def test_selftest_actually_runs_the_recovery_gate(tmp_path):
    """On real data gate 5 cannot run. selftest exists so that it can."""
    out = tmp_path / "c.html"
    cli.main(["selftest", "--survey", _survey(tmp_path)] + BASE
             + ["--body-depth", "400", "--body-radius", "300",
                "--out", str(out)])
    html = out.read_text(encoding="utf-8")
    assert "recovery" in html
    assert "PROVISIONAL" not in html, "selftest checks a truth, so not provisional"


def test_report_mode_says_PROVISIONAL_not_CLAIMABLE(tmp_path):
    """Silence must never read as success on real data.

    The old wording printed the green word CLAIMABLE and then, in the same
    breath, 'nothing here has been checked against a right answer'. A reader
    who scans stops at the green word. Real data can never run gate 5, so
    this is the normal case, not an edge case."""
    out = tmp_path / "r2.html"
    cli.main(["report", "--survey", _survey(tmp_path)] + BASE
             + ["--out", str(out)])
    html = out.read_text(encoding="utf-8")
    if "NOT CLAIMED" not in html:
        assert "PROVISIONAL" in html
        assert "class='word'>CLAIMABLE<" not in html


# ------------------------------------------------------- refusals at the door
def test_bad_csv_is_refused_with_a_useful_message(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b,c,d" + NL + "1,2,3,4" + NL, encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        cli.main(["report", "--survey", str(p), "--noise", "0.02",
                  "--prior-sd", "0.1", "--corr-len", "400"])
    assert "missing column" in str(e.value)


def test_too_few_stations_is_refused(tmp_path):
    p = _write(tmp_path, "tiny.csv", "x,y,z,gz",
               ["0,0,0,1", "250,250,0,1", "500,0,0,1"])
    with pytest.raises(SystemExit) as e:
        cli.main(["report", "--survey", p, "--noise", "0.02",
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


# ------------------------------- the adversarial pass (all of these crashed)
def test_collinear_survey_is_refused_not_crashed(tmp_path):
    """A single line of stations cannot constrain a 3-D field. This died
    inside the sparse factoriser with 'zero-size array to reduction operation
    minimum', which tells a geologist nothing."""
    rows = [f"{331000 + i * 250},4263000,1650,0.01" for i in range(40)]
    p = _write(tmp_path, "line.csv", "x,y,z,gz", rows)
    with pytest.raises(SystemExit) as e:
        cli.main(["report", "--survey", p] + BASE)
    assert "single line" in str(e.value)


def test_nan_is_refused_at_the_door_with_a_line_number(tmp_path):
    """A blank reading sailed through and was caught three gates later by
    luck. Refuse it, and say which line."""
    rows = _grid()
    rows[4] = "331000,4264000,1650,"
    p = _write(tmp_path, "nan.csv", "x,y,z,gz", rows)
    with pytest.raises(SystemExit) as e:
        cli.main(["report", "--survey", p] + BASE)
    msg = str(e.value)
    assert "line 6" in msg and "non-numeric" in msg


def test_semicolon_csv_is_read(tmp_path):
    """The European export default. Died inside genfromtxt."""
    p = _write(tmp_path, "semi.csv", "x;y;z;gz", _grid(sep=";"))
    out = tmp_path / "s.html"
    cli.main(["report", "--survey", p] + BASE + ["--out", str(out)])
    assert out.exists()


def test_mismatched_delimiters_name_the_real_problem(tmp_path):
    """Header comma-separated, data semicolon-separated: reported 'only 0
    stations', which blames the wrong thing."""
    p = _write(tmp_path, "mixed.csv", "x,y,z,gz", _grid(sep=";"))
    with pytest.raises(SystemExit) as e:
        cli.main(["report", "--survey", p] + BASE)
    assert "separat" in str(e.value)


def test_utf8_bom_does_not_hide_the_x_column(tmp_path):
    """A BOM turned the first column into a name starting with U+FEFF, so the
    error blamed a missing x. Excel writes BOMs by default."""
    p = _write(tmp_path, "bom.csv", "x,y,z,gz", _grid(), bom=True)
    out = tmp_path / "b.html"
    cli.main(["report", "--survey", p] + BASE + ["--out", str(out)])
    assert out.exists()


def test_uppercase_and_padded_headers_are_accepted(tmp_path):
    """Real exports write ' X , Y , Z , GZ '. Refusing those is pedantry."""
    p = _write(tmp_path, "case.csv", " X , Y , Z , GZ ", _grid())
    out = tmp_path / "u.html"
    cli.main(["report", "--survey", p] + BASE + ["--out", str(out)])
    assert out.exists()


def test_wrong_units_are_caught_by_the_gates(tmp_path):
    """microGal submitted where mGal is required: 1000x too big. The gates
    must refuse rather than return a confident wrong answer."""
    rng = np.random.default_rng(5)
    gx, gy = np.meshgrid(np.arange(8) * 250.0, np.arange(5) * 250.0,
                         indexing="ij")
    x, y = gx.ravel() + 331000, gy.ravel() + 4263000
    g = 1000 * (0.1 * np.exp(-((x - x.mean()) ** 2 + (y - y.mean()) ** 2) / 4e5)
                + rng.normal(0, 0.02, x.size))
    p = tmp_path / "ug.csv"
    np.savetxt(p, np.column_stack([x, y, np.full(x.size, 1650.0), g]),
               delimiter=",", header="x,y,z,gz", comments="", fmt="%.3f")
    out = tmp_path / "w.html"
    code = cli.main(["report", "--survey", str(p)] + BASE + ["--out", str(out)])
    assert code == 1, "a 1000x unit error must not come back claimable"
    assert "NOT CLAIMED" in out.read_text(encoding="utf-8")


def test_duplicate_stations_are_warned_about(tmp_path, capsys):
    """Repeats add no information but make the survey look better resolved."""
    rows = _grid()
    p = _write(tmp_path, "dup.csv", "x,y,z,gz", rows + rows[:6])
    cli.main(["report", "--survey", p] + BASE + ["--out", str(tmp_path / "d.html")])
    assert "repeat an existing" in capsys.readouterr().out


def test_the_report_shows_the_figure_it_promises(tmp_path):
    """The footer advertised 'per-cell map from N posterior samples' while no
    map was ever rendered: the arrays were computed and thrown away."""
    out = tmp_path / "f.html"
    cli.main(["report", "--survey", _survey(tmp_path)] + BASE
             + ["--out", str(out)])
    html = out.read_text(encoding="utf-8")
    assert "data:image/png;base64" in html, "the promised map must exist"
    assert out.with_suffix(".png").exists()


def test_gate_details_survive_html_escaping(tmp_path):
    """Gate thresholds contain '<' ('median < 0.115'), which a browser reads
    as a tag and silently eats the rest of the row."""
    out = tmp_path / "e.html"
    cli.main(["report", "--survey", _survey(tmp_path)] + BASE
             + ["--out", str(out)])
    html = out.read_text(encoding="utf-8")
    import re
    details = re.findall(r"<span class='detail'>(.*?)</span>", html, re.S)
    assert len(details) == 4, "one detail line per core gate"
    assert any("&lt;" in d for d in details), "'<' must be escaped, not eaten"
    assert all("pct" in d or "floor" in d or "ratio" in d or "pp" in d
               for d in details), "no row may be truncated"


def test_the_report_is_dated_and_versioned(tmp_path):
    """It goes into a meeting where money is decided."""
    out = tmp_path / "v.html"
    cli.main(["report", "--survey", _survey(tmp_path)] + BASE
             + ["--out", str(out)])
    html = out.read_text(encoding="utf-8")
    import re
    assert re.search(r"\d{4}-\d{2}-\d{2}", html), "must carry a date"
    assert "gurutva" in html, "must say which code produced it"
    assert "@media print" in html, "it will be printed"


# ---------------------------------- magnetics, geography, terrain, lease blocks
def _grid_xyz(n_x=8, n_y=6, spacing=250.0):
    gx, gy = np.meshgrid(np.arange(n_x) * spacing, np.arange(n_y) * spacing,
                         indexing="ij")
    return gx.ravel() + 331000, gy.ravel() + 4263000


def _write_rows(tmp_path, name, header, cols, fmt="%.6f"):
    p = tmp_path / name
    np.savetxt(p, np.column_stack(cols), delimiter=",", header=header,
               comments="", fmt=fmt)
    return str(p)


def test_magnetic_survey_runs_end_to_end(tmp_path):
    """A second physics through the same engine: nT in, susceptibility out."""
    from src import mag_forward as mf
    x, y = _grid_xyz()
    z = np.full(x.size, 1650.0)
    st = np.column_stack([x, y, z])
    c = np.array([[x.mean(), y.mean(), -700.0]])
    t = mf.mag_tf(st, c, np.array([[400.0, 400.0, 400.0]]), [0.05],
                  55000.0, 70.0, 2.0)
    t = t + np.random.default_rng(1).normal(0, 1.0, x.size)
    p = _write_rows(tmp_path, "mag.csv", "x,y,z,tmi", [x, y, z, t])
    out = tmp_path / "m.html"
    cli.main(["report", "--survey", p, "--field", "mag", "--noise", "1.0",
              "--prior-sd", "3e-4", "--corr-len", "400", "--cell", "300",
              "--depth", "1200", "--b0", "55000", "--inclination", "70",
              "--samples", "120", "--out", str(out)])
    html = out.read_text(encoding="utf-8")
    assert "susceptibility" in html and "nT" in html
    assert "induced magnetisation only" in html, "state the model-class limit"


def test_lon_lat_input_gives_the_same_answer_as_projected_metres(tmp_path):
    """THE projection gate at product level: the same survey expressed in
    degrees and in metres must produce the same number, or the tangent plane
    is silently rescaling the customer's ground."""
    import json
    x, y = _grid_xyz()
    z = np.full(x.size, 1650.0)
    g = (0.08 * np.exp(-((x - x.mean())**2 + (y - y.mean())**2) / 6e5)
         + np.random.default_rng(2).normal(0, 0.01, x.size))
    pm = _write_rows(tmp_path, "m.csv", "x,y,z,gz", [x, y, z, g], fmt="%.4f")

    lat0, lon0 = 38.5, -112.85
    s = np.sin(np.radians(lat0))
    w = 1.0 - SV.WGS84_E2 * s * s
    n_rad = SV.WGS84_A / np.sqrt(w)
    m_rad = SV.WGS84_A * (1.0 - SV.WGS84_E2) / w**1.5
    lon = lon0 + np.degrees((x - x.mean()) / (n_rad * np.cos(np.radians(lat0))))
    lat = lat0 + np.degrees((y - y.mean()) / m_rad)
    gp = _write_rows(tmp_path, "d.csv", "lon,lat,elev,mgal", [lon, lat, z, g],
                     fmt="%.9f")

    args = ["--noise", "0.01", "--prior-sd", "0.05", "--corr-len", "400",
            "--cell", "300", "--depth", "1200", "--samples", "120"]
    a, b = tmp_path / "a.html", tmp_path / "b.html"
    cli.main(["report", "--survey", pm] + args + ["--out", str(a)])
    cli.main(["report", "--survey", gp] + args + ["--out", str(b)])
    va = json.loads(a.with_suffix(".json").read_text())["value"]
    vb = json.loads(b.with_suffix(".json").read_text())["value"]
    assert abs(va - vb) < 1e-3 * max(abs(va), 1e-9), f"{va} vs {vb}"
    assert "tangent plane" in b.read_text(encoding="utf-8")


def test_topography_removes_air_cells(tmp_path):
    """Over 600 m of relief a flat-topped mesh hands the inversion air to put
    density into, and it will."""
    import json
    x, y = _grid_xyz()
    z = 1650 + 320 * np.sin((x - x.min()) / 700.0)
    g = np.random.default_rng(3).normal(0, 0.01, x.size)
    p = _write_rows(tmp_path, "t.csv", "x,y,z,gz", [x, y, z, g], fmt="%.4f")
    args = ["--noise", "0.01", "--prior-sd", "0.05", "--corr-len", "400",
            "--cell", "200", "--depth", "1200", "--samples", "120"]
    on, off = tmp_path / "on.html", tmp_path / "off.html"
    cli.main(["report", "--survey", p] + args + ["--topography", "on",
                                                 "--out", str(on)])
    cli.main(["report", "--survey", p] + args + ["--topography", "off",
                                                 "--out", str(off)])
    j_on = json.loads(on.with_suffix(".json").read_text())
    j_off = json.loads(off.with_suffix(".json").read_text())
    assert j_on["cells_active"] < j_off["cells_active"], "air must be dropped"
    assert j_off["cells_active"] == j_off["cells_total"]
    assert "draped" in on.read_text(encoding="utf-8")


def test_a_lease_block_polygon_is_used_as_the_region(tmp_path):
    """A customer reports on their own ground, not a box we chose."""
    import json
    x, y = _grid_xyz()
    z = np.full(x.size, 1650.0)
    g = (0.08 * np.exp(-((x - x.mean())**2 + (y - y.mean())**2) / 6e5)
         + np.random.default_rng(4).normal(0, 0.01, x.size))
    p = _write_rows(tmp_path, "s.csv", "x,y,z,gz", [x, y, z, g], fmt="%.4f")
    cx, cy = x.mean(), y.mean()
    poly = tmp_path / "block.csv"
    np.savetxt(poly, np.array([[cx - 600, cy - 500], [cx + 400, cy - 600],
                               [cx + 600, cy + 300], [cx - 500, cy + 500]]),
               delimiter=",", header="x,y", comments="", fmt="%.2f")
    out = tmp_path / "p.html"
    cli.main(["report", "--survey", p, "--noise", "0.01", "--prior-sd", "0.05",
              "--corr-len", "400", "--cell", "300", "--depth", "1200",
              "--samples", "120", "--region-file", str(poly), "--out", str(out)])
    html = out.read_text(encoding="utf-8")
    assert "block.csv" in html and "km2" in html


def test_a_mis_scaled_prior_is_warned_about_with_a_number(tmp_path, capsys):
    """--prior-sd is a PER-CELL sd, so a user hunting a 0.06 SI body types
    0.06 and declares a prior predicting hundreds of nT over a survey that
    reads single digits. Say so, and say what to use instead."""
    x, y = _grid_xyz()
    z = np.full(x.size, 1650.0)
    g = np.random.default_rng(5).normal(0, 0.01, x.size)
    p = _write_rows(tmp_path, "w.csv", "x,y,z,gz", [x, y, z, g], fmt="%.4f")
    cli.main(["report", "--survey", p, "--noise", "0.01", "--prior-sd", "5.0",
              "--corr-len", "400", "--cell", "300", "--depth", "1200",
              "--samples", "120", "--out", str(tmp_path / "w.html")])
    o = capsys.readouterr().out
    assert "WARNING" in o and "PER-CELL" in o and "try about" in o


def test_every_core_gate_actually_runs(tmp_path):
    """A refactor once inserted an early `return` after the licensing gate,
    which orphaned calibration and stability. Every run afterwards executed
    two of four gates and reported INCOMPLETE, and the printed PASS lines
    looked entirely normal. Count them."""
    out = tmp_path / "gates.html"
    cli.main(["report", "--survey", _survey(tmp_path)] + BASE
             + ["--out", str(out)])
    html = out.read_text(encoding="utf-8")
    for gate in ("licensing", "calibration", "stability", "adequacy"):
        assert gate in html, f"core gate {gate} never ran"
    assert "INCOMPLETE" not in html


# ------------------------- remanence, derived priors, joint inversion (CLI)
def _mag_survey(tmp_path, name="mag.csv", q=0.0, rem_inc=None):
    from src import mag_forward as mf
    x, y = _grid_xyz()
    z = np.full(x.size, 1650.0)
    st = np.column_stack([x, y, z])
    c = np.array([[x.mean(), y.mean(), -700.0]])
    d = np.array([[400.0, 400.0, 400.0]])
    t = mf.mag_tf(st, c, d, [0.05], 55000.0, 70.0, 2.0, q=q,
                  rem_inclination=rem_inc)
    t = t + np.random.default_rng(7).normal(0, 1.0, x.size)
    return _write_rows(tmp_path, name, "x,y,z,tmi", [x, y, z, t]), st


MAGBASE = ["--field", "mag", "--noise", "1.0", "--corr-len", "400",
           "--cell", "300", "--depth", "1200", "--b0", "55000",
           "--inclination", "70", "--declination", "2", "--samples", "120"]


def test_prior_can_be_derived_from_a_declared_target(tmp_path):
    """--prior-sd is a per-cell sd, so typing the contrast of the body you
    are hunting declares a prior hundreds of times too wide. Declaring the
    BODY instead fixes the amplitude from physics."""
    p, _ = _mag_survey(tmp_path)
    out = tmp_path / "t.html"
    cli.main(["report", "--survey", p] + MAGBASE
             + ["--target-contrast", "0.05", "--target-radius", "400",
                "--target-depth", "700", "--out", str(out)])
    html = out.read_text(encoding="utf-8")
    assert "derived from a declared target" in html
    assert "would make a" in html and "nT anomaly" in html


def test_prior_sd_or_a_target_is_required(tmp_path):
    """One of the two must be given: the tool never invents an amplitude."""
    p, _ = _mag_survey(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["report", "--survey", p, "--field", "mag", "--noise", "1.0",
                  "--corr-len", "400"])


def test_remanence_is_declared_in_the_report(tmp_path):
    """A declared magnetisation direction is an assumption the reader must
    see, not a hidden default."""
    p, _ = _mag_survey(tmp_path, q=2.0, rem_inc=-60.0)
    out = tmp_path / "r.html"
    cli.main(["report", "--survey", p] + MAGBASE
             + ["--prior-sd", "3e-4", "--remanence-q", "2.0",
                "--remanence-inclination", "-60", "--out", str(out)])
    html = out.read_text(encoding="utf-8")
    assert "Koenigsberger" in html and "DECLARED, not solved for" in html
    assert "induced magnetisation only" not in html


def test_induced_only_still_says_so(tmp_path):
    p, _ = _mag_survey(tmp_path)
    out = tmp_path / "i.html"
    cli.main(["report", "--survey", p] + MAGBASE
             + ["--prior-sd", "3e-4", "--out", str(out)])
    assert "induced magnetisation only" in out.read_text(encoding="utf-8")


def _pair(tmp_path):
    """A gravity and a magnetic survey at the SAME stations."""
    from src import mag_forward as mf, prism_forward as pf
    x, y = _grid_xyz()
    z = np.full(x.size, 1650.0)
    st = np.column_stack([x, y, z])
    c = np.array([[x.mean(), y.mean(), -700.0]])
    d = np.array([[400.0, 400.0, 400.0]])
    rng = np.random.default_rng(8)
    g = pf.prism_gz(st, c, d, [0.3]) + rng.normal(0, 0.01, x.size)
    t = mf.mag_tf(st, c, d, [0.05], 55000.0, 70.0, 2.0) + rng.normal(0, 1.0, x.size)
    return (_write_rows(tmp_path, "g.csv", "x,y,z,gz", [x, y, z, g], "%.5f"),
            _write_rows(tmp_path, "m.csv", "x,y,z,tmi", [x, y, z, t], "%.5f"))


JBASE = ["--noise", "0.01", "--noise-mag", "1.0", "--prior-sd", "0.05",
         "--prior-sd-chi", "3e-4", "--corr-len", "400", "--cell", "300",
         "--depth", "1200", "--b0", "55000", "--inclination", "70",
         "--samples", "120"]


def test_joint_inversion_runs_and_reports_both_properties(tmp_path):
    g, m = _pair(tmp_path)
    out = tmp_path / "j.html"
    cli.main(["joint", "--survey", g, "--survey-mag", m] + JBASE
             + ["--rho-chi-correlation", "0.7", "--out", str(out)])
    html = out.read_text(encoding="utf-8")
    assert "density" in html and "susceptibility" in html
    assert "AN ASSUMPTION about" in html, "the correlation must be flagged"


def test_joint_at_zero_correlation_buys_exactly_nothing(tmp_path):
    """The product-level twin of the exactness gate in test_joint.py: with
    the two properties declared unrelated, adding magnetics cannot narrow a
    density interval by even a rounding error."""
    import json
    g, m = _pair(tmp_path)
    out = tmp_path / "j0.html"
    cli.main(["joint", "--survey", g, "--survey-mag", m] + JBASE
             + ["--rho-chi-correlation", "0.0", "--out", str(out)])
    j = json.loads(out.with_suffix(".json").read_text())
    assert abs(j["tighter_pct"]) < 0.05, f"bought {j['tighter_pct']:.3f}%"


def test_joint_with_correlation_narrows_the_interval(tmp_path):
    import json
    g, m = _pair(tmp_path)
    vals = {}
    for r in ("0.0", "0.8"):
        out = tmp_path / f"j{r}.html"
        cli.main(["joint", "--survey", g, "--survey-mag", m] + JBASE
                 + ["--rho-chi-correlation", r, "--out", str(out)])
        vals[r] = json.loads(out.with_suffix(".json").read_text())["value_sd"]
    assert vals["0.8"] < vals["0.0"], "a declared correlation must buy something"


def test_joint_refuses_surveys_at_different_stations(tmp_path):
    """Interpolating one survey onto the other invents data, and the invented
    part would carry no error bar."""
    g, m = _pair(tmp_path)
    x, y = _grid_xyz(n_x=7, n_y=5)
    z = np.full(x.size, 1650.0)
    bad = _write_rows(tmp_path, "bad.csv", "x,y,z,tmi",
                      [x + 40.0, y, z, np.zeros(x.size)])
    with pytest.raises(SystemExit) as e:
        cli.main(["joint", "--survey", g, "--survey-mag", bad] + JBASE
                 + ["--out", str(tmp_path / "x.html")])
    assert "SAME stations" in str(e.value)


def test_joint_mode_requires_its_extra_inputs(tmp_path):
    g, m = _pair(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["joint", "--survey", g, "--noise", "0.01",
                  "--prior-sd", "0.05", "--corr-len", "400"])
