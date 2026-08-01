"""The deliverable: a one-page verdict the customer can hand to a regulator,
a board, or a drilling contractor.

Design rules, all learned the hard way in this repo:
- The verdict sentence comes FIRST, before any figure. A reader who stops
  after one line must still get the honest answer.
- A failed suite prints just as loudly as a passing one, and the numbers are
  stamped DIAGNOSTIC ONLY so they cannot be quoted out of context.
- Every number carries the interval. No bare point estimates anywhere.
- Self-contained HTML with the figure embedded — it survives being emailed,
  which is how these documents actually travel.
"""

import base64
from pathlib import Path

CSS = """
:root{--ink:#12151a;--mut:#5d6670;--line:#dfe3e8;--ok:#0a7b52;--no:#b3261e;
--okbg:#e8f5ef;--nobg:#fdeceb}
*{box-sizing:border-box}
body{margin:0;padding:40px 44px;font:15px/1.6 -apple-system,Segoe UI,Roboto,
sans-serif;color:var(--ink);max-width:900px}
h1{font-size:22px;margin:0 0 2px;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:13px;margin-bottom:26px}
.verdict{padding:18px 20px;border-radius:8px;font-size:17px;font-weight:600;
margin-bottom:8px;line-height:1.45}
.pass{background:var(--okbg);color:var(--ok);border:1px solid #b7e0cd}
.fail{background:var(--nobg);color:var(--no);border:1px solid #f3c4c0}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;
color:var(--mut);margin:32px 0 10px;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:14px}
td,th{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;
letter-spacing:.05em}
td.n{font-variant-numeric:tabular-nums;white-space:nowrap}
.g{color:var(--ok);font-weight:600}.b{color:var(--no);font-weight:600}
img{width:100%;border:1px solid var(--line);border-radius:6px;margin-top:6px}
.note{font-size:13px;color:var(--mut);margin-top:8px}
footer{margin-top:38px;padding-top:14px;border-top:1px solid var(--line);
font-size:12px;color:var(--mut)}
code{background:#f4f6f8;padding:1px 5px;border-radius:3px;font-size:12.5px}
"""


def _rows(pairs):
    return "\n".join(
        f"<tr><td>{k}</td><td class='n'>{v}</td></tr>" for k, v in pairs)


def _gate_rows(gates):
    out = []
    for g in gates:
        passed = g.startswith("[PASS]")
        body = g.split("] ", 1)[1]
        name, rest = body.split(":", 1)
        cls, mark = ("g", "PASS") if passed else ("b", "FAIL")
        out.append(f"<tr><td class='{cls}'>{mark}</td><td>{name}</td>"
                   f"<td class='n'>{rest.strip()}</td></tr>")
    return "\n".join(out)


def render(path, title, subject, verdict, figure=None, sections=(),
           caption="", footer=""):
    """Write the standalone HTML report. Returns the path."""
    cls = "pass" if verdict.claimable else "fail"
    parts = [f"<h1>{title}</h1><div class='sub'>{subject}</div>",
             f"<div class='verdict {cls}'>{verdict.headline}</div>"]

    parts.append("<h2>Gates</h2><table><tr><th></th><th>Check</th>"
                 f"<th>Result</th></tr>{_gate_rows(verdict.gates)}</table>")

    if verdict.numbers:
        parts.append("<h2>What the data supports</h2><table>"
                     + _rows(verdict.numbers.items()) + "</table>")

    for head, pairs, note in sections:
        parts.append(f"<h2>{head}</h2><table>{_rows(pairs)}</table>")
        if note:
            parts.append(f"<div class='note'>{note}</div>")

    if figure:
        b64 = base64.b64encode(Path(figure).read_bytes()).decode()
        parts.append(f"<h2>Where the model is real</h2>"
                     f"<img src='data:image/png;base64,{b64}'>")
        if caption:
            parts.append(f"<div class='note'>{caption}</div>")

    parts.append(f"<footer>{footer}</footer>")
    html = (f"<!doctype html><meta charset='utf-8'><title>{title}</title>"
            f"<style>{CSS}</style><body>" + "".join(parts))
    Path(path).write_text(html, encoding="utf-8")
    return path
