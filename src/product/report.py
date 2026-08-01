"""The deliverable: a one-page verdict a customer hands to a regulator, a
board, or a drilling contractor.

Design rules, all learned the hard way in this repo:
- The verdict sentence comes FIRST, before any figure. A reader who stops
  after one line must still get the honest answer.
- A failed suite prints just as loudly as a passing one, and the numbers are
  stamped DIAGNOSTIC ONLY so they cannot be quoted out of context.
- Every number carries the interval. No bare point estimates anywhere.
- Self-contained HTML with the figure embedded — it survives being emailed,
  which is how these documents actually travel.

Rewritten 2026-08-05 after a design review of a real generated report found
that it read like debug output rather than a document:
- The gate table printed raw internals ("0.01338 (median < 0.115
  (600-sample floor)) exact functional sd vs sampled"). A geologist cannot
  read that. Each gate now leads with the plain-English question it asks,
  and the machine detail sits underneath in small type for whoever wants it.
- The number the whole decision turns on was a table cell. It is now the
  largest thing on the page.
- The footer promised "per-cell map from 600 posterior samples" and no map
  was ever rendered. Callers now pass one.
- No date and no version, on a document destined for a meeting where money
  is decided. Both are stamped in the header now.
- No print stylesheet, on the one artifact people definitely print.
"""

import base64
from datetime import date
from html import escape
from pathlib import Path

# What each gate asks, in the language of the person paying for the answer.
# The raw threshold string stays, but underneath and quieter.
GATE_QUESTION = {
    "licensing": "Could your declared assumptions have produced this data?",
    "calibration": "When it says 95% sure, is it right 95% of the time?",
    "stability": "Run it twice from scratch — same answer?",
    "adequacy": "Does the answer reproduce the data it came from?",
    "recovery": "Does the interval contain a known right answer?",
}

CSS = """
:root{--ink:#12151a;--mut:#5d6670;--line:#dfe3e8;--paper:#fff;
--ok:#0a7b52;--okbg:#e8f5ef;--okln:#b7e0cd;
--no:#b3261e;--nobg:#fdeceb;--noln:#f3c4c0;
--pv:#7a5200;--pvbg:#fdf4e3;--pvln:#e8d3a3;
--display:"Iowan Old Style","Sitka Text","Palatino Linotype",Palatino,Georgia,serif;
--ui:"Söhne","Neue Haas Grotesk Text","Helvetica Neue",Helvetica,sans-serif}
*{box-sizing:border-box}
body{margin:0;padding:44px 46px 64px;background:var(--paper);color:var(--ink);
font:17px/1.6 var(--display);max-width:940px}
.head{display:flex;justify-content:space-between;align-items:baseline;
gap:24px;flex-wrap:wrap;margin-bottom:28px}
h1{font-size:24px;margin:0;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:13px;font-family:var(--ui);margin-top:4px}
.stamp{font-family:var(--ui);font-size:11px;color:var(--mut);
letter-spacing:.05em;text-align:right;line-height:1.7;white-space:nowrap}
.verdict{padding:18px 22px;border-radius:8px;margin-bottom:10px}
.verdict .word{font-family:var(--ui);font-size:12px;letter-spacing:.1em;
font-weight:600;margin-bottom:8px}
.verdict p{margin:0;font-size:18px;line-height:1.45}
.pass{background:var(--okbg);border:1px solid var(--okln);color:var(--ok)}
.prov{background:var(--pvbg);border:1px solid var(--pvln);color:var(--pv)}
.fail{background:var(--nobg);border:1px solid var(--noln);color:var(--no)}
.headline{margin:34px 0 30px;padding:22px 0;border-top:1px solid var(--line);
border-bottom:1px solid var(--line)}
.headline .v{font-size:46px;line-height:1.1;letter-spacing:-.02em;
font-variant-numeric:tabular-nums}
.headline .c{color:var(--mut);margin-top:8px;max-width:60ch}
h2{font-family:var(--ui);font-size:12px;text-transform:uppercase;
letter-spacing:.09em;color:var(--mut);margin:30px 0 10px;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:15px}
td,th{padding:10px 10px;border-bottom:1px solid var(--line);
text-align:left;vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;
letter-spacing:.05em;font-family:var(--ui)}
td.n{font-variant-numeric:tabular-nums}
td.k{width:38%;color:var(--mut)}
.mark{font-family:var(--ui);font-size:11px;font-weight:700;letter-spacing:.06em;
width:52px;white-space:nowrap}
.g{color:var(--ok)}.b{color:var(--no)}
.detail{display:block;font-family:var(--ui);font-size:11.5px;color:var(--mut);
margin-top:4px;font-variant-numeric:tabular-nums}
img{width:100%;border:1px solid var(--line);border-radius:6px;margin-top:6px}
.note{font-size:13px;color:var(--mut);margin-top:8px;max-width:78ch}
footer{margin-top:40px;padding-top:14px;border-top:1px solid var(--line);
font-family:var(--ui);font-size:11.5px;color:var(--mut);line-height:1.7}
code{background:#f4f6f8;padding:1px 5px;border-radius:3px;font-size:12px}
@media print{
  body{padding:0;font-size:11pt;max-width:none}
  .verdict,.mark{-webkit-print-color-adjust:exact;print-color-adjust:exact}
  h2{break-after:avoid}
  table,img,.headline{break-inside:avoid}
  footer{break-before:avoid}
}
"""


def _rows(pairs):
    return "\n".join(
        f"<tr><td class='k'>{k}</td><td class='n'>{v}</td></tr>" for k, v in pairs)


def _gate_rows(gates):
    """One row per gate: the question first, the machine detail underneath."""
    out = []
    for g in gates:
        passed = g.startswith("[PASS]")
        body = g.split("] ", 1)[1]
        name, rest = body.split(":", 1)
        key = next((k for k in GATE_QUESTION if name.startswith(k)), None)
        question = GATE_QUESTION.get(key, name)
        cls, mark = ("g", "PASS") if passed else ("b", "FAIL")
        # ESCAPE. Gate thresholds are machine-generated and routinely contain
        # "<" — "(median < 0.115 ...)" — which a browser reads as the start of
        # a tag and silently eats the rest of the row. Found by reading the
        # rendered page rather than the source. Section notes are NOT escaped:
        # those are author-written and intentionally carry markup.
        detail = escape(f"{name.strip()} · {rest.strip()}")
        out.append(
            f"<tr><td class='mark {cls}'>{mark}</td>"
            f"<td>{escape(question)}<span class='detail'>{detail}</span>"
            f"</td></tr>")
    return "\n".join(out)


def render(path, title, subject, verdict, figure=None, sections=(),
           caption="", footer="", headline=None, version=""):
    """Write the standalone HTML report. Returns the path.

    headline: (value, caption) — the one number the decision turns on. It is
    set at 46px because a table cell is not where you put the number a
    drilling budget depends on.
    """
    cls = {"CLAIMABLE": "pass", "PROVISIONAL": "prov"}.get(
        getattr(verdict, "status", ""), "fail")
    word = getattr(verdict, "status", "NOT CLAIMED")
    body = verdict.headline.split("—", 1)[-1].strip() if "—" in verdict.headline \
        else verdict.headline

    parts = [
        "<div class='head'><div>",
        f"<h1>{title}</h1><div class='sub'>{subject}</div></div>",
        f"<div class='stamp'>{date.today().isoformat()}<br>"
        f"{version or 'gurutva'}</div></div>",
        f"<div class='verdict {cls}'><div class='word'>{word}</div>"
        f"<p>{body}</p></div>",
    ]

    if headline:
        value, cap = headline
        parts.append(f"<div class='headline'><div class='v'>{value}</div>"
                     f"<div class='c'>{cap}</div></div>")

    parts.append("<h2>Checks</h2><table>" + _gate_rows(verdict.gates)
                 + "</table>")

    if verdict.numbers:
        parts.append("<h2>What the data supports</h2><table>"
                     + _rows(verdict.numbers.items()) + "</table>")

    if figure:
        b64 = base64.b64encode(Path(figure).read_bytes()).decode()
        parts.append("<h2>Where the model is real</h2>"
                     f"<img src='data:image/png;base64,{b64}' alt='{caption}'>")
        if caption:
            parts.append(f"<div class='note'>{caption}</div>")

    for head, pairs, note in sections:
        parts.append(f"<h2>{head}</h2><table>{_rows(pairs)}</table>")
        if note:
            parts.append(f"<div class='note'>{note}</div>")

    parts.append(f"<footer>{footer}</footer>")
    html = (f"<!doctype html><meta charset='utf-8'><title>{title}</title>"
            f"<style>{CSS}</style><body>" + "".join(parts))
    Path(path).write_text(html, encoding="utf-8")
    return path
