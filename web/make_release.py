"""Cut the GitHub release the landing page already points at.

The page links to `releases/latest/download/Gurutva.exe`. That URL is a
promise with a filename in it: GitHub resolves it to whichever release most
recently carried an asset with that exact name. Rename the asset and every
download button on the page 404s, silently, for everyone, with the page still
looking perfectly fine.

So the asset name is checked against the page rather than trusted, the binary
is checked for being the one that was actually verified, and the checksum is
published so a stranger downloading an unsigned 70 MB executable has some way
to tell they got what was uploaded.

Run:  py -3 web/make_release.py --tag v0.1.0
      py -3 web/make_release.py --tag v0.1.0 --publish
"""

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "dist" / "onefile" / "Gurutva.exe"
ZIP = ROOT / "web" / "Gurutva-windows.zip"
PAGE = ROOT / "web" / "index.html"
ASSET = "Gurutva.exe"


def checks():
    """Everything that has to be true before a stranger can download this."""
    problems = []

    if not EXE.exists():
        problems.append(f"no binary at {EXE} — run web/make_exe.py first")
    if not PAGE.exists():
        problems.append("no web/index.html — run web/build_interactive.py")

    if PAGE.exists():
        html = PAGE.read_text(encoding="utf-8")
        want = f"releases/latest/download/{ASSET}"
        if want not in html:
            problems.append(f"the page does not link to {want} — the asset "
                            f"name and the page have drifted apart")
        if "data:application/zip;base64," in html and html.count(
                "releases/latest/download") < 2:
            problems.append("the page still leads with the embedded zip")
        if "Repository private" in html or "access on request" in html:
            problems.append("the page still calls the repository private")

    lic = (ROOT / "LICENSE").read_text(encoding="utf-8", errors="replace")
    if "Apache License" not in lic:
        problems.append("LICENSE is not the Apache-2.0 text")
    if "All rights reserved" in lic:
        problems.append("LICENSE still says all rights reserved")

    # The binary carries LICENSE inside it, and a download that disagrees with
    # the repository about its own terms is worse than no download. The
    # bundled files are compressed, so scanning the .exe for the licence text
    # finds nothing whether it is in there or not — an earlier version of this
    # check did exactly that and would have blocked every release forever.
    # Ask the binary instead.
    if EXE.exists():
        r = subprocess.run([str(EXE), "--where"], capture_output=True,
                           text=True, timeout=600, cwd=EXE.parent)
        said = (EXE.parent / "gurutva-where.log")
        out = said.read_text(encoding="utf-8") if said.exists() else r.stdout
        if "Apache License" not in out:
            problems.append(
                "the .exe reports a licence that is not Apache-2.0 — it was "
                f"built before the licence changed; rebuild. It said: "
                f"{[ln for ln in out.splitlines() if ln.startswith('licence')]}")
        if "MISSING" in out:
            problems.append(f"the .exe cannot find its own bundled files: "
                            f"{[ln for ln in out.splitlines() if 'MISSING' in ln]}")

    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        problems.append(f"working tree is dirty ({len(dirty.splitlines())} "
                        f"files) — commit before tagging")

    return problems


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def notes(tag):
    digest = sha256(EXE)
    (ROOT / "dist" / "SHA256SUMS.txt").write_text(
        f"{digest}  {ASSET}\n", encoding="utf-8")
    return f"""**Download `{ASSET}` below and double-click it.** No Python, no
install, no admin rights. Nothing is uploaded; it runs entirely on your
machine.

Windows will warn you that the publisher is unrecognised. That is a statement
about a code-signing certificate nobody has bought yet, not about the file.
Verify what you got:

```
certutil -hashfile {ASSET} SHA256
{digest}
```

**What it does.** Point it at a four-column CSV (`x,y,z,gz`), answer three
questions about your site, and it writes an HTML report next to your data
saying whether your survey can support the decision you are about to make —
`NOT CLAIMED`, `PROVISIONAL`, or `CLAIMABLE`. If a check fails it refuses to
give you a number. That is the product, not a fault.

**Known limits, stated up front.**

- Windows only. The Python version in the repo runs anywhere.
- Unsigned, so SmartScreen will interrupt you once.
- The single file unpacks itself on every launch, which costs a few seconds
  each time. `Gurutva-windows.zip` is the same build as a folder and starts
  in under a second.
- Nobody has used this on their own survey yet. If it breaks on yours, that
  is the most useful thing that can happen today:
  https://github.com/vaishak-v-nair/Gurutva/issues
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--publish", action="store_true",
                    help="actually create the release (default is dry-run)")
    a = ap.parse_args()

    problems = checks()
    for p in problems:
        print(f"BLOCKED: {p}")
    if problems:
        raise SystemExit(1)

    body = notes(a.tag)
    notes_file = ROOT / "dist" / "RELEASE_NOTES.md"
    notes_file.write_text(body, encoding="utf-8")
    print(f"binary : {EXE} ({EXE.stat().st_size / 1e6:.1f} MB)")
    print(f"sha256 : {sha256(EXE)}")
    print(f"notes  : {notes_file}")

    cmd = ["gh", "release", "create", a.tag,
           f"{EXE}#Gurutva.exe (Windows, no Python needed)",
           f"{ZIP}#Gurutva-windows.zip (same build, starts faster)",
           str(ROOT / "dist" / "SHA256SUMS.txt"),
           "--title", f"Gurutva {a.tag}",
           "--notes-file", str(notes_file)]
    if not a.publish:
        print("\ndry run — would run:\n  " + " ".join(f'"{c}"' for c in cmd))
        return
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"\npublished: https://github.com/vaishak-v-nair/Gurutva/"
          f"releases/tag/{a.tag}")


if __name__ == "__main__":
    main()
