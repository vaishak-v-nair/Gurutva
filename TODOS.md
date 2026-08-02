
## Landing page follow-ups (from /plan-design-review, 2026-08-05)

- [x] **Host the two verdict reports so the page can link to them.** DONE 2026-08-02. Three of them, at `docs/reports/`, linked from each stat block. The "blocked on: nothing, needs a static host" line sat here after the host already existed — publishing to Vercel and Pages removed the blocker and nobody came back to close it.
- [x] **Verify green-on-tint contrast at 12px.** DONE 2026-08-02, measured not estimated. `--ok #0a7b52` on `--okbg #e8f5ef` = **4.72:1**, clears 4.5. Dark `#5cd6a0` on `#0e271d` = **8.73:1**. All eight audited combinations pass, including the SmartScreen block added the same day (white on `#c42b1c` = 5.66).
- [ ] **Move the page to gurutva.ai when the domain is registered.** The canonical link is now `https://gurutva.vercel.app` (GitHub Pages is a mirror). Registration is still deferred by choice; payment is Vaishak's hands only.

## Engineering follow-ups (from /plan-eng-review + /plan-devex-review, 2026-08-02)

- [ ] **Code-signing certificate.** The only thing that removes the SmartScreen red screen. An OV certificate still needs reputation to accumulate; an EV certificate carries instant SmartScreen reputation and costs more. Everything else on that screen has already been done: version metadata, published SHA-256, VirusTotal link, the two clicks named on the page. Wake: when a real user says the warning stopped them, or before the HN post.
- [ ] **Gate 5's open research question.** Coverage at a fixed realistic truth, versus a prior class declared wide enough to contain the object being searched for. Recorded unresolved in B10; the dark-matter demo is the case that exposed it.
- [ ] **macOS and Linux builds.** `make_exe.py` is written to be platform-agnostic and has never been run anywhere but Windows. The Python path already works everywhere. Wake: a non-Windows user asks.
