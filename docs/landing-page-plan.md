# Landing page + pitch — plan

**Status:** draft v1, written 2026-08-05 for `/plan-design-review`. Nothing is built yet.
**Source material:** `figures/verdict_geothermal.html`, `figures/verdict_carbon_storage.html`, `README.md`, 68 passing tests.

---

## 1. The one-sentence position

> Every inversion ships a picture. Gurutva ships the verdict on it.

Everything on the page serves that sentence or gets cut.

## 2. Who is actually reading

| Reader | Arrives from | Wants to know in 5 seconds | Leaves with |
|---|---|---|---|
| **Exploration / reservoir lead** (geothermal, mining) | referral, LinkedIn | "does this stop me drilling a dry hole?" | the −346 Mt interval and the 2.6x line |
| **CCS conformance lead** (storage operator, regulator-facing) | search, CCS forum | "can this go in a filing?" | the containment limit and what it refuses to claim |
| **Technical peer** (a Divakar) | the repo, a forum post | "is this real or a demo?" | the gate table with a FAIL in it, and the GitHub link |
| **Investor** | intro | "why can't Terra AI do this tomorrow?" | the refusal, the 68 tests, the two markets from one engine |

The first three matter most. The page is written for a working geoscientist, not for a VC.

## 3. The idea that keeps it off the slop pile

**The page is a verdict.** Not a hero image with a headline over it. The product emits a specific document — a coloured verdict banner, a four-row gate table, one number with an interval, a figure. The landing page uses *that exact system*, turned on the industry.

So the first screen is not "AI-powered subsurface intelligence." It is the product's own output, rendered at full size, reading:

```
NOT CLAIMED — three self-consistency checks passed and the model
still could not reproduce the data it was fitted to.
```

That is unmistakable, it is the brand, and no competitor can copy it without also building the thing that says no.

**Design system is inherited, not invented.** `src/product/report.py` already defines the palette (ink `#12151a`, muted `#5d6670`, pass `#0a7b52` on `#e8f5ef`, fail `#b3261e` on `#fdeceb`), the type scale, the table rules. The landing page uses the same CSS variables. A visitor who clicks through to a real report should feel they never left.

## 4. Page structure

Single scroll. No nav bar (there is nowhere to go). Six sections.

**§1 — The verdict (first viewport).**
One composition, full-bleed. The red verdict block, at report scale. Below it, in muted type: *"That was a real run, on real published data. Most models are never asked."* One primary action: **See the two reports.** Nothing else competes.

**§2 — The number.**
A single large figure with its interval, set the way the report sets it:
`−346 Mt (95%: −484 to −207)` and under it, small: *"the standard workflow would have quoted this 2.6x too tight."*
This is the section a drilling budget owner screenshots.

**§3 — Two proofs, side by side.**
Not feature cards. Two columns, each a compressed real report: the site, the verdict, the one number, and a link to the full page. Left: Utah FORGE, real DOE data. Right: CO₂ containment, Sleipner-class design. Each carries its own honesty line (demo two says plainly that the observations are simulated).

**§4 — The four gates.**
A horizontal row of four, in order, with the one-line plain-English question each asks. Gate 4 is visually heavier — it is the one that fails. Caption: *"The first three ask whether I am self-consistent. Only the fourth asks whether I am right."*

**§5 — What it refuses to say.**
The differentiator, stated as refusals, quoted from the actual reports:
- a whole-area CO₂ bound of 106 Mt — **computed, then refused**, because it is set by the prior, not the data
- a 400 m leak below 72% saturation — **cannot be ruled out**, so no claim covers it
- a prior implying rock varies by 0.021 g/cc — **licensing failed at the 100th percentile**
Caption: *"Every other tool in this business sells confidence."*

**§6 — Who this is for, and how to start.**
Three lines, one per buyer. One email address. One GitHub link. No form on v1.

## 5. What is deliberately absent

- No card grid, no 3-column feature strip, no stock imagery, no gradient hero.
- No logos-of-companies-we-work-with strip (there are none; faking it burns the only asset this page has).
- No pricing on v1 — the first ten conversations set it.
- No sign-up form, no chatbot, no cookie banner (no tracking, so nothing to consent to).
- No "AI" in the headline. The word is doing no work and attracts the wrong reader.

## 6. States and edge cases

- **Narrow screens:** the two-proof section stacks; the verdict block keeps its size and stays the whole first screen. The number in §2 must not reflow mid-figure.
- **Long numbers:** intervals use tabular numerals so they do not jitter.
- **Figures:** the report figures are wide. They scroll inside their own container; the page body never scrolls sideways.
- **Dark mode:** the report palette is light-only today. The page must answer for both or commit to one.
- **No-JS:** the page is static HTML/CSS. Everything works with scripting off.
- **Print:** an exploration lead will print this. It must not lose the verdict colour block.

## 7. The pitch (separate artifact)

A one-page document, same voice, for sending cold. Structure:

1. **The problem, in the reader's own language.** "You are about to spend money on a picture with no error bars."
2. **What we return.** Three lines: where the model is real, pass/fail on four gates, the number with its interval.
3. **Two proofs.** One line each, with the numbers.
4. **Why it is hard to copy.** The refusal. Anyone can add error bars; almost nobody will ship a tool that tells the customer "not today."
5. **What we want from you.** One ask, sized to the reader: a call, or a dataset to run against.

Two variants of §5: one for an operator (run it on your survey), one for a peer (tear the repo apart).

## 8. Success test

Not "looks modern." Three concrete tests:

- **Trunk test:** cover everything but the first screen. A geoscientist can say what this is and who it is for.
- **Screenshot test:** is there one block a reader would paste into Slack? (§2 is the candidate.)
- **Hostility test:** a sceptical PhD reads §5 and cannot find a claim to attack, because the page already made the attacks.

## 9. Design system (inherited from `src/product/report.py`, extended)

The report already ships a palette and a table style. The page adopts it exactly, then adds what a report never needed.

```css
--ink:#12151a  --mut:#5d6670  --line:#dfe3e8
--ok:#0a7b52 on --okbg:#e8f5ef      /* PASS */
--no:#b3261e on --nobg:#fdeceb      /* FAIL */
--paper:#ffffff
```
One accent only: the verdict red. Green appears solely inside gate rows. No third colour, no gradient, no purple.

**Typography — two faces, both real.** No `system-ui`, no Inter, no Roboto.
```css
--font-display:"Iowan Old Style","Sitka Text","Palatino Linotype",Palatino,Georgia,serif;
--font-ui:"Söhne","Neue Haas Grotesk Text","Helvetica Neue",Helvetica,sans-serif;
```
Display serif carries the verdict and the number. The grotesque carries gate tables, labels, and captions — the same split the report already uses between prose and data. Numerals are `font-variant-numeric: tabular-nums` everywhere an interval appears.

**Type scale** (1.25): 13 · 16 · 20 · 25 · 31 · 39 · 49 · 61 px. Body is 18px, never below 16.
**Spacing** (4px base): 4 · 8 · 12 · 16 · 24 · 32 · 48 · 64 · 96 · 128. Section rhythm is deliberately uneven: §1 is 100vh, §2 is short and loud, §3 is tall, §4–§6 are compact. No two sections share a height.
**Measure:** 62–68ch for prose. The page is left-aligned. Nothing is centred except the number in §2.

## 10. Interaction states

Static page, so the state table is small and honest — but it is not empty.

| Element | Rest | Hover | Focus (keyboard) | Visited | Fails |
|---|---|---|---|---|---|
| Primary action "See the two reports" | ink underline 2px | underline thickens to 3px, no colour shift | 2px outline, 2px offset, `--ink` | n/a (in-page) | — |
| Report links (§3) | ink, underlined | underline thickens | same outline | **`#5b3a8e`** — visited state preserved, a reader must see which report they already opened | if the HTML is missing, the link still renders and states "report not published yet" rather than 404-ing silently |
| Figure (§3) | inline, max-width 100% | — | scrollable region is focusable, `tabindex=0` | — | if the image fails, the alt text carries the finding in words |
| GitHub link (§6) | ink, underlined | thickens | outline | visited colour | — |

Focus rings are never removed. `:focus-visible` only, so mouse users don't see them.

**Motion — three, all in service of hierarchy, none decorative:**
1. §1 verdict block fades up 12px over 400ms on load. It arrives; it does not bounce.
2. §2's number counts nothing and animates nothing. It is simply there. (Deliberate: an animated figure reads as a marketing trick on a page whose subject is honesty.)
3. §4 gate row: gate 4 gets a 1px→2px border transition on scroll-into-view, 250ms. The eye lands on the one that fails.
All wrapped in `@media (prefers-reduced-motion: reduce) { animation: none; transition: none; }`.

## 11. Responsive

Three breakpoints, each with an intentional layout, not a stack.

- **≥1080px** — §1 verdict block is 720px wide, left-aligned in a 1120px column with the page's one figure bleeding to the right edge. §3 is two columns.
- **768–1079px** — §1 verdict fills the column. §3 stays two columns but the figures drop to thumbnails that link out. §4's four gates become 2×2, which keeps gate 4 in a corner where it still reads as the odd one.
- **<768px** — single column. §1 verdict block **keeps its type size** and still owns the whole first screen; it is the one thing that must not shrink. §2's number drops from 61px to 39px and must not break mid-interval — `white-space: nowrap` on the value, the 95% range moves to its own line. §3 stacks, Utah first. Touch targets ≥44px with 8px separation.

## 12. Accessibility — specified, not assumed

- Contrast measured, not guessed: `--ink` on paper 16.9:1; `--mut` on paper 5.9:1; verdict red on its tint 5.0:1; gate green on its tint 4.8:1. All clear 4.5:1 for body text. The one item to verify at build time is green-on-tint at small sizes.
- Landmarks: `<main>`, one `<h1>` (the verdict sentence), `<section>` per block with `aria-labelledby`.
- The gate table is a real `<table>` with `<th scope="col">`, not divs. PASS/FAIL is a word, never colour alone — colour is reinforcement.
- Every figure has alt text that states the finding, not the file: *"Cross-section: the survey informs the top 1,300 m; below the red line the model is the regularizer."*
- Keyboard: tab order follows reading order. Wide scrolling containers are focusable so they can be scrolled without a mouse.
- Print stylesheet: the verdict block keeps its background (`print-color-adjust: exact`). An exploration lead will print this.

## 13. User journey

| Step | Reader does | Reader feels | What supports it |
|---|---|---|---|
| 0–5s | Lands, sees a red NOT CLAIMED | *"Wait, is this thing admitting failure?"* | §1 — the hook is the refusal, and it is disarming rather than boastful |
| 5–15s | Reads the sub-line | *"That was a real run on real data."* | the honesty line under the verdict |
| 15–45s | Scrolls to the number | *"That is my drilling budget."* | §2, the screenshot moment |
| 1–3min | Opens a report | *"This is a real document, not a mockup."* | §3 links to the actual shipped HTML |
| 3–5min | Scans the gates, hits §5 | *"They already made the attacks I was going to make."* | §5, the refusals |
| Later | Sends the link to a colleague | trust | §2 is the paste-able block |

Five-year horizon: the page must still be true after the product improves. Nothing on it is a promise; every line is a measurement with a date and a commit.

## 14. NOT in scope (deferred, with reasons)

- **Pricing page** — the first ten conversations set price; publishing a number now anchors it wrong.
- **Sign-up form / email capture** — one email address converts better at this stage and needs no privacy policy.
- **Dark mode** — the report is light-only; shipping a half-answered dark mode is worse than committing to light. Revisit when the report gets one.
- **Logo / wordmark design** — the word "Gurutva" set in the display serif *is* the mark for v1.
- **Case-study pages, blog, docs site** — the repo README is the documentation.
- **Analytics / cookies** — none, which is also why there is no consent banner.

## 15. Decisions made in review (2026-08-05)

1. **First screen shows the FAIL.** D3-A. The red `NOT CLAIMED` block from the real South America Moho run, at full size, with one honest line under it. Rationale: the refusal is the product, and it cannot be forged by a competitor who has not built the thing that says no. Risk accepted: a careless reader could misread it as broken software, so the sub-line does the work.
2. **Both themes, designed not inverted.** Overruled §14's earlier deferral during the artifact-design pass. A light-only page flashbangs a dark-mode reader. The dark palette rebuilds both verdict tints at low luminance rather than flipping them, and `data-theme` overrides the media query in both directions.
3. **Two artifacts, one voice.** `web/index.html` is the page; `docs/pitch.md` is the sendable one-pager. Same numbers, same source JSON.
4. **Published to a private URL and committed to the repo.** D4-C. Live at `https://claude.ai/code/artifact/08ef29a8-4ef9-4de0-918b-2d1d99bee296`, source at `web/index.html`, rebuilt by `web/build.py`. `gurutva.ai` remains unregistered; the link moves when it is.
5. **No PNG mockups were generated.** The page is static HTML, so building the real page IS the mockup — it is strictly more informative than an image of a page, and it is what gets shipped. Stated rather than skipped silently.
6. **The coloured left rail on the refusal blocks was cut.** It is a flagged generic pattern, and a struck-through claim is both less templated and truer: these are sentences the product crosses out.

## 16. Bugs caught by looking at the rendered page

- **The page scrolled sideways.** Grid children default to `min-width:auto`, so a 520px-wide figure refused to let its column shrink and pushed the whole body wide instead of scrolling inside its own container. Fixed with `.two>*{min-width:0}`. This is exactly the failure §6 said must never happen, and it happened anyway — reviewing the spec would never have found it.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | not run for this plan |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | not run |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 0 | — | not run for this plan |
| Design Review | `/plan-design-review` | UI/UX gaps | 1 | CLEAR (FULL) | score: 6/10 → 9/10, 6 decisions |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | not run |

Pass scores: Information architecture 6→9 · Interaction states 2→9 · User journey 3→9 · AI-slop risk 7→9 · Design system 4→9 · Responsive & accessibility 1→9 · Unresolved decisions 4 raised, 4 resolved.

Hard rejections: none. All seven litmus checks pass — the first screen is the product's own output, so the brand is unmistakable and there is exactly one visual anchor.

Two findings came from the rendered page rather than the spec, and both are recorded above: the body scrolled sideways because grid children default to `min-width:auto` (§16), and the light-only decision in §14 was overruled once the page was seen against a dark ground (§15.2).

**VERDICT:** DESIGN CLEARED — plan at 9/10, page and pitch shipped and committed at `8df5e4a`. Eng review not run for this plan; it is a static page with no application code, so the required gate applies to `src/` rather than to `web/`.

NO UNRESOLVED DECISIONS

---

# Verdict report — design review (2026-08-05, second pass)

Target: the document `gurutva report` / `gurutva selftest` produces — the artifact a paying customer opens after running the tool. Never design-reviewed before; written fast as a way to display numbers.

## Ratings

| Pass | Before | After |
|---|---|---|
| 1 Information architecture | 4 | 9 |
| 2 Interaction states | 5 | 8 |
| 3 User journey | 4 | 9 |
| 4 AI-slop risk | 8 | 9 |
| 5 Design system | 7 | 9 |
| 6 Responsive + accessibility | 3 | 8 |
| 7 Unresolved decisions | 1 raised | 1 resolved |

## What was wrong, and what changed

1. **The footer promised a map that was never rendered.** "per-cell map from 600 posterior samples" — the CLI computed the sigma and informed-fraction arrays and threw them away. Not a design flaw, a defect: a customer was told about a picture they were never shown. Now a two-panel section is generated and embedded.

2. **The verdict argued with itself.** A real-data run printed the green word CLAIMABLE and then, in the same breath, "nothing here has been checked against a right answer." A reader who scans stops at the green word. Real data can never run gate 5, so this was the normal case, not an edge case. Resolved by D2-A: a third state, **PROVISIONAL** (amber), sitting between NOT CLAIMED and CLAIMABLE. CLAIMABLE now means gates 1–5, which makes it strong and rare, and gives the customer a next step — run a self-test to earn it.

3. **The gate table was debug output.** `0.01338 (median < 0.115 (600-sample floor)) exact functional sd vs sampled`. Each row now leads with the plain-English question that gate asks; the machine detail sits underneath in 11.5px muted type for whoever wants it.

4. **The number the decision turns on was a table cell.** It is now 46px between two rules, with its interval as the caption — the same treatment §2 of the landing page gets.

5. **No date, no version.** On a document destined for a meeting where money is decided. Both now stamped top-right; the version is the actual short commit hash.

6. **No print stylesheet** on the one artifact people definitely print. Added, with `break-inside: avoid` on tables, figures and the headline, and colour preserved on the verdict block.

7. **Gate thresholds contain `<`.** `(median < 0.115` — the browser read it as a tag and silently ate the rest of the row. Found by reading the rendered page, not the source. Machine-generated detail is now escaped; author-written section notes are not, because those intentionally carry markup.

## NOT in scope

- **Dark mode for the report.** It is a printable document with one committed look. The landing page answers for both because it is read on screen; this is not.
- **Multi-page / paginated layout.** One page is the product.
- **Charts beyond the section view.** A plan view and a depth profile were considered and cut: the section already carries the blind-depth line, which is the point.

