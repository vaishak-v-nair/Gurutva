# Verifiable compute integrity — company plan
### #1 (silent data corruption detection) → #3 (compute attestation)

## Context

Large training runs fail silently. A defective GPU can produce wrong arithmetic without
crashing and without raising an ECC error — Google called these "mercurial cores": they
fail intermittently, on specific instruction/data combinations, often correlated with
temperature and age. The result is a diverging loss curve weeks into a run, or worse, a run
that never visibly diverges and just produces a slightly worse model.

Practice below the top three labs is folklore. Notice a loss spike, roll back to a
checkpoint, empirically swap nodes. Meta's Llama 3 run reported 466 interruptions over 54
days on 16k GPUs, the large majority hardware-related — and that is the *detected* tail.

**This is not novel science, and the plan should not pretend otherwise.** Google (*Cores
That Don't Count*, 2021) and Meta (*Silent Data Corruptions at Scale*, 2021) published on
it and run internal programs with fleet telemetry no startup will match. Algorithm-based
fault tolerance for matrix operations dates to Huang & Abraham, 1984. The opportunity is
that these methods are unevenly applied: everyone below hyperscaler scale has the problem
and none of the tooling.

**#3 is the same primitive, sequenced later.** Both produce verifiable evidence about a
computation you cannot directly observe — #1 asks whether it computed the right thing, #3
asks whether it computed where and how it claims. #3 requires trusted instrumentation
inside the training loop, and #1 is how that instrumentation gets there.

---

## Product: a layered stack, not one detector

The binding design constraint is that nobody accepts 15% overhead always-on. Cheap layers
run continuously and escalate to expensive ones on suspicion.

| Layer | What it does | Target overhead | Catches |
|---|---|---|---|
| **L0** | Always-on invariants: grad-norm/loss anomaly, checksums on collectives | <1% | Gross corruption, NCCL/network path errors |
| **L1** | Sampled deterministic replay of microbatches on a second device | tunable, 1–5% | Anything, within the sampled fraction |
| **L2** | ABFT on GEMM — checksum-encoded operands, verified output | single-digit to low-teens % | Arithmetic defects, with localization |
| **L3** | Offline fleet scan between jobs: known-answer tests | idle time only | Deterministic device defects |

L0 and L3 always on. L1 samples. L2 triggered, never continuous.

**Note the wedge inversion:** L0 is the better product, but **L3 is the easier first sale** —
it runs between jobs, needs no integration into anyone's training loop, and carries no risk
of breaking a live run. It is also the fastest path to finding a real defect. Land with L3,
expand into L0.

---

# Build plan

### Phase 0 — Fault injection harness *(single GPU, no dependency)*

You cannot measure a detector without ground truth, and real SDC is far too rare to develop
against. Everything starts with manufacturing corruption on demand.

**Build:** a seeded injector corrupting values at known rates in known places — GEMM output,
all-reduce payload, optimizer state, activations — with a seeded generator so every
experiment replays bit-identically.

**Gates**
- Injection sequences reproduce bit-identically from a seed, verified by pytest.
- A ~100M-param model trained with injection at a stated rate measurably diverges from
  clean; clean-vs-clean at the same seed is bit-identical.

### Phase 1 — L0 detectors, measured honestly *(single GPU)*

**Build:** the always-on layer. Hook PyTorch and NCCL. Collective checksums plus
distributional invariants on gradient norms and weight deltas.

**Gates**
- Wall-clock overhead **<1%** on a real training loop — *measured, not estimated*.
- Detection rate for injected bit flips reported as a **curve against injection rate**, not
  a single number.
- **≤1 false positive per 24h of clean training.** The gate that decides whether the product
  is usable at all. Alert fatigue kills monitoring products.

### Phase 2 — Localization and confirmation *(single GPU; simulated topology)*

Detection without "which device" is not actionable — the customer's next question is always
"so what do I do."

**Build:** device attribution from L0 signals, plus L1 sampled replay to confirm a suspect
before anyone is asked to pull hardware.

**Gates**
- Correct device identified in ≥90% of injected cases.
- Replay confirmation yields **zero** false confirmations across the clean corpus.

> With single-GPU access, Phases 0–2 are fully achievable — multi-device behaviour is
> simulated within one device plus injected collective payloads. State this limitation
> openly; it is the honest caveat a design partner will probe.

### Phase 3 — Real hardware, real defect *(needs a partner)*

Everything above is preparation. This is the phase that makes it a company, and **its clock
is a business-development clock, not an engineering one.** Start partner conversations
during Phase 1, not after Phase 2.

**Gate:** find **one real SDC in production hardware the operator did not already know
about.** One is sufficient. That single result is the sales motion, the fundraise, and the
proof the injection work generalizes.

---

# Then #3 — Attestation *(gated on #1 deployments)*

Do not start until #1 is deployed somewhere real. The entire advantage is already being
trusted inside the training loop; a standalone attestation startup earns that from scratch.

**The bridge is one sentence.** Root #1's output in hardware attestation — H100 and
Blackwell expose confidential-computing attestation reports — and the artifact stops being
"we found no corruption" and becomes *a signed statement that this run executed on this
hardware, in this configuration, with these integrity checks passing.*

**Build:** (1) signed run manifest — attestation report, code hash, data hash, config, #1's
integrity results; (2) an independent verifier that checks manifests against public keys
without trusting your service.

**Gates**
- A third party holding only the manifest and public keys can confirm a run executed on the
  claimed hardware.
- Any tampered field is detected. No exceptions.

**Buyer:** compliance and legal — a *different* buyer in the same account. Higher ACV, much
slower cycle. Let customers pull it out of you when their compliance org asks where a model
was trained. Poor place to start, strong place to expand.

---

# Competition

**NVIDIA DCGM** — the serious one. Ships free with every GPU; health checks, XID monitoring,
diagnostics r1–r4. But its diagnostics are *offline/between-jobs*, with no live-run
detection, no localization inside a distributed job, and no cross-fleet correlation. Two
things work in your favour: NVIDIA has a structural disincentive to advertise that its chips
silently corrupt, and **the vendor cannot credibly audit its own hardware** — you cannot be
both the accused and the auditor. That neutrality argument is the durable one. Watch their
diagnostics releases anyway; this is the risk that most plausibly ends the company.

**Hyperscaler internal tooling** — non-customers, not competitors. They are a talent pool and
a published-methodology source. Risk: one of them open-sources it (Meta has form). That would
compress the market while validating it, and their tooling would be built for their stack,
not a neocloud's.

**Weights & Biases and ML observability** — the most plausible adjacent threat, because W&B
already owns the in-training-loop integration surface. They monitor *metrics*, not
*correctness* — a different layer today, one acquisition away from not being.

**Datadog / Grafana** — infrastructure metrics. Wrong layer, not a threat.

**Direct startups** — assume someone is doing this and treat it as a diligence item to
resolve before writing code, not an assumption to make either way.

---

# Customers, wedge, pricing

**Not Google or Meta.** They built this and will never buy it.

**First: neoclouds** (CoreWeave, Crusoe, Lambda, Nebius). They have thousands of GPUs, a live
lemons problem with customers complaining about bad nodes, and a reason to *resell*
reliability as differentiation. Land with L3 (zero integration risk), expand to L0.

**Then:** labs outside the top three, sovereign AI programs, enterprises running serious
fine-tunes.

**Pricing anchor — the cost of one lost run.** A 1,000-GPU run for a week at ~$2/GPU-hr is
roughly $340k. Detection that saves one run a year justifies substantial spend. Price
per-GPU-in-fleet for neoclouds; per-cluster or per-run for labs. Seat-based does not fit.
The #3 attestation product prices as a compliance subscription at materially higher ACV.

**Design-partner motion:** free, in exchange for fleet access and a named case study.

---

# Team and hiring

**Your edge is the false-positive gate**, and it is the gate most likely to kill the product.
Deciding whether a noisy telemetry signal is a real defect is hypothesis testing under a
brutally asymmetric loss function — the same problem as asking what a gravity inversion
actually constrains. Own detection statistics personally; it is the part where you are
genuinely differentiated rather than merely competent.

**Honest gaps:** CUDA kernel work, having operated large training runs, and neocloud
relationships. All three are hireable; none are things to pretend at in a partner meeting.

**Sequence — do not hire before Phase 1 gates pass.**
1. **GPU kernel engineer** (CUDA) — for L2 ABFT and low-overhead hooks.
2. **ML infra engineer who has *operated* large runs** — the scarcest and most valuable hire.
   Credibility with buyers comes from having been the buyer.

---

# Funding path

**Phases 0–2 are self-fundable.** A rented GPU is ~$1–2/hr; the whole build to Phase 2 gates
costs less than a used laptop. This is a real strategic advantage — stay unfunded until
capital actually unblocks something, which it does not until cluster time.

| Milestone | Raise | What it buys |
|---|---|---|
| Phase 2 gates + design-partner LOI | Pre-seed, $1–3M | Cluster time, first two hires |
| **Phase 3 — one real SDC found in the wild** | Seed | The fundraise slide. Nothing before it substitutes. |
| 2–3 paying design partners | Series A | GTM |

---

# Timeline

Engineering estimates only; Phase 3 is not an engineering estimate.

- **Phase 0:** weeks
- **Phase 1:** weeks to a couple of months
- **Phase 2:** months — localization is the hard part
- **Phase 3:** gated on partner access, not on code. **Begin outreach during Phase 1.**

---

## Explicitly not doing

- **Thermal / cooling hardware (#2).** Different buyer, sales cycle, and balance sheet.
  Telemetry showing heat correlates with errors confers no advantage in building cold plates.
  If it ever happens it is a separate company, not a roadmap item.
- **Rad-hard silicon or anything orbital.** The space scenario is free upside — radiation
  worsens #1, jurisdictional ambiguity makes #3 urgent — but nothing here depends on it.
- **Selling to hyperscalers.**

## Kill conditions — declared now

- **Determinism unachievable** in real stacks (nondeterministic atomics, kernel selection).
  L1 replay dies, only statistical detection survives, product is much weaker.
- **False-positive floor too high** to clear the Phase 1 gate. Unusable at any price.
- **Real SDC rate at target-customer scale too low to justify spend.** The main commercial
  risk, and not a technical question — a customer losing two runs a year will absorb it.
  Phase 3 answers this, which is why outreach starts early.
- **NVIDIA ships live SDC detection in DCGM** and it is good enough.

## Verification

Phases 0–2 are verifiable on one GPU: every gate a pytest, every number recomputable from a
seed — the same discipline as `gravity-posterior`. Phase 3 cannot be simulated and is the
first point requiring an outside dependency.
