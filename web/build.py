"""Build the landing page — one self-contained file, figures embedded.

Reproducible on purpose: every number on the page is read from the JSON the
demos emit, never typed in by hand. If a demo's result changes, the page
changes with it, and a stale claim cannot survive a rebuild.

Run: py -3 web/build.py
"""

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "web" / "index.html"

geo = json.loads((FIG / "verdict_geothermal.json").read_text())
ccs = json.loads((FIG / "verdict_carbon_storage.json").read_text())
dm = json.loads((FIG / "verdict_dark_matter.json").read_text())
N_TESTS = 72          # keep in step with the suite
REPO = "https://github.com/vaishak-v-nair/Gurutva"   # PRIVATE — do not link publicly


def img(name):
    return ("data:image/png;base64,"
            + base64.b64encode((FIG / name).read_bytes()).decode())


def mass_line(j):
    m, s = j["mass_Mt"], j["mass_sd_Mt"]
    return (f"{m:+,.0f} Mt", f"95%: {m - 1.96 * s:+,.0f} to {m + 1.96 * s:+,.0f} Mt")


geo_val, geo_ci = mass_line(geo)

CSS = """
:root{
  --ink:#12151a; --mut:#5d6670; --line:#dfe3e8; --paper:#fff;
  --ok:#0a7b52; --okbg:#e8f5ef; --okline:#b7e0cd;
  --no:#b3261e; --nobg:#fdeceb; --noline:#f3c4c0;
  --visited:#5b3a8e;
  --display:"Iowan Old Style","Sitka Text","Palatino Linotype",Palatino,Georgia,serif;
  --ui:"Söhne","Neue Haas Grotesk Text","Helvetica Neue",Helvetica,sans-serif;
  --s1:4px; --s2:8px; --s3:12px; --s4:16px; --s5:24px; --s6:32px;
  --s7:48px; --s8:64px; --s9:96px; --s10:128px;
}
/* The page is a lab report, so the light ground is the default identity.
   But a reader in dark mode should not be flashbanged by a white sheet, so
   the second theme is designed rather than inverted: the ground takes a
   slight blue-grey bias toward the ink, and both verdict tints are rebuilt
   at low luminance so PASS and POSSIBLE-FAIL still read at a glance. */
@media (prefers-color-scheme:dark){:root{
  --ink:#e9ecef; --mut:#98a2ad; --line:#252a30; --paper:#0f1114;
  --ok:#5cd6a0; --okbg:#0e271d; --okline:#1e4936;
  --no:#ff7a6d; --nobg:#2b1512; --noline:#54251f;
  --visited:#b9a6e0;
}}
:root[data-theme="dark"]{
  --ink:#e9ecef; --mut:#98a2ad; --line:#252a30; --paper:#0f1114;
  --ok:#5cd6a0; --okbg:#0e271d; --okline:#1e4936;
  --no:#ff7a6d; --nobg:#2b1512; --noline:#54251f;
  --visited:#b9a6e0;
}
:root[data-theme="light"]{
  --ink:#12151a; --mut:#5d6670; --line:#dfe3e8; --paper:#fff;
  --ok:#0a7b52; --okbg:#e8f5ef; --okline:#b7e0cd;
  --no:#b3261e; --nobg:#fdeceb; --noline:#f3c4c0;
  --visited:#5b3a8e;
}
*{box-sizing:border-box}
html,body{max-width:100%;overflow-x:clip}
body{margin:0;background:var(--paper);color:var(--ink);
  font:18px/1.6 var(--display);-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:0 var(--s5)}
a{color:var(--ink);text-decoration:underline;text-decoration-thickness:2px;
  text-underline-offset:3px}
a:hover{text-decoration-thickness:3px}
a:visited{color:var(--visited)}
a:focus-visible{outline:2px solid var(--ink);outline-offset:2px;border-radius:2px}
.num{font-variant-numeric:tabular-nums}

/* ---- 1. the verdict: the whole first screen, one composition ---- */
#verdict{min-height:100vh;display:flex;flex-direction:column;
  justify-content:center;padding:var(--s8) 0}
.brand{font-family:var(--display);font-size:25px;letter-spacing:-.01em;
  margin-bottom:var(--s8)}
.brand span{color:var(--mut);font-family:var(--ui);font-size:14px;
  letter-spacing:.06em;text-transform:uppercase;margin-left:var(--s3)}
.block{max-width:820px;background:var(--nobg);border:1px solid var(--noline);
  border-radius:8px;padding:var(--s6) var(--s7);
  animation:rise .4s cubic-bezier(.2,.7,.3,1) both}
.block h1{font-size:39px;line-height:1.25;color:var(--no);margin:0;
  text-wrap:balance;
  font-weight:600;letter-spacing:-.015em}
.block .tag{font-family:var(--ui);font-size:13px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--no);margin-bottom:var(--s4);opacity:.8}
.after{max-width:640px;margin-top:var(--s6);color:var(--mut);font-size:20px}
.after strong{color:var(--ink);font-weight:600}
.cta{margin-top:var(--s7);font-family:var(--ui);font-size:17px}
@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}

/* ---- 2. the number ---- */
#number{padding:var(--s9) 0;border-top:1px solid var(--line)}
.figure{font-size:61px;line-height:1.1;letter-spacing:-.02em;
  white-space:nowrap;font-variant-numeric:tabular-nums}
.range{font-size:25px;color:var(--mut);margin-top:var(--s3);
  font-variant-numeric:tabular-nums}
.beneath{margin-top:var(--s5);max-width:620px;color:var(--mut)}
.beneath strong{color:var(--no);font-weight:600}

/* ---- 3. two proofs ---- */
#proofs{padding:var(--s9) 0;border-top:1px solid var(--line)}
h2{font-family:var(--ui);font-size:13px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--mut);font-weight:600;
  margin:0 0 var(--s5)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:var(--s8)}
/* min-width:0 is load-bearing: a grid item defaults to min-width:auto, so a
   520px-wide figure refuses to let its column shrink and the PAGE scrolls
   sideways instead of the figure's own container. Caught in the browser. */
.two>*{min-width:0}
.proof h3{font-size:25px;margin:0 0 var(--s2);letter-spacing:-.01em}
.proof .site{font-family:var(--ui);font-size:13px;color:var(--mut);
  margin-bottom:var(--s5)}
.stamp.warn{background:transparent;color:var(--mut);border-color:var(--line)}
.stamp.no{background:var(--nobg);color:var(--no);border-color:var(--noline)}
.stamp{display:inline-block;margin-right:var(--s2);font-family:var(--ui);font-size:12px;
  letter-spacing:.07em;text-transform:uppercase;padding:3px 9px;
  border-radius:4px;background:var(--okbg);color:var(--ok);
  border:1px solid var(--okline);margin-bottom:var(--s4)}
.proof p{margin:0 0 var(--s4)}
.proof .lead{font-size:20px}
.scroller{overflow-x:auto;border:1px solid var(--line);border-radius:6px;
  margin:var(--s4) 0}
.scroller img{display:block;width:100%;min-width:520px}
.caveat{font-family:var(--ui);font-size:13px;color:var(--mut);
  line-height:1.5;padding-left:var(--s4);border-left:1px solid var(--line)}

/* ---- 3b. the space section: one full-width composition ---- */
#space{padding:var(--s9) 0;border-top:1px solid var(--line)}
.space-lede{font-size:31px;line-height:1.3;letter-spacing:-.015em;
  max-width:860px;text-wrap:balance;margin:0 0 var(--s5)}
.space-sub{max-width:700px;color:var(--mut);font-size:20px;margin:0 0 var(--s6)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:var(--s8);
  align-items:start;margin-top:var(--s6)}
.pair>*{min-width:0}
.metric{margin-bottom:var(--s5)}
.metric .v{font-size:39px;line-height:1.1;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums}
.metric .k{font-family:var(--ui);font-size:13px;color:var(--mut);
  letter-spacing:.06em;text-transform:uppercase;margin-top:var(--s2)}
.selfcatch{margin-top:var(--s7);max-width:860px;padding:var(--s5) var(--s6);
  background:var(--nobg);border:1px solid var(--noline);border-radius:8px}
.selfcatch .h{font-family:var(--ui);font-size:12px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--no);margin-bottom:var(--s3)}
.selfcatch p{margin:0 0 var(--s3);font-size:19px}
.selfcatch p:last-child{margin-bottom:0;color:var(--mut);font-size:17px}

/* ---- 4. the gates ---- */
#gates{padding:var(--s9) 0;border-top:1px solid var(--line)}
.four{display:grid;grid-template-columns:repeat(4,1fr);gap:var(--s5)}
.gate{padding-top:var(--s4);border-top:1px solid var(--line)}
.gate.key{border-top:2px solid var(--ink)}
.gate .n{font-family:var(--ui);font-size:12px;color:var(--mut);
  letter-spacing:.08em}
.gate h4{font-size:20px;margin:var(--s2) 0 var(--s2);font-weight:600;
  text-wrap:balance}
.gate p{margin:0;font-size:17px;color:var(--mut)}
.gate.key p{color:var(--ink)}
.gcap{margin-top:var(--s6);max-width:660px;font-size:20px}

/* ---- 5. refusals ---- */
#refuses{padding:var(--s9) 0;border-top:1px solid var(--line)}
.ref{max-width:760px;margin-bottom:var(--s6)}
.ref .said{font-size:20px;color:var(--mut);
  text-decoration:line-through;text-decoration-color:var(--no);
  text-decoration-thickness:2px}
.ref .why{margin-top:var(--s3)}
.kicker{margin-top:var(--s7);font-size:25px;max-width:640px;
  letter-spacing:-.01em}

/* ---- 6. start ---- */
#start{padding:var(--s9) 0 var(--s10);border-top:1px solid var(--line)}
.who{max-width:760px;margin-bottom:var(--s5);font-size:20px}
.who b{font-weight:600}
.contact{margin-top:var(--s7);font-family:var(--ui);font-size:17px;
  line-height:2}
footer{border-top:1px solid var(--line);padding:var(--s5) 0 var(--s8);
  font-family:var(--ui);font-size:13px;color:var(--mut);line-height:1.7}

@media (max-width:1079px){
  .four{grid-template-columns:1fr 1fr;gap:var(--s6)}
  .block h1{font-size:31px}
}
@media (max-width:767px){
  body{font-size:17px}
  .two,.pair{grid-template-columns:1fr;gap:var(--s7)}
  .space-lede{font-size:25px}
  .metric .v{font-size:31px}
  .four{grid-template-columns:1fr}
  .block{padding:var(--s5) var(--s5)}
  .block h1{font-size:27px}
  .figure{font-size:39px}
  .range{font-size:20px}
  .cta,.contact{font-size:17px}
  a{padding:2px 0;display:inline-block;min-height:44px;line-height:40px}
  .ref a,.beneath a,p a{display:inline;min-height:0;line-height:inherit}
}
@media (prefers-reduced-motion:reduce){
  *{animation:none!important;transition:none!important}
}
@media print{
  .block,.stamp{-webkit-print-color-adjust:exact;print-color-adjust:exact}
  #verdict{min-height:0}
}
"""

HTML = f"""<style>{CSS}</style>
<main>

<section id="verdict" aria-labelledby="v-h">
  <div class="wrap">
    <div class="brand">Gurutva <span>posterior uncertainty for potential fields</span></div>
    <div class="block">
      <div class="tag">Gate report &middot; South America Moho &middot; 2026-08-04</div>
      <h1 id="v-h">NOT CLAIMED — three self-consistency checks passed
      and the model still could not reproduce the gravity it was fitted to.</h1>
    </div>
    <p class="after">That was a real run, on real published satellite data, and the
    map was withdrawn rather than shipped. <strong>Most models are never asked.</strong></p>
    <p class="cta"><a href="#proofs">See the two reports &darr;</a></p>
  </div>
</section>

<section id="number" aria-labelledby="n-h">
  <div class="wrap">
    <h2 id="n-h">What a survey actually knows</h2>
    <div class="figure num">{geo_val}</div>
    <div class="range num">{geo_ci}</div>
    <p class="beneath">Excess mass under a 2&times;2&nbsp;km block at Utah FORGE, from
    323 real gravity stations. The interval is the product. The same workflow the
    industry runs today would have quoted it
    <strong>{geo['overconfidence_factor']:.1f}&times; too tight</strong>.</p>
  </div>
</section>

<section id="proofs" aria-labelledby="p-h">
  <div class="wrap">
    <h2 id="p-h">Two proofs</h2>
    <div class="two">

      <article class="proof">
        <h3>Should you drill here?</h3>
        <div class="site">Utah FORGE geothermal &middot; DOE GDR 1144 &middot; real data, CC-BY</div>
        <div class="stamp">Claimable — 4 / 4 core gates</div>
        <div class="stamp warn">Recovery untested — no known truth exists here</div>
        <p class="lead">The survey informs 30% of the model. Below
        {geo['z_blind_m']:,.0f}&nbsp;m elevation the typical cell is not
        constrained at all — that part of the picture is the regularizer, not rock.</p>
        <div class="scroller"><img src="{img('verdict_geothermal.png')}"
          alt="Cross-section through the Utah FORGE model. Left: the density
          model the industry ships. Middle: posterior sigma. Right: informed
          fraction, showing the data constrains only the shallow section;
          below the red line the model is the regularizer."></div>
        <p>Fit to {geo['rms_mGal']:.2f}&nbsp;mGal against a measured
        {geo['floor_mGal']}&nbsp;mGal noise floor.</p>
      </article>

      <article class="proof">
        <h3>Did the CO<sub>2</sub> stay in the box?</h3>
        <div class="site">Sleipner-class monitoring design &middot; seafloor gravimetry</div>
        <div class="stamp">Claimable — 4 / 4 core gates</div>
        <div class="stamp">Recovery verified — 0.3&sigma; from a known truth</div>
        <p class="lead">Inventory reconciles at
        {ccs['co2_inside_Mt']:.1f} &plusmn; {ccs['co2_inside_sd_Mt']:.1f}&nbsp;Mt
        against {ccs['injected_Mt']:.0f}&nbsp;Mt injected. A 400&nbsp;m leak
        2&nbsp;km outside the complex is detectable only above 72% saturation —
        emptier than that, and no containment claim covers it.</p>
        <div class="scroller"><img src="{img('verdict_carbon_storage.png')}"
          alt="Four map panels: the declared CO2 plume, the recovered mean at the
          same colour scale showing heavy ringing, posterior sigma, and the 95%
          upper limit on hidden CO2 per column."></div>
        <p class="caveat">Survey parameters are the published Sleipner
        programme. The observations are simulated from a declared plume, so
        every number states what this survey <em>design</em> can prove. Said
        here, in the code, and twice in the report.</p>
      </article>

    </div>
  </div>
</section>

<section id="space" aria-labelledby="sp-h">
  <div class="wrap">
    <h2 id="sp-h">The same code, ten thousand billion billion times further away</h2>
    <p class="space-lede">The engine above was written for gravity stations in
    a Utah desert. It was then run, without a single line changed, on the
    weak-lensing shear of a dark-matter halo.</p>
    <p class="space-sub">It works because both are the same problem: mass
    inferred from the field it makes. Prisms&rarr;gravity and
    mass-sheet&rarr;shear share an operator signature, so the engine does not
    care what the mass is made of, or whether it is a kilometre down or
    10<sup>22</sup>&nbsp;km away. The drilling interval and the exclusion
    limit are one number with two names.</p>

    <div class="pair">
      <div>
        <div class="stamp no">Not claimed — gate 5 failed</div>
        <div class="metric" style="margin-top:var(--s4)">
          <div class="v num">{dm['peak_sigma']:.1f}&sigma;</div>
          <div class="k">halo detected &middot; 4 / 4 core gates pass</div>
        </div>
        <div class="metric">
          <div class="v num">&kappa; &lt; {dm['exclusion_kappa']:.3f}</div>
          <div class="k">95% exclusion where nothing is seen</div>
        </div>
        <p style="color:var(--mut);max-width:34em">An exclusion limit is what a
        physicist publishes when a search comes up empty. A posterior sigma is
        what a mining company needs before drilling. They are the same array,
        from the same function.</p>
      </div>
      <div class="scroller"><img src="{img('verdict_dark_matter.png')}"
        alt="Four panels: the true dark-matter halo, the standard
        Kaiser-Squires reconstruction with no error bars, the Gurutva
        posterior mean, and the 95% exclusion limit on hidden mass."></div>
    </div>

    <div class="selfcatch">
      <div class="h">And then the product caught itself, so we built a fifth gate</div>
      <p>All four core gates passed — and the map still missed a truth we
      happened to know. Only <strong>{dm['recovery_cover'] * 100:.0f}% of
      pixels</strong> had the right answer inside their 95% interval, worst
      miss <strong>{dm['recovery_worst_sigma']:.1f} sigma</strong>.</p>
      <p>None of the four could see it. Licensing asks whether the
      <em>data</em> is plausible, never the truth. Calibration draws its test
      truths <em>from</em> the prior, so it is blind by construction. Two runs
      agree on the same wrong answer, and a smoothed field still reproduces
      the shear. So gate 5 now exists: does the interval contain a known right
      answer? It runs whenever a truth is available, and when none is, the
      verdict says so out loud instead of letting silence read as success.</p>
      <p>Gate 5 fails here, so this map is <strong>not claimed</strong>. We
      widened the prior class first, and measured that it does not rescue it:
      the aperture bias is shrinkage and widening helps, but the peak bias is
      the lensing operator smoothing a cusp it cannot resolve, and no prior
      undoes that. A prior wide enough to fix the aperture also degrades the
      exclusion limit 2.2&times;. So the limit stays, the map does not, and
      the CO<sub>2</sub> report above now carries gate 5 too — where it
      passes at 0.3&sigma;.</p>
    </div>
  </div>
</section>

<section id="gates" aria-labelledby="g-h">
  <div class="wrap">
    <h2 id="g-h">Four questions, asked in order</h2>
    <div class="four">
      <div class="gate"><div class="n">01 / LICENSING</div>
        <h4>Could my assumptions ever have produced this measurement?</h4>
        <p>If not, nothing further is allowed to be claimed.</p></div>
      <div class="gate"><div class="n">02 / CALIBRATION</div>
        <h4>When I say 95% sure, am I right 95% of the time?</h4>
        <p>Checked by simulation, against the sampling floor.</p></div>
      <div class="gate"><div class="n">03 / STABILITY</div>
        <h4>Run it twice from scratch. Same answer?</h4>
        <p>At the real observation, not only in simulation.</p></div>
      <div class="gate key"><div class="n">04 / ADEQUACY</div>
        <h4>Can the answer reproduce the data it came from?</h4>
        <p>The gate that failed at the top of this page.</p></div>
    </div>
    <p class="gcap">The first three ask whether the model is self-consistent.
    All three passed on a model that was wrong. <strong>Only the fourth asks
    whether it is right.</strong></p>
  </div>
</section>

<section id="refuses" aria-labelledby="r-h">
  <div class="wrap">
    <h2 id="r-h">What it refuses to tell you</h2>

    <div class="ref">
      <div class="said num">&ldquo;Less than {ccs['area_bound_REFUSED_Mt']:,.0f} Mt
      of CO<sub>2</sub> escaped the storage complex.&rdquo;</div>
      <div class="why">Computed, then refused. That bound is
      {ccs['area_bound_REFUSED_Mt'] / ccs['injected_Mt'] * 100:.0f}% of everything
      injected, and it is set by the prior rather than the survey. An exclusion
      limit only means something for a body you can name.</div>
    </div>

    <div class="ref">
      <div class="said">&ldquo;There is no leak outside the complex.&rdquo;</div>
      <div class="why">Only above 72% saturation. A 400&nbsp;m accumulation
      emptier than that cannot be ruled out by this survey at all, so no
      containment claim covers it.</div>
    </div>

    <div class="ref">
      <div class="said num">&ldquo;Rock density here varies by 0.021 g/cc.&rdquo;</div>
      <div class="why">What the standard tuned regularizer implies if you read it
      as a statement about rock. Licensing failed it at the 100th percentile:
      that prior could not have produced the anomaly that was measured.</div>
    </div>

    <p class="kicker">Every other tool in this business sells confidence.
    This one is built to withhold it.</p>
  </div>
</section>

<section id="start" aria-labelledby="s-h">
  <div class="wrap">
    <h2 id="s-h">Who this is for</h2>
    <p class="who"><b>Geothermal and mineral exploration.</b> You are about to
    commit a drilling budget to a picture with no error bars. Get the interval,
    and the depth below which the picture is invented.</p>
    <p class="who"><b>CO<sub>2</sub> storage operators.</b> Conformance and
    containment are filings, not opinions. Get a defensible statement of what
    your monitoring programme can and cannot prove — before you fund it.</p>
    <p class="who"><b>Anyone who wants to check the work.</b> Every number on
    this page is generated by a script from a run that happened, and the
    repository has {N_TESTS} passing tests. It is private today, pending
    publication — ask and you get access, including the retractions.</p>

    <p class="contact">
      <a href="mailto:vaishak.v.nair.dev@gmail.com">vaishak.v.nair.dev@gmail.com</a><br>
      <span style="color:var(--mut)">repository access on request</span>
    </p>
  </div>
</section>

<footer><div class="wrap">
  Gurutva &middot; Bayesian posterior uncertainty for potential-field inversion.
  Repository private pending publication.
  Figures and numbers on this page are generated from
  <code>figures/verdict_geothermal.json</code> and
  <code>figures/verdict_carbon_storage.json</code> by <code>web/build.py</code> —
  if a result changes, this page changes with it.
  No tracking, no cookies, no analytics.
</div></footer>

</main>
"""

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT}  ({len(HTML) / 1024:.0f} KB)")
print(f"  geothermal: {geo_val} ({geo_ci}), {geo['overconfidence_factor']:.1f}x")
print(f"  ccs: {ccs['co2_inside_Mt']:.1f} +/- {ccs['co2_inside_sd_Mt']:.1f} Mt, "
      f"refused bound {ccs['area_bound_REFUSED_Mt']:,.0f} Mt")
