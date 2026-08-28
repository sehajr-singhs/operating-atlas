"""
Build the standalone results page. Figures are embedded as base64 data URIs
because the artifact CSP blocks every external host.
"""

import base64
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), '..')
WEB = os.path.join(ROOT, 'figs', 'web')
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'site', 'index.html')
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def img(name):
    p = os.path.join(WEB, name)
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        return 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode()


def load(p, default=None):
    f = os.path.join(RES, p)
    return json.load(open(f)) if os.path.exists(f) else default


def figure(name, caption, cls='', label=''):
    src = img(name)
    if not src:
        return ''
    lab = f'<span class="figlabel">{label}</span>' if label else ''
    return f'''<figure class="fig {cls}">
      <div class="figwell"><img src="{src}" alt="{caption[:110]}" loading="lazy"></div>
      <figcaption>{lab}{caption}</figcaption>
    </figure>'''


# ---------------------------------------------------------------------------

def results_table():
    s = load('summary.json', {})
    if not s:
        return '<p class="muted">Sweeps still running.</p>'
    names = {'mono': 'monolith', 'raw': 'raw state', 'invariant': 'invariants (R, Pe)',
             'raw+inv': 'raw + invariants', 'naive': 'Tr V, log det V',
             'activity': 'local activity', 'random': 'random projection'}
    order = ['mono', 'raw', 'invariant', 'raw+inv', 'naive', 'activity', 'random']
    out = []
    for ds, d in s.items():
        rows = []
        ref = d['arms'].get('raw', {}).get('test_mean')
        for a in order:
            if a not in d['arms']:
                continue
            v = d['arms'][a]
            rel = v.get('rel_vs_raw') or '—'
            p = v.get('p_vs_raw') or '—'
            sig = ' sig' if p not in ('—', '') and float(p) < 0.05 else ''
            mark = ' class="ref"' if a == 'raw' else (f' class="{sig.strip()}"' if sig else '')
            rows.append(
                f'<tr{mark}><td>{names[a]}</td>'
                f'<td class="num">{v["test_mean"]:.1f}<span class="pm"> ± {v["test_sd"]:.1f}</span></td>'
                f'<td class="num">{rel}</td><td class="num">{p}</td>'
                f'<td class="num">{v["n"]}</td></tr>')
        out.append(f'''<div class="tablewrap">
      <h4>{ds}</h4>
      <table><thead><tr><th>router input</th><th>test error</th>
      <th>vs raw</th><th>p (paired)</th><th>seeds</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table></div>''')
    return ''.join(out)


def diffusion_table():
    d = load('diffusion_check.json', [])
    if not d:
        return ''
    rows = ''.join(
        f'<tr><td>{r["dataset"]}</td>'
        f'<td class="num">{r["excess_kurtosis"]:.1f}</td>'
        f'<td class="num">{r["tail_excess_ratio"]:,.0f}×</td>'
        f'<td class="num">{r["ks_radial"]:.3f}</td>'
        f'<td class="num">{r["sd"]:.3f}</td></tr>' for r in d)
    return f'''<div class="tablewrap"><table>
      <thead><tr><th>system</th><th>excess kurtosis</th><th>&gt;5σ vs Gaussian</th>
      <th>KS (radial)</th><th>whitened sd</th></tr></thead>
      <tbody>{rows}</tbody></table></div>'''


FINDINGS = [
    ('confirmed', 'The curvature implementation is exact',
     'Reproduces closed forms on the 2-sphere, Poincaré half-plane, torus, '
     'S²×R², flat R⁵ and under nonlinear reparameterisation. Worst error '
     '1.3×10⁻¹⁵.'),
    ('confirmed', 'The switching identity is exact',
     'dF/dr equals the gate-weighted covariance between router logit gradients '
     'and expert outputs, verified to 2.1×10⁻¹⁷ across gate temperatures. It '
     'is an identity, not a bound, and it yields a design rule: transitions '
     'are smooth exactly where experts agree.'),
    ('confirmed', 'Pe is invariant and estimable',
     'Survives a refit in relabelled coordinates at Spearman ρ = 0.999, and '
     'matches ground truth at 0.998.'),
    ('refuted', 'Curvature is not reliably estimable',
     'The coordinate with the most physical meaning is the one finite '
     'telemetry pins down worst: ρ = 0.60 ± 0.06 across the same coordinate '
     'change. R is a second derivative of an estimated diffusion field.'),
    ('null', 'Geometric routing does not improve accuracy',
     'At matched parameters, matched expert bank and matched expert inputs, no '
     'routing coordinate separates from the raw state beyond seed noise. '
     'Controls for dimensionality, basis dependence and cheap local proxies '
     'all run.'),
    ('surprise', 'The diffusion precondition fails on real data, not simulated',
     'Whitened increments on the real motor bench have excess kurtosis 215 and '
     '8,641× the Gaussian rate of 5σ events. The simulated turbofan is far '
     'closer to Gaussian. Real machine telemetry jumps.'),
]


def findings_html():
    kinds = {'confirmed': 'holds', 'refuted': 'fails', 'null': 'no effect',
             'surprise': 'unexpected'}
    return ''.join(
        f'''<li class="finding f-{k}">
          <span class="verdict">{kinds[k]}</span>
          <div><h4>{t}</h4><p>{b}</p></div>
        </li>''' for k, t, b in FINDINGS)


HTML = '''<title>Operating Manifolds</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,380;0,8..60,600;1,8..60,380&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{
  --ground:#F3F5F6; --surface:#FFFFFF; --sunk:#EAEEF0;
  --ink:#161B1F; --body:#2C353B; --muted:#5A6670; --rule:#D9E0E3;
  --hot:#C8571F; --hot-soft:#F4E2D8; --cold:#1B6E78; --cold-soft:#DCEAEC;
  --warn:#8A6A12;
  --maxw:1180px; --text:70ch;
  --f-display:'Archivo','Helvetica Neue',Arial,sans-serif;
  --f-body:'Source Serif 4',Georgia,'Times New Roman',serif;
  --f-mono:'JetBrains Mono','SFMono-Regular',Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#101417; --surface:#171C20; --sunk:#1D2429;
    --ink:#E9EEF0; --body:#C3CDD2; --muted:#8D9BA3; --rule:#283137;
    --hot:#E8843F; --hot-soft:#3A2317; --cold:#4FB3BE; --cold-soft:#123037;
    --warn:#D6AC3C;
  }
}
:root[data-theme="dark"]{
  --ground:#101417; --surface:#171C20; --sunk:#1D2429;
  --ink:#E9EEF0; --body:#C3CDD2; --muted:#8D9BA3; --rule:#283137;
  --hot:#E8843F; --hot-soft:#3A2317; --cold:#4FB3BE; --cold-soft:#123037;
  --warn:#D6AC3C;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--body);
  font-family:var(--f-body); font-size:17px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:var(--maxw); margin:0 auto; padding:0 28px}
h1,h2,h3,h4{font-family:var(--f-display); color:var(--ink); text-wrap:balance; margin:0}
h1{font-size:clamp(2.3rem,5.2vw,3.9rem); font-weight:700; letter-spacing:-.022em; line-height:1.04}
h2{font-size:clamp(1.4rem,2.5vw,1.85rem); font-weight:600; letter-spacing:-.014em}
h3{font-size:1.12rem; font-weight:600; letter-spacing:-.005em}
h4{font-size:.98rem; font-weight:600}
p{margin:0 0 1.05em; max-width:var(--text)}
a{color:var(--cold)}
.mono{font-family:var(--f-mono)}
.muted{color:var(--muted)}

/* ---- masthead ---- */
header.top{border-bottom:1px solid var(--rule); background:var(--surface)}
.top .wrap{padding-top:52px; padding-bottom:44px}
.eyebrow{
  font-family:var(--f-mono); font-size:.7rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--hot); margin-bottom:20px;
}
.standfirst{
  font-size:1.2rem; max-width:64ch; margin-top:22px; color:var(--body);
}
.standfirst strong{color:var(--ink); font-weight:600}
.meta{
  display:flex; flex-wrap:wrap; gap:26px; margin-top:30px;
  font-family:var(--f-mono); font-size:.74rem; color:var(--muted);
  border-top:1px solid var(--rule); padding-top:18px;
}
.meta b{color:var(--ink); font-weight:600}

/* ---- stage spine ---- */
main{padding:0 0 90px}
.stage{border-bottom:1px solid var(--rule); padding:56px 0}
.stage .wrap{display:grid; grid-template-columns:88px 1fr; gap:34px}
.stagenum{
  font-family:var(--f-mono); font-size:.72rem; letter-spacing:.1em;
  color:var(--muted); border-top:2px solid var(--hot); padding-top:10px;
}
.stagenum span{display:block; color:var(--hot); font-weight:600}
@media(max-width:760px){
  .stage .wrap{grid-template-columns:1fr; gap:14px}
  .stagenum{border-top:none; border-left:2px solid var(--hot); padding:0 0 0 12px}
}

/* ---- figures ---- */
.fig{margin:26px 0 8px}
.figwell{
  background:var(--surface); border:1px solid var(--rule); border-radius:3px;
  padding:12px; overflow-x:auto;
}
.figwell img{display:block; width:100%; height:auto; border-radius:2px}
figcaption{
  font-family:var(--f-mono); font-size:.73rem; line-height:1.55;
  color:var(--muted); margin-top:10px; max-width:78ch;
}
.figlabel{color:var(--hot); font-weight:600; margin-right:8px}
.fig.bleed{margin-left:calc(-1 * min(7vw, 90px)); margin-right:calc(-1 * min(7vw, 90px))}
@media(max-width:900px){.fig.bleed{margin-left:0;margin-right:0}}

/* ---- callout ---- */
.callout{
  background:var(--sunk); border-left:3px solid var(--cold);
  padding:20px 24px; margin:26px 0; border-radius:0 3px 3px 0;
}
.callout p:last-child{margin-bottom:0}
.callout.hot{border-left-color:var(--hot)}
.eq{
  font-family:var(--f-mono); font-size:.94rem; color:var(--ink);
  background:var(--sunk); padding:14px 18px; border-radius:3px;
  overflow-x:auto; margin:18px 0; border:1px solid var(--rule);
}

/* ---- tables ---- */
.tablewrap{overflow-x:auto; margin:22px 0}
.tablewrap h4{margin-bottom:8px; font-family:var(--f-mono); font-size:.76rem;
  letter-spacing:.1em; text-transform:uppercase; color:var(--hot)}
table{border-collapse:collapse; width:100%; font-size:.86rem}
th,td{text-align:left; padding:8px 14px 8px 0; border-bottom:1px solid var(--rule)}
th{font-family:var(--f-mono); font-size:.7rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); font-weight:600}
td{font-family:var(--f-body)}
td.num{font-family:var(--f-mono); font-variant-numeric:tabular-nums; font-size:.82rem}
.pm{color:var(--muted)}
tr.ref td{color:var(--ink)}
tr.ref td:first-child::after{content:' — reference'; color:var(--muted);
  font-family:var(--f-mono); font-size:.7rem}
tr.sig td{background:var(--hot-soft)}

/* ---- findings ledger ---- */
ul.findings{list-style:none; padding:0; margin:22px 0 0; display:grid; gap:2px}
.finding{
  display:grid; grid-template-columns:112px 1fr; gap:20px; align-items:start;
  background:var(--surface); border:1px solid var(--rule); padding:18px 20px;
}
.finding h4{margin-bottom:4px}
.finding p{margin:0; font-size:.94rem}
.verdict{
  font-family:var(--f-mono); font-size:.66rem; letter-spacing:.1em;
  text-transform:uppercase; padding:4px 8px; border-radius:2px;
  display:inline-block; font-weight:600; white-space:nowrap;
}
.f-confirmed .verdict{background:var(--cold-soft); color:var(--cold)}
.f-refuted .verdict{background:var(--hot-soft); color:var(--hot)}
.f-null .verdict{background:var(--sunk); color:var(--muted)}
.f-surprise .verdict{background:var(--hot); color:var(--surface)}
@media(max-width:640px){.finding{grid-template-columns:1fr; gap:10px}}

/* ---- systems grid ---- */
.systems{display:grid; grid-template-columns:repeat(auto-fit,minmax(215px,1fr));
  gap:2px; margin:24px 0}
.sys{background:var(--surface); border:1px solid var(--rule); padding:16px 18px}
.sys .kind{font-family:var(--f-mono); font-size:.66rem; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted)}
.sys .name{font-family:var(--f-display); font-weight:600; color:var(--ink);
  font-size:1.05rem; margin:5px 0 6px}
.sys dl{margin:0; font-family:var(--f-mono); font-size:.74rem; color:var(--muted)}
.sys dl div{display:flex; justify-content:space-between; gap:10px; padding:2px 0}
.sys dl b{color:var(--body); font-weight:400}

footer{border-top:1px solid var(--rule); background:var(--surface); padding:36px 0 60px}
footer .wrap{font-family:var(--f-mono); font-size:.74rem; color:var(--muted)}
::selection{background:var(--hot); color:#fff}
:focus-visible{outline:2px solid var(--cold); outline-offset:2px}
@media(prefers-reduced-motion:no-preference){
  .fig img{transition:none}
}
</style>

<header class="top"><div class="wrap">
  <div class="eyebrow">Multi-physics machine learning · research note</div>
  <h1>What the geometry of an operating manifold does, and does not, buy</h1>
  <p class="standfirst">A machine under load moves along a coupled thermal,
  mechanical and electrical trajectory. Routing computation on the
  <strong>local geometry</strong> of that trajectory is an appealing idea with a
  sharp, testable promise. We formalised it, proved the parts that are provable,
  and tested it on five systems. <strong>The central accuracy claim does not
  hold</strong> — and the reason why turned out to be more interesting than the
  claim.</p>
  <div class="meta">
    <div><b>5</b> systems · real + simulated</div>
    <div><b>1.76M</b> telemetry samples</div>
    <div><b>7</b> routing arms, matched capacity</div>
    <div><b>3</b> validated robot platforms</div>
  </div>
</div></header>

<main>

<section class="stage"><div class="wrap">
  <div class="stagenum">Stage<span>00</span></div>
  <div>
    <h2>The machine is not at an operating point. It is on one.</h2>
    <p>A servo joint accelerating a payload dissipates ohmic heat into a winding
    whose rising temperature derates the torque available to the next command —
    on a thermal time constant three decades slower than the control loop that
    issued it. A traction motor's rotor temperature, which no production sensor
    can reach, is a functional of the entire recent history of current and
    speed.</p>
    <p>In both cases the target depends on <em>where</em> the machine sits on a
    coupled multi-physics trajectory, and one monolithic network has to serve
    every part of that trajectory at once. Mixture-of-experts routing is the
    standard answer, and the router almost always reads the raw state. The
    proposal tested here is to route on the geometry instead.</p>
    __FIG_SUBSYS_UR5E__
  </div>
</div></section>

<section class="stage"><div class="wrap">
  <div class="stagenum">Stage<span>01</span></div>
  <div>
    <h2>The volatility is not a feature. It is the metric.</h2>
    <p>The tempting formulation hands a router the state alongside separate
    curvature and volatility features. That double-counts. For a non-degenerate
    diffusion the canonical Riemannian structure on the state space is the
    inverse diffusion tensor —</p>
    <div class="eq">g(z) = V(z)⁻¹ ,&nbsp;&nbsp;&nbsp; V = ΣΣᵀ</div>
    <p>the metric in which Varadhan's short-time heat-kernel asymptotics reduce
    to geodesic distance. The volatility field <em>is</em> the geometry.
    Curvature and volatility are one object, not two.</p>
    <div class="callout">
      <p><strong>This is not just tidier — it is about a hundred times
      cheaper.</strong> Because g comes straight from the fitted Cholesky
      factor, the curvature needs second derivatives of one small network
      instead of third derivatives of a multi-step flow map. Measured cost fell
      from roughly 1.8 s per point to 2 ms.</p>
    </div>
    __FIG_SHAPE__
  </div>
</div></section>

<section class="stage"><div class="wrap">
  <div class="stagenum">Stage<span>02</span></div>
  <div>
    <h2>Two coordinates, both invariant</h2>
    <div class="eq">R(z) = g^{jk} Ric_{jk}
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Pe(z) = ‖a‖²_g = aᵀ V⁻¹ a</div>
    <p><b>R</b> measures how the volatility geometry bends. <b>Pe</b> is a local
    Péclet number — a squared drift-to-noise ratio separating advective
    operating points from noise-dominated ones. Under a smooth relabelling of
    the sensors both are unchanged, which is the sharp promise: a router reading
    invariants keeps making the same decisions after a fleet is recalibrated,
    and a router reading the raw state does not.</p>
    <div class="callout">
      <p><strong>The Stratonovich convention is load-bearing.</strong> It is
      what makes the drift transform by the classical chain rule, and hence
      what makes Pe <em>exactly</em> rather than approximately invariant. The
      fitted Euler–Maruyama drift is an Itô drift and has to be converted.</p>
    </div>
    __FIG_FIELDS__
    __FIG_JOINTINV__
  </div>
</div></section>

<section class="stage"><div class="wrap">
  <div class="stagenum">Stage<span>03</span></div>
  <div>
    <h2>How sharply may an assembly switch?</h2>
    <p>Observing that a softmax of smooth experts is smooth is true and empty.
    The useful statement is an identity:</p>
    <div class="eq">∂F/∂r = (1/τ) · Cov_{k∼p}( ∂u_k/∂r , O_k(x) )</div>
    <p>The assembly's sensitivity to its routing coordinate is exactly the
    gate-weighted covariance between router logit gradients and expert outputs.
    It moves only where the experts <em>disagree</em> and the gate is moving at
    the same time. Smooth handover across an operating boundary is therefore
    bought either with gate temperature or with expert agreement in the overlap
    — and the two are interchangeable at fixed product.</p>
    __FIG_VALIDATION__
  </div>
</div></section>

<section class="stage"><div class="wrap">
  <div class="stagenum">Stage<span>04</span></div>
  <div>
    <h2>Five systems, two physics families</h2>
    <p>Rigid-body dynamics for the robots come from MuJoCo Menagerie, whose
    inertial and actuator parameters are the manufacturers'. What we add is the
    physics Menagerie does not model: a two-node lumped-parameter thermal
    network per joint, with derating and a hysteretic drive trip, so a hot
    winding changes the closed-loop dynamics that heated it.</p>
    __SYSTEMS__
    __FIG_THERMAL__
    <p class="muted">Under identical excitation the platforms differ in thermal
    headroom, and we report that rather than engineering it away: the Panda
    reaches all five regimes including drive trips, the UR5e four, and the
    iiwa14 — oversized for this duty at 0.20 of rated torque — never leaves the
    nominal regime.</p>
  </div>
</div></section>

<section class="stage"><div class="wrap">
  <div class="stagenum">Stage<span>05</span></div>
  <div>
    <h2>Where the axes cross</h2>
    <p>Every pair of operating axes, coloured by the routing invariant. The
    crossings are the problem: a single 2-D shadow of the operating point maps
    to physically different regimes, which is precisely what a router is
    supposed to disentangle.</p>
    __FIG_AXES_PMSM__
    __FIG_AXES_UR5E__
  </div>
</div></section>

<section class="stage"><div class="wrap">
  <div class="stagenum">Stage<span>06</span></div>
  <div>
    <h2>What held, and what did not</h2>
    <ul class="findings">__FINDINGS__</ul>
    <h3 style="margin-top:38px">In-distribution accuracy</h3>
    <p>Every arm shares one expert bank, one training budget and one set of
    expert inputs. Only the router's input differs, so the comparison isolates
    what the routing decision is made <em>on</em>. Three controls run alongside:
    a random projection (does extra router input alone help?), the
    basis-dependent stochastic features (is invariance what matters?), and a
    cheap local activity proxy (does anything local do just as well?).</p>
    __RESULTS__
    __FIG_RESULTS__
  </div>
</div></section>

<section class="stage"><div class="wrap">
  <div class="stagenum">Stage<span>07</span></div>
  <div>
    <h2>The precondition fails on the real machine, not the simulation</h2>
    <p>The whole construction rests on one assumption: increments are locally
    Gaussian with covariance V(z)dt. That is a precondition, not a preference,
    and it is checkable without reference to any downstream accuracy. Whiten
    every held-out increment by its own predicted Cholesky factor and ask
    whether the result looks standard normal.</p>
    __DIFFUSION__
    <div class="callout hot">
      <p><strong>We expected the simulated benchmark to be the one that broke,
      and it was the opposite.</strong> The real motor bench has excess kurtosis
      215 and thousands of times the Gaussian rate of 5σ events; the simulated
      turbofan is close to Gaussian. Real machine telemetry contains genuine
      jumps — commanded setpoint changes, load steps — that a diffusion cannot
      represent. The fitted V absorbs those jumps as if they were local noise,
      so g = V⁻¹ describes a geometry the machine does not have.</p>
    </div>
    <p>This is the most useful practical output of the work: a cheap test a
    practitioner can run <em>before</em> investing in the machinery, and a clear
    statement of when not to.</p>
  </div>
</div></section>

<section class="stage"><div class="wrap">
  <div class="stagenum">Stage<span>08</span></div>
  <div>
    <h2>What survives</h2>
    <p>The construction is mathematically clean and, in its central claim,
    correct: the routing coordinates are exact invariants, the metric is
    canonical rather than arbitrary, and the switching behaviour obeys an exact
    identity with a real design consequence. None of that translated into better
    in-distribution predictions on any of five systems.</p>
    <p>The honest reading is threefold. The information a geometric router
    extracts is largely already available to one reading the state directly,
    because the geometry <em>is</em> a deterministic function of that state —
    invariance changes how the information is packaged, not how much of it there
    is. The coordinate carrying the most physical meaning, the curvature, is the
    one finite telemetry estimates worst, and that instability propagates into
    the routing decision. And the promised advantage is conditional: invariance
    only pays when the coordinate change is severe enough to actually break a
    raw-state router.</p>
    <p>What remains is narrower and still useful — a correct formulation
    unifying curvature and volatility at a hundredth of the cost, an exact
    switching identity, a cheap and robustly estimable Pe statistic, and a
    testable precondition with a demonstration of what happens when it fails.
    The negative result is worth recording, because the intuition behind
    geometric routing is strong enough that it will be proposed again.</p>
  </div>
</div></section>

</main>

<footer><div class="wrap">
  Data: PMSM electric motor temperature bench and NASA C-MAPSS (both public on
  Kaggle); robot models from MuJoCo Menagerie. All figures generated from the
  released code. Statistics are paired over seeds; error bars are standard
  deviations across seeds.
</div></footer>
'''


def systems_html():
    S = [('real', 'PMSM bench', [('samples', '1,330,816'), ('rate', '2 Hz'),
                                 ('state', '8-d'), ('physics', 'EM–thermal–mech')]),
         ('simulated', 'C-MAPSS FD004', [('samples', '61,249'), ('regimes', '6'),
                                         ('state', '17-d'), ('task', 'RUL')]),
         ('sim + LPTN', 'UR5e', [('samples', '431,520'), ('joints', '6'),
                                 ('state', '32-d'), ('regimes hit', '4 of 5')]),
         ('sim + LPTN', 'Franka Panda', [('samples', '431,520'), ('joints', '7'),
                                         ('state', '37-d'), ('regimes hit', '5 of 5')]),
         ('sim + LPTN', 'KUKA iiwa14', [('samples', '431,520'), ('joints', '7'),
                                        ('state', '37-d'), ('regimes hit', '2 of 5')])]
    out = []
    for kind, name, rows in S:
        dl = ''.join(f'<div><span>{k}</span><b>{v}</b></div>' for k, v in rows)
        out.append(f'<div class="sys"><div class="kind">{kind}</div>'
                   f'<div class="name">{name}</div><dl>{dl}</dl></div>')
    return f'<div class="systems">{"".join(out)}</div>'


def main():
    h = HTML
    subs = {
        '__FIG_SUBSYS_UR5E__': figure(
            'f_subsys_ur5e.jpg',
            'One UR5e episode. Kinematic and mechanical channels oscillate at '
            '~1 Hz while joint 1\'s winding heats over minutes, crossing the '
            'derating knee at t ≈ 90 s and flipping the ground-truth regime. '
            'The three-decade time-scale separation is the multi-physics '
            'coupling, and the regime strip is never shown to any model.',
            label='Fig 1'),
        '__FIG_SHAPE__': figure(
            'f_shape_pmsm.jpg',
            'The PMSM operating manifold in chart coordinates, under three '
            'colourings: scalar curvature, log(1+Pe), and step size. The dense '
            'slab is steady operation; the spikes are fast excursions.',
            label='Fig 2'),
        '__FIG_FIELDS__': figure(
            'f_fields_pmsm.jpg',
            'The two invariants as fields on 2-D slices of the chart. This is '
            'the routing coordinate system itself.', label='Fig 3'),
        '__FIG_JOINTINV__': figure(
            'f_jointinv_pmsm.jpg',
            'Are R and Pe two coordinates or one dressed twice? Their joint '
            'density, and the distribution of operating points across the '
            'advective/diffusive boundary at Pe = 1.', label='Fig 4'),
        '__FIG_VALIDATION__': figure(
            'f_validation.jpg',
            'Left: cost of the exact invariants against chart dimension — '
            'geometry is affordable only on a low-dimensional chart, which is '
            'what a chart is for. Centre and right: the predicted 1/τ scaling '
            'of routing sensitivity, and the identity residual sitting at '
            'double-precision epsilon.', label='Fig 5'),
        '__FIG_THERMAL__': figure(
            'f_thermal_platforms.jpg',
            'Thermal headroom against normalised torque, coloured by '
            'ground-truth regime, with the derating knee and trip threshold '
            'marked.', label='Fig 6'),
        '__FIG_AXES_PMSM__': figure(
            'f_axes_pmsm.jpg',
            'PMSM: all pairwise operating-axis crossovers, coloured by the '
            'drift-to-noise invariant, with subsystem-coded marginals on the '
            'diagonal and rank correlations above it.',
            cls='bleed', label='Fig 7'),
        '__FIG_AXES_UR5E__': figure(
            'f_axes_ur5e.jpg',
            'UR5e per-joint sub-axes. Top: mechanical phase portrait. Middle: '
            'torque–speed envelope against the datasheet limit. Bottom: the '
            'ohmic heating law, τ² against winding temperature, with the '
            'derating and trip thresholds. Colour is winding temperature.',
            cls='bleed', label='Fig 8'),
        '__FIG_RESULTS__': figure(
            'f_results.jpg',
            'Test error by routing arm, normalised to the raw-state router.',
            label='Fig 9'),
        '__SYSTEMS__': systems_html(),
        '__FINDINGS__': findings_html(),
        '__RESULTS__': results_table(),
        '__DIFFUSION__': diffusion_table(),
    }
    for k, v in subs.items():
        h = h.replace(k, v or '')
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(h)
    print(f'wrote {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB)')


if __name__ == '__main__':
    main()
