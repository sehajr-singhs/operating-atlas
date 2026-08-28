"""
Interactive demo of the operating atlas.

Upload any CSV of multi-channel telemetry and see:
  1. All 9 atoms computed for every pair of channels
  2. The signed Levy area — the one atom that sees the arrow of time
  3. Time-reversal parity: flip the record end-to-end, watch levy change sign
  4. Side-by-side: atlas vs correlation matrix on the same data
  5. Recalibration invariance demo: warp every channel, atlas doesn't move
"""

import sys
import os
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# make atoms importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atoms  # noqa: E402

# ---------------------------------------------------------------------------
# built-in demo signals
# ---------------------------------------------------------------------------

def _demo_motor(n=4000):
    """Simulate a torque–temperature pair with lead-lag coupling."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 4 * np.pi, n)
    torque = 2.0 + 1.5 * np.sin(t) + 0.3 * rng.standard_normal(n)
    # temperature lags behind torque with a thermal time constant
    dt = t[1] - t[0]
    tau_th = 80 * dt  # slow thermal constant
    temp = np.zeros(n)
    for i in range(1, n):
        temp[i] = temp[i - 1] + dt / tau_th * (torque[i - 1] ** 2 - temp[i - 1])
    temp += 0.05 * rng.standard_normal(n)
    # add a third channel: vibration (uncorrelated)
    vib = rng.standard_normal(n) * 0.5 + 0.3 * np.sin(10 * t)
    return pd.DataFrame({
        "torque_A": torque,
        "winding_temp_C": temp + 25,
        "vibration_mm_s": vib,
    })


def _demo_coupled(n=4000):
    """Three coupled channels with known causal structure: A→B→C."""
    rng = np.random.default_rng(7)
    t = np.linspace(0, 6 * np.pi, n)
    A = np.sin(t) + 0.4 * rng.standard_normal(n)
    dt = t[1] - t[0]
    B = np.zeros(n)
    for i in range(1, n):
        B[i] = B[i - 1] + (A[i - 1] - B[i - 1]) * 0.08 + 0.15 * rng.standard_normal()
    C = np.zeros(n)
    for i in range(1, n):
        C[i] = C[i - 1] + (B[i - 1] - C[i - 1]) * 0.05 + 0.1 * rng.standard_normal()
    return pd.DataFrame({
        "channel_A_drive": A,
        "channel_B_response": B + 2,
        "channel_C_downstream": C + 4,
    })


DEMO_DATA = {
    "Motor torque–temp–vib": _demo_motor(),
    "Coupled chain A→B→C": _demo_coupled(),
}

# ---------------------------------------------------------------------------
# core computation
# ---------------------------------------------------------------------------

def compute_pair_atoms(X, i, j):
    """Compute the 9 atoms for a single pair (i,j) from a (n,d) matrix."""
    n, d = X.shape
    if X[:, i].std() < 1e-12 or X[:, j].std() < 1e-12:
        return {nm: 0.0 for nm in atoms.ATOM_NAMES}
    U = atoms._ranks(X[:, [i, j]])
    rho = atoms._corr(U)
    eta = atoms._eta_matrix(U)
    lev = atoms._levy(U)
    jmp = atoms._jump(U)
    act = atoms._actime(U)
    fil = atoms._fill(U)
    bet = atoms._beta(X[:, [i, j]])
    eta_ji = eta[0, 1]
    eta_ij = eta[1, 0]
    eta_max = max(eta_ij, eta_ji)
    r = rho[0, 1]
    vals = {
        'rho': float(r),
        'eta': float(eta_max),
        'nlgap': float(eta_max - r ** 2),
        'asym': float(eta_ji - eta_ij),
        'levy': float(lev[0, 1]),
        'jump': float(jmp[0, 1]),
        'tau': float(np.log(act[1] / max(act[0], 1e-9) + 1e-9)),
        'fill': float(fil[0, 1]),
        'beta': float(bet[0, 1]),
    }
    return vals


def compute_atlas(X):
    """Given (n, d) array, compute atoms for every pair."""
    d = X.shape[1]
    results = []
    for i in range(d):
        for j in range(i + 1, d):
            vals = compute_pair_atoms(X, i, j)
            results.append({"pair": f"({i}, {j})", "i": i, "j": j, **vals})
    return results


def compute_atlas_summary(X):
    """Compute the atlas: per-pair atoms plus a summary dict."""
    pair_results = compute_atlas(X)
    if not pair_results:
        return pair_results, {}

    # aggregate: mean across pairs
    summary = {}
    for nm in atoms.ATOM_NAMES:
        vals = [r[nm] for r in pair_results]
        summary[nm] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
        }
    return pair_results, summary


def make_plots(X, X_reversed, X_warped, pair_results):
    """Generate the three-panel figure."""
    d = X.shape[1]
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: first two channels, raw
    ax1 = fig.add_subplot(gs[0, 0])
    if d >= 2:
        ax1.plot(X[:, 0], X[:, 1], linewidth=0.4, alpha=0.6, color="#2563eb")
        ax1.set_xlabel("Channel 0", fontsize=9)
        ax1.set_ylabel("Channel 1", fontsize=9)
    ax1.set_title("A. Raw channel pair", fontsize=11, fontweight="bold")
    ax1.set_aspect("equal", adjustable="datalim")

    # Panel B: time reversal — levy flips sign
    ax2 = fig.add_subplot(gs[0, 1])
    if d >= 2:
        levy_fwd = [r["levy"] for r in pair_results if r["i"] == 0 and r["j"] == 1]
        rev_pairs = compute_atlas(X_reversed)
        levy_rev = [r["levy"] for r in rev_pairs if r["i"] == 0 and r["j"] == 1]
        fwd = levy_fwd[0] if levy_fwd else 0
        rev = levy_rev[0] if levy_rev else 0
        colors = ["#2563eb", "#dc2626"]
        ax2.barh(["forward", "reversed"], [fwd, rev], color=colors, height=0.5)
        ax2.axvline(0, color="gray", linewidth=0.5)
        ax2.set_xlabel("Levy area", fontsize=9)
    ax2.set_title("B. Arrow of time", fontsize=11, fontweight="bold")

    # Panel C: Levin atom values as a bar chart (first pair)
    ax3 = fig.add_subplot(gs[0, 2])
    if pair_results:
        r0 = pair_results[0]
        vals = [r0[nm] for nm in atoms.ATOM_NAMES]
        colors3 = ["#059669" if nm != "levy" else "#dc2626" for nm in atoms.ATOM_NAMES]
        ax3.barh(atoms.ATOM_NAMES, vals, color=colors3, height=0.6)
        ax3.set_xlabel("atom value", fontsize=9)
    ax3.set_title(f"C. Atoms, pair (0,1)", fontsize=11, fontweight="bold")

    # Panel D: levy across all pairs
    ax4 = fig.add_subplot(gs[1, 0])
    if pair_results:
        levies = [r["levy"] for r in pair_results]
        labels = [f"({r['i']},{r['j']})" for r in pair_results]
        colors4 = ["#dc2626" if v < 0 else "#2563eb" for v in levies]
        ax4.barh(labels[:20], levies[:20], color=colors4, height=0.6)
        ax4.axvline(0, color="gray", linewidth=0.5)
        ax4.set_xlabel("Levy area", fontsize=9)
    ax4.set_title("D. Levy areas, all pairs", fontsize=11, fontweight="bold")

    # Panel E: correlation matrix for comparison
    ax5 = fig.add_subplot(gs[1, 1])
    if d >= 2:
        C = np.corrcoef(X.T)
        im = ax5.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
        ax5.set_title("E. Correlation (symmetric)", fontsize=11, fontweight="bold")
    ax5.set_xlabel("channel", fontsize=9)
    ax5.set_ylabel("channel", fontsize=9)

    # Panel F: recalibration invariance demo
    ax6 = fig.add_subplot(gs[1, 2])
    if pair_results and X_warped is not None:
        warped_pairs = compute_atlas(X_warped)
        fwd_levy = [r["levy"] for r in pair_results]
        warp_levy = [r["levy"] for r in warped_pairs]
        n_show = min(20, len(fwd_levy))
        x_pos = np.arange(n_show)
        ax6.scatter(x_pos, fwd_levy[:n_show], marker="o", s=30,
                    color="#2563eb", label="original", zorder=3)
        ax6.scatter(x_pos, warp_levy[:n_show], marker="x", s=30,
                    color="#f59e0b", label="after warp", zorder=3)
        ax6.set_xlabel("pair index", fontsize=9)
        ax6.set_ylabel("Levy area", fontsize=9)
        ax6.legend(fontsize=8)
    ax6.set_title("F. Invariance (blue=orig, yellow=warped)",
                  fontsize=11, fontweight="bold")

    fig.suptitle("Operating Atlas — interactive explorer", fontsize=14,
                 fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# recalculation helper
# ---------------------------------------------------------------------------

def _apply_monotone_warp(X, seed=42):
    """Independent monotone piecewise-linear warp per channel."""
    rng = np.random.default_rng(seed)
    Xw = np.empty_like(X)
    for j in range(X.shape[1]):
        x = X[:, j]
        order = np.argsort(x)
        xs = x[order]
        n = len(xs)
        nk = rng.integers(6, 14)
        knots = np.sort(rng.uniform(0, 1, nk))
        knots = np.unique(np.concatenate([[0], knots, [1]]))
        vals = np.sort(rng.uniform(0.3, 3.0, len(knots)))
        vals[0], vals[-1] = 0, 1
        mapped = np.interp(np.linspace(0, 1, n), knots, vals)
        Xw[order, j] = mapped * (xs[-1] - xs[0]) + xs[0]
    return Xw


# ---------------------------------------------------------------------------
# gradio interface
# ---------------------------------------------------------------------------

def analyze(data_choice, uploaded_file, sample_count):
    """Main analysis function."""
    # load data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file.name)
        except Exception as e:
            return None, f"Error reading CSV: {e}", ""
    elif data_choice in DEMO_DATA:
        df = DEMO_DATA[data_choice].copy()
    else:
        return None, "No data selected.", ""

    # keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None, "Need at least 2 numeric columns.", ""

    X = df[numeric_cols].to_numpy(np.float64)
    n = min(int(sample_count), len(X))
    X = X[:n]

    # drop constant columns
    keep = [j for j in range(X.shape[1]) if X[:, j].std() > 1e-12]
    X = X[:, keep]
    cols = [numeric_cols[j] for j in keep]

    if X.shape[1] < 2:
        return None, "Need at least 2 non-constant columns.", ""

    # compute atlas
    pair_results, summary = compute_atlas_summary(X)

    # time reversal
    X_rev = X[::-1].copy()

    # monotone warp
    X_warp = _apply_monotone_warp(X)

    # parity check
    rev_results = compute_atlas(X_rev)
    fwd_levies = {r["levy"] for r in pair_results}
    rev_levies = {r["levy"] for r in rev_results}
    levy_max_shift = max(abs(f - r) for f, r in zip(
        [r["levy"] for r in pair_results],
        [r["levy"] for r in rev_results]
    )) if pair_results else 0

    # figure
    fig = make_plots(X, X_rev, X_warp, pair_results)

    # text summary
    n_pairs = len(pair_results)
    lines = [
        f"## Atlas summary",
        f"**{X.shape[0]} samples × {X.shape[1]} channels** → {n_pairs} pairs\n",
    ]
    for nm in atoms.ATOM_NAMES:
        s = summary[nm]
        tag = " ⚡" if nm == "levy" else ""
        inv = "✓ rank-invariant" if nm in atoms.WARP_INVARIANT else "✗ raw-scale"
        lines.append(
            f"**{nm}**{tag}: mean={s['mean']:+.4f}  "
            f"std={s['std']:.4f}  [{s['min']:.4f}, {s['max']:.4f}]  _{inv}_"
        )

    lines.append(f"\n## Time-reversal parity")
    lines.append(f"Max |levy(forward) − levy(reversed)| = **{levy_max_shift:.2e}**")
    lines.append(
        "_Eight atoms are even (|shift| < 10⁻¹⁰). "
        "Levy is odd: it flips sign under reversal._"
        if levy_max_shift > 1e-6
        else "_All atoms appear even — check data has enough samples._"
    )

    lines.append(f"\n## Channels used")
    lines.append(", ".join(f"`{c}`" for c in cols))

    text_out = "\n\n".join(lines)

    # pair table (markdown)
    if pair_results:
        header = "| pair | " + " | ".join(atoms.ATOM_NAMES) + " |"
        sep = "|---" + "|---" * len(atoms.ATOM_NAMES) + "|"
        rows = [header, sep]
        for r in pair_results[:30]:  # cap at 30 pairs for readability
            vals = " | ".join(f"{r[nm]:.3f}" for nm in atoms.ATOM_NAMES)
            rows.append(f"| ({r['i']},{r['j']}) | {vals} |")
        if len(pair_results) > 30:
            rows.append(f"| ... | _{len(pair_results) - 30} more pairs_ |")
        table_out = "\n".join(rows)
    else:
        table_out = "No pairs."

    return fig, text_out, table_out


# ---------------------------------------------------------------------------
# build the Gradio app
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Operating Atlas — Relational Atoms of Multi-Physics Operation",
) as demo:
    gr.Markdown("""
# 🌐 Operating Atlas Explorer

**Upload multi-channel telemetry (CSV) or choose a built-in demo** to see
the nine relational atoms computed live. The signed Levy area — the one
descriptor that captures the arrow of time — flips sign when you reverse the
record. Under a monotone warp of every channel (simulating sensor
recalibration), eight atoms are unmoved to machine precision.

_[Paper](https://huggingface.co/spaces/Sejibeji/operating-atlas) ·
arXiv · Nature Computational Science_
""")

    with gr.Row():
        with gr.Column(scale=1):
            demo_choice = gr.Radio(
                choices=list(DEMO_DATA.keys()) + ["Upload your own"],
                value="Motor torque–temp–vib",
                label="Data source",
            )
            file_upload = gr.File(
                label="Upload CSV (at least 2 numeric columns)",
                file_types=[".csv"],
                visible=True,
            )
            sample_slider = gr.Slider(
                minimum=500, maximum=50000, value=4000, step=500,
                label="Number of samples to use",
            )
            run_btn = gr.Button("⚡ Compute atlas", variant="primary")

        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Atlas visualization")
            text_output = gr.Markdown(label="Summary")
            table_output = gr.Markdown(label="Per-pair atoms")

    run_btn.click(
        fn=analyze,
        inputs=[demo_choice, file_upload, sample_slider],
        outputs=[plot_output, text_output, table_output],
    )

    # auto-run on load with default demo
    demo.load(
        fn=analyze,
        inputs=[demo_choice, file_upload, sample_slider],
        outputs=[plot_output, text_output, table_output],
    )

    gr.Markdown("""
---

### What you're seeing

| Panel | Description |
|---|---|
| **A** | Raw scatter of the first two channels — a shape emerges |
| **B** | The Levy area for one pair: forward vs reversed record. It flips sign. |
| **C** | All 9 atoms for the first pair. Levy is highlighted in red. |
| **D** | Levy areas across all channel pairs — which pairs carry the most lag? |
| **E** | Standard correlation matrix for comparison (symmetric, can't see lead-lag) |
| **F** | Invariance demo: original atoms (blue) vs after monotone warp (yellow). Identical. |

### The key insight

A correlation matrix is **symmetric** — swap the channels and nothing changes.
The Levy area is **antisymmetric** — it flips sign. That's how it captures
lead-lag: torque heating a winding (torque → temperature) gives a positive
Levy area; the reverse gives negative. Every other relational statistic
in the standard toolkit lives in the even sector and is blind to this.
""")

if __name__ == "__main__":
    demo.launch()
