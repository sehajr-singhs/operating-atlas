# The Shapes a Machine Draws

**Recalibration-invariant relational atoms of multi-physics operation for industrial telemetry**

> Every pair of a machine's channels traces a shape as it runs. Nine scalar descriptors of those shapes separate kinds of machine, diagnose real faults, and do not move at all when the instrumentation is changed.

[Paper (22 pages, NCS format)](https://sehajr-singhs.github.io/operating-atlas/manuscript_NCS.pdf) · [Interactive site](https://sehajr-singhs.github.io/operating-atlas/) · [Kaggle kernels](https://www.kaggle.com/sehajrsingh) · [HF Space](https://huggingface.co/spaces/Sejibeji/operating-atlas)

## Results at a glance

| Claim | Result |
|---|---|
| 7 real systems, 309 records | **99.4%** identification (CI [97.7, 100.0]) |
| Under re-instrumentation | Atlas **99.0%** vs marginals 80.3% |
| Fleet control (3 platforms × 3 types) | Atlas at chance on level, significant on shape |
| AE baseline | Wins clean (58.3%), collapses to 2.1% under warp |
| Lévy area (time reversal) | Odd atom to 4.7×10⁻¹⁴; 32.5% vs 7.5% Spearman |

## Quick start

```bash
git clone https://github.com/sehajr-singhs/operating-atlas
cd operating-atlas
pip install -r requirements.txt

# Compute atoms from a motor session
python -m atlas.atoms --input data/robot_ur5e.npz

# Build the findings page
python atlas/build_page.py
```

## Repository structure

```
atlas/
  atoms.py          # Core atom computation (9 atoms per channel pair)
  real_systems.py   # Loaders for 7 real industrial systems
  bench_real.py     # 7-system benchmark + fault diagnosis
  figs_real.py      # Real-systems figure generation
  build_page.py     # Static findings page generator
  app.py            # Gradio interactive demo (HF Space)
paper/
  ncs.tex           # Manuscript source (22 pages, 44 references)
  ncs.pdf           # Compiled PDF
kaggle_kernel/
  ioo-shape-level/     # UR5e fleet experiment
  ioo-shape-level-panda/  # Panda fleet experiment
  ioo-shape-level-iiwa14/  # iiwa14 fleet experiment
  ioo-real2/           # 7-system benchmark
  ioo-deep-baselines/  # AE/LSTM/transformer comparison
  ioo-patchtst/        # PatchTST baseline
```

## The nine atoms

| # | Atom | Type | What it measures |
|---|---|---|---|
| 1 | Spearman ρ | Even, rank | Monotone association |
| 2 | Distance correlation | Even, rank | Any dependence |
| 3 | Hoeffding D | Even, rank | Nonlinearity gap |
| 4 | Copula asymmetry | Even, rank | Tail dependence asymmetry |
| 5 | **Signed Lévy area** | **Odd, time** | **Arrow of time** |
| 6 | Jump share | Even, rank | Discontinuity |
| 7 | Timescale ratio | Even, rank | Frequency distribution |
| 8 | Support occupancy | Even, rank | Filling of rank-plane |
| 9 | Scaling exponent | Even, rank | Hurst-like persistence |

## Reproducibility

All figures and tables are reproducible from the released code. The benchmark runs end-to-end as Kaggle notebooks (GPU-free for the fleet experiments, GPU for deep baselines).

## Author

**Sehaj Randhir Singh** — Department of Electrical and Computer Engineering, NYU Tandon School of Engineering; Independent Researcher

## License

Code is released for reproducibility. Please cite the paper if you use this work.

## Citation

```bibtex
@article{singh2026shapes,
  title={The shapes a machine draws: a recalibration-invariant relational fingerprint of multi-physics operation},
  author={Singh, Sehaj Randhir},
  year={2026}
}
```
