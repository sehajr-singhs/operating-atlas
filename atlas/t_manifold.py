"""
Does the estimator recover geometry we already know the answer to?

Two kinds of case, kept apart because they test different things.

PART A is pure geometry: i.i.d. samples of a known shape, in no time order. The
SNR weighting reads temporal coherence and is meaningless here, so it is off.
Two of these cases are built to be invisible to any pairwise construction: the
Swiss roll and a sphere in higher dimensions both have unremarkable 2-D
marginals while being strongly curved as bodies.

PART B is the case that actually matters for machines: a TRAJECTORY, to which
dead channels are appended. The body must not change. This is the property the
first version of the estimator failed, moving a Swiss roll from dimension 1.97
to 4.02 when two channels of noise were added.
"""

import numpy as np
import manifold_local as ml

rng = np.random.default_rng(0)
N = 5000
PROBE = 900


def report(name, X, expect, k=48, embed='z', snr=False):
    geo = ml.local_geometry(X, k=k, n_probe=PROBE, seed=0, embed=embed,
                            snr_weight=snr)
    f = geo['fields']
    print(f'  {name:<32} dim {np.median(f["dim"]):5.2f}   '
          f'curv {np.median(f["curv"]):7.3f}   tear {np.median(f["tear"]):5.2f}'
          f'    expect {expect}')
    return geo


print('PART A  pure geometry (i.i.d. samples, SNR weighting off)')

A = rng.normal(size=(N, 2))
E = np.linalg.qr(rng.normal(size=(5, 5)))[0][:, :2]
report('flat 2-plane in R^5', A @ E.T + 0.001 * rng.normal(size=(N, 5)),
       'dim 2, flat')

t = np.sort(rng.uniform(0, 4 * np.pi, N))
C = np.stack([np.cos(t), np.sin(t), t / 6, 0.3 * np.cos(2 * t), 0.2 * t], 1)
report('1-D curve in R^5', C + 0.002 * rng.normal(size=C.shape), 'dim 1')

# curvature must scale as 1/R, so this pair is run WITHOUT normalisation:
# z-scoring rescales a radius-4 sphere to radius 1 and hides the effect.
for R in (1.0, 4.0):
    v = rng.normal(size=(N, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    report(f'2-sphere radius {R:g}', R * v, f'dim 2, curv ~ 1/{R:g}',
           embed='none')

u = rng.uniform(1.5 * np.pi, 4.5 * np.pi, N)
h = rng.uniform(0, 21, N)
S = np.stack([u * np.cos(u), h, u * np.sin(u)], 1)
report('Swiss roll (curved sheet)', S, 'dim 2, curved')

# A tear is a property of a POINT, so it has to be tested where a point sits on
# an edge. Two parallel disjoint sheets have no such point: every neighbourhood
# lies wholly inside its own sheet and looks like ordinary interior, which is
# why they scored the same as a plane. A sheet with a slot cut out of it does
# have edge points, and the statistic has to fire there and nowhere else.
P2 = rng.uniform(-1, 1, size=(N * 2, 2))
P2 = P2[~((np.abs(P2[:, 0]) < 0.25) & (P2[:, 1] > -0.3))][:N]
slot = np.stack([P2[:, 0], P2[:, 1], 0.01 * rng.normal(size=len(P2))], 1)
g_slot = ml.local_geometry(slot, k=48, n_probe=PROBE, seed=0, embed='z',
                           snr_weight=False)
q = slot[g_slot['probe']]
edge = (np.abs(np.abs(q[:, 0]) - 0.25) < 0.12) & (q[:, 1] > -0.2)
interior = (np.abs(q[:, 0]) > 0.6)
te = g_slot['fields']['tear']
print(f'  {"sheet with a slot cut out":<32} tear at the cut '
      f'{np.median(te[edge]):.2f} vs interior {np.median(te[interior]):.2f}'
      f'    expect higher at the cut')

# dimension is a local property, not a number for the record
n2 = N // 2
p1 = np.stack([np.linspace(0, 5, n2), np.zeros(n2), np.zeros(n2)], 1)
p2 = np.stack([np.linspace(5, 10, n2), rng.uniform(-1, 1, n2), np.zeros(n2)], 1)
M = np.concatenate([p1, p2]) + 0.01 * rng.normal(size=(N, 3))
g = ml.local_geometry(M, k=48, n_probe=PROBE, seed=0, embed='z', snr_weight=True)
x = M[g['probe'], 0]
print(f'  {"curve opening into a sheet":<32} '
      f'dim on curve {np.median(g["fields"]["dim"][x < 4.5]):.2f}, '
      f'on sheet {np.median(g["fields"]["dim"][x > 5.5]):.2f}    expect 1 then 2')

print('\nPART B  a trajectory with dead channels appended (SNR weighting on)')
# a genuine time-ordered trajectory on a curved sheet
tt = np.linspace(0, 60, N)
uu = 2.0 + 1.3 * np.sin(0.7 * tt) + 0.9 * np.sin(0.11 * tt)
hh = 3.0 * np.sin(0.05 * tt)
T3 = np.stack([uu * np.cos(uu), hh, uu * np.sin(uu)], 1)
T3 = T3 + 0.01 * rng.normal(size=T3.shape)

base = ml.local_geometry(T3, k=48, n_probe=PROBE, seed=1, snr_weight=True)
print(f'  {"trajectory, 3 live channels":<32} '
      f'dim {np.median(base["fields"]["dim"]):.2f}   '
      f'curv {np.median(base["fields"]["curv"]):.3f}')

for extra, label in [(2, 'plus 2 dead channels'), (6, 'plus 6 dead channels')]:
    Tx = np.concatenate([T3, 0.001 * rng.normal(size=(N, extra))], 1)
    gx = ml.local_geometry(Tx, k=48, n_probe=PROBE, seed=1, snr_weight=True)
    print(f'  {label:<32} dim {np.median(gx["fields"]["dim"]):.2f}   '
          f'curv {np.median(gx["fields"]["curv"]):.3f}    expect unchanged')

# and a genuinely NEW live variable must be allowed to raise the dimension
live = np.sin(0.31 * tt)[:, None] + 0.01 * rng.normal(size=(N, 1))
Tl = np.concatenate([T3, live], 1)
gl = ml.local_geometry(Tl, k=48, n_probe=PROBE, seed=1, snr_weight=True)
print(f'  {"plus 1 LIVE channel":<32} dim {np.median(gl["fields"]["dim"]):.2f}'
      f'                     expect higher')
