"""
Which atoms can see the arrow of time?

The claim that a correlation matrix cannot express a lead-lag direction is
usually argued from the symmetry of the pair (i, j). There is a sharper and
more useful statement available, in terms of TIME reversal rather than channel
exchange.

Reverse a record end to end. A lag becomes a lead, so any descriptor that
detects lag must change. Almost every relational statistic in use is built from
the joint distribution of simultaneous values, which is invariant to reordering
the samples, and is therefore exactly unchanged. Correlation, mutual
information, distance correlation, coherence magnitude and the support geometry
all fall in that class.

Measured here on real motor telemetry: eight of the nine atoms are invariant
under time reversal to machine precision, and the signed Levy area flips sign
exactly. It is the only atom in the vocabulary that carries the direction of
time, which is why removing it costs most of the identification.
"""

import os
import numpy as np
import atoms

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, '_pmsm_sessions.npz'))
X_all = z['X']

fwd, rev = [], []
for u in range(len(X_all)):
    X = X_all[u].astype(np.float64)
    fwd.append(atoms.atlas_unit(X, n_cells=1, min_cell=50)[0])
    rev.append(atoms.atlas_unit(X[::-1].copy(), n_cells=1, min_cell=50)[0])
F, Rv = np.array(fwd), np.array(rev)

print(f'{len(F)} real motor sessions, forward vs time-reversed\n')
print(f"{'atom':>7}  {'max|A(rev) - A(fwd)|':>22}  {'max|A(rev) + A(fwd)|':>22}"
      f"  verdict")
for ai, nm in enumerate(atoms.ATOM_NAMES):
    a, b = F[..., ai], Rv[..., ai]
    m = np.isfinite(a) & np.isfinite(b)
    same = float(np.max(np.abs(b[m] - a[m])))
    flip = float(np.max(np.abs(b[m] + a[m])))
    if flip < same * 1e-3:
        verdict = 'ODD  (flips sign: sees the arrow of time)'
    elif same < max(flip, 1e-12) * 1e-3:
        verdict = 'EVEN (invariant: blind to the arrow of time)'
    else:
        verdict = 'neither cleanly'
    print(f'{nm:>7}  {same:>22.3e}  {flip:>22.3e}  {verdict}')

lv = atoms.ATOM_NAMES.index('levy')
mag = np.abs(F[..., lv])
print(f'\nlevy magnitude on these sessions: median {np.median(mag):.3f}, '
      f'p95 {np.quantile(mag, 0.95):.3f}, max {mag.max():.3f}')
print('An even atom cannot distinguish a lag from a lead, however it is scaled,')
print('because reversing the record leaves it numerically identical.')
