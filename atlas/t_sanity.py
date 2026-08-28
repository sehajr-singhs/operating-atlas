"""Synthetic sanity check: do the atoms report the mechanism we built in?"""
import itertools
import numpy as np
import atoms

rng = np.random.default_rng(0)
n = 6000
t = np.linspace(0, 60, n)
a = np.sin(t) + 0.03 * rng.normal(size=n)              # ch0 drive
# ch1: causal first-order lag of a, i.e. what a thermal mass actually does.
# (a CENTRED moving average would be zero-phase and enclose no area at all)
from scipy.signal import lfilter
alpha = 0.99
b = lfilter([1 - alpha], [1, -alpha], a)                # -> Levy area, slow tau
c = a ** 2 + 0.03 * rng.normal(size=n)                  # ch2 even fn of a -> rho ~ 0, eta high
# ch3: piecewise-constant setpoint switching between levels -- what a commanded
# load step actually is. NOT a monotone random walk: a rank transform turns a
# monotone trend into a straight ramp, so a cumulative walk has no jumps left in
# rank space and would test nothing.
sw = np.flatnonzero(rng.random(n) < 0.004)
lvl = np.zeros(n)
cur = 0.0
prev = 0
for s in sw:
    lvl[prev:s] = cur
    cur = rng.choice([0.0, 1.0, 2.0, 3.0])
    prev = s
lvl[prev:] = cur
d = lvl + 0.02 * rng.normal(size=n)
X = np.stack([a, b, c, d], 1)
A = atoms.atlas_unit(X, n_cells=1)[0]

lbl = {0: 'drive', 1: 'lagged', 2: 'square', 3: 'jumpy'}
print(f"{'pair':<18}" + ''.join(f'{k:>8}' for k in atoms.ATOM_NAMES))
for k, (i, j) in enumerate(itertools.combinations(range(4), 2)):
    name = f'{lbl[i]}-{lbl[j]}'
    print(f'{name:<18}' + ''.join(f'{v:>8.3f}' for v in A[k]))

print()
print('EXPECTED')
print('  drive-lagged : |levy| large (oriented loop), tau > 0 (ch1 slower), eta high')
print('  drive-square : rho ~ 0 but eta high -> nlgap large (nonlinearity a corr misses)')
print('  *-jumpy      : jump share near 1 (co-movement carried by the steps)')

# warp invariance
W = atoms.warp_channels(X, np.random.default_rng(7))
B = atoms.atlas_unit(W, n_cells=1)[0]
print()
for ai, nm in enumerate(atoms.ATOM_NAMES):
    x, y = A[:, ai], B[:, ai]
    m = np.isfinite(x) & np.isfinite(y)
    dev = np.abs(x[m] - y[m]).max() if m.sum() else np.nan
    print(f'  warp max|delta| {nm:>6} = {dev:.2e}')
