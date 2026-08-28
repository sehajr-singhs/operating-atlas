"""Which atom leaks on the PMSM cache, and why?"""
import os
import numpy as np
import atoms

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, '_pmsm_sessions.npz'))
X_all, CH = z['X'], [str(c) for c in z['channels']]

worst = {nm: 0.0 for nm in atoms.ATOM_NAMES}
worst_u = {nm: -1 for nm in atoms.ATOM_NAMES}
for u in range(len(X_all)):
    Xb = X_all[u, 8000:].astype(np.float64)
    W = atoms.warp_channels(Xb, np.random.default_rng(9000 + u))
    A = atoms.atlas_unit(Xb, n_cells=1, min_cell=50)[0]
    B = atoms.atlas_unit(W, n_cells=1, min_cell=50)[0]
    for ai, nm in enumerate(atoms.ATOM_NAMES):
        dmax = np.nanmax(np.abs(A[:, ai] - B[:, ai]))
        if np.isfinite(dmax) and dmax > worst[nm]:
            worst[nm], worst_u[nm] = float(dmax), u

print('worst |shift| per atom over 40 sessions:')
for nm in atoms.ATOM_NAMES:
    print(f'  {nm:>6} = {worst[nm]:.3e}   (session index {worst_u[nm]})')

# inspect the offending session
u = max(atoms.WARP_INVARIANT, key=lambda nm: worst[nm])
ui = worst_u[u]
print(f'\nworst invariant atom is {u} at session index {ui}')
Xb = X_all[ui, 8000:].astype(np.float64)
W = atoms.warp_channels(Xb, np.random.default_rng(9000 + ui))
print('  per-channel std before -> after warp:')
for j, c in enumerate(CH):
    print(f'    {c:>15}  {Xb[:, j].std():.4e} -> {W[:, j].std():.4e}'
          f'   nunique {len(np.unique(Xb[:, j])):>6} -> {len(np.unique(W[:, j])):>6}')
kb = Xb.std(0) > 1e-10
kw = W.std(0) > 1e-10
print('  channels kept before:', int(kb.sum()), ' after:', int(kw.sum()))
print('  differing:', [CH[j] for j in range(len(CH)) if kb[j] != kw[j]])
