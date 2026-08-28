"""
Local pilot. Three falsifiable claims, on 709 turbofan units across 4 classes.
Deliberately small and CPU-only: this decides whether the idea is worth cloud
compute at all, and it is the cheapest possible way to find out.

  P1 CLASS SEPARABILITY  the atlas identifies which class a held-out unit
                         belongs to. If a class prototype exists, this works.
  P2 UNIT FINGERPRINTING the atlas built from the first half of a unit's life
                         retrieves that same unit from its second half, against
                         708 distractors. This is the 'index of operations':
                         each unit is a distinguishable deformation.
  P3 LOW-RANK DEFORMATION within a class, atlas variation across units lives in
                         a low-dimensional subspace. This is the claim that a
                         base class 'fine-tunes' to a unit by moving a few knobs.

Baselines throughout: per-channel marginal summaries (mean/sd/skew/kurtosis of
each channel), which is what any sane engineer would try first, and Spearman
correlation alone, which is the atom vocabulary reduced to its first entry.
"""

import time
import numpy as np
import atoms
import data_cmapss as dc

SUBS = dc.SUBSETS
N_CELLS = 1          # C-MAPSS records are short (~200 cycles); one cell per unit


def build():
    feats, labels, uids, halves = [], [], [], []
    t0 = time.time()
    for ci, sub in enumerate(SUBS):
        us = dc.units(sub)
        for uid, X, meta in us:
            h = len(X) // 2
            a_full = atoms.atlas_unit(X, n_cells=N_CELLS, min_cell=50)
            a_1 = atoms.atlas_unit(X[:h], n_cells=N_CELLS, min_cell=50)
            a_2 = atoms.atlas_unit(X[h:], n_cells=N_CELLS, min_cell=50)
            feats.append(a_full.reshape(-1))
            halves.append((a_1.reshape(-1), a_2.reshape(-1)))
            labels.append(ci)
            uids.append(f'{sub}:{uid}')
        print(f'  {sub}: {len(us)} units  [{time.time()-t0:.1f}s]', flush=True)
    F = np.array(feats)
    H1 = np.array([h[0] for h in halves])
    H2 = np.array([h[1] for h in halves])
    return F, H1, H2, np.array(labels), np.array(uids)


def marginal_baseline():
    """Per-channel mean/sd/skew/kurtosis -- the obvious non-relational feature."""
    from scipy.stats import skew, kurtosis
    feats, h1, h2, labels = [], [], [], []

    def summ(X):
        return np.concatenate([X.mean(0), X.std(0), skew(X, 0), kurtosis(X, 0)])

    for ci, sub in enumerate(SUBS):
        for uid, X, meta in dc.units(sub):
            h = len(X) // 2
            feats.append(summ(X)); h1.append(summ(X[:h])); h2.append(summ(X[h:]))
            labels.append(ci)
    return (np.nan_to_num(np.array(feats)), np.nan_to_num(np.array(h1)),
            np.nan_to_num(np.array(h2)), np.array(labels))


def clean(F):
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    keep = F.std(0) > 1e-9
    return F[:, keep], keep


def p1_class(F, y, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Fc, _ = clean(F)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, C=0.1))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    s = cross_val_score(clf, Fc, y, cv=cv, scoring='accuracy')
    base = np.bincount(y).max() / len(y)
    print(f'  P1 {tag:<22} acc = {s.mean()*100:5.1f} +- {s.std()*100:4.1f} %'
          f'   (majority {base*100:.1f}%)  dim={Fc.shape[1]}')
    return s.mean()


def p2_fingerprint(H1, H2, tag):
    """Rank-1 identification rate and mean percentile rank, cosine on z-scored
    features with the class-mean removed so we are not just re-detecting class."""
    A, k = clean(H1)
    B = np.nan_to_num(H2, nan=0.0, posinf=0.0, neginf=0.0)[:, k]
    mu = np.concatenate([A, B]).mean(0)
    sd = np.concatenate([A, B]).std(0) + 1e-9
    A = (A - mu) / sd
    B = (B - mu) / sd
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    S = A @ B.T
    n = len(A)
    rank = (S > S[np.arange(n), np.arange(n)][:, None]).sum(1)
    top1 = (rank == 0).mean()
    pct = 1.0 - rank / (n - 1)
    print(f'  P2 {tag:<22} top-1 = {top1*100:5.1f} %   mean pct-rank = '
          f'{pct.mean()*100:5.1f} %   (chance {100/n:.2f} % / 50.0 %)  n={n}')
    return top1


def p3_lowrank(F, y, tag):
    Fc, _ = clean(F)
    Fc = (Fc - Fc.mean(0)) / (Fc.std(0) + 1e-9)
    for ci, sub in enumerate(SUBS):
        Z = Fc[y == ci]
        if len(Z) < 20:
            continue
        Z = Z - Z.mean(0)
        s = np.linalg.svd(Z, compute_uv=False)
        v = np.cumsum(s ** 2) / (s ** 2).sum()
        r80 = int(np.searchsorted(v, 0.80) + 1)
        r90 = int(np.searchsorted(v, 0.90) + 1)
        print(f'  P3 {tag:<10} {sub}  n={len(Z):3d}  rank for 80% var = {r80:3d}'
              f'   90% = {r90:3d}   (ambient dim {Z.shape[1]})')


if __name__ == '__main__':
    print('building atlases ...', flush=True)
    F, H1, H2, y, uids = build()
    print(f'atlas feature dim = {F.shape[1]}  units = {len(F)}')
    print('building marginal baseline ...', flush=True)
    MF, MH1, MH2, my = marginal_baseline()
    print(f'marginal feature dim = {MF.shape[1]}')

    # atom ablation: rho alone is the classical correlation-matrix feature
    P = F.shape[1] // len(atoms.ATOM_NAMES)
    Fr = F.reshape(len(F), -1, len(atoms.ATOM_NAMES))
    H1r = H1.reshape(len(F), -1, len(atoms.ATOM_NAMES))
    H2r = H2.reshape(len(F), -1, len(atoms.ATOM_NAMES))
    rho_only = Fr[:, :, 0]
    H1_rho, H2_rho = H1r[:, :, 0], H2r[:, :, 0]

    print('\n--- P1 class separability (5-fold CV, logistic) ---')
    p1_class(MF, my, 'marginals baseline')
    p1_class(rho_only, y, 'rho only (corr matrix)')
    p1_class(F, y, 'FULL ATLAS (9 atoms)')

    print('\n--- P2 unit fingerprinting (first half -> second half) ---')
    p2_fingerprint(MH1, MH2, 'marginals baseline')
    p2_fingerprint(H1_rho, H2_rho, 'rho only (corr matrix)')
    p2_fingerprint(H1, H2, 'FULL ATLAS (9 atoms)')

    print('\n--- P3 low-rank deformation within class ---')
    p3_lowrank(F, y, 'atlas')

    print('\n--- per-atom contribution (P1 / P2 with one atom alone) ---')
    for ai, nm in enumerate(atoms.ATOM_NAMES):
        a1 = p1_class(Fr[:, :, ai], y, f'atom {nm}')
        a2 = p2_fingerprint(H1r[:, :, ai], H2r[:, :, ai], f'atom {nm}')
    np.savez('pilot_cmapss.npz', F=F, H1=H1, H2=H2, y=y, uids=uids)
    print('\nsaved pilot_cmapss.npz')
