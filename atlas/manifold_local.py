"""
The operating manifold, as a geometric body rather than a set of 2-D shadows.

A machine sweeping through its duty cycle traces a set in R^d. That set is the
object. It has, at every operating point:

    local dimension   how many degrees of freedom are actually live here. A
                      motor holding a setpoint may be moving on a 1-D curve; the
                      same motor in a loaded transient may be on a 5-D sheet.
                      This is a property of the point, not of the machine.
    curvature         how the tangent space bends as you move. A hard constraint
                      between variables is a flat sheet; a saturating or
                      derating coupling is a curved one.
    tear              whether the body is locally connected. A commanded
                      setpoint change is a genuine discontinuity, and the
                      manifold has a boundary there rather than curvature.
    flow structure    the trajectory's velocity split into the part along the
                      manifold and the part off it, plus how much the motion
                      circulates in the tangent plane, which is the arrow of
                      time.

Adding a variable adds a coordinate to R^d. Nothing in the estimator changes,
which is the property the pairwise construction did not have: pairs are the
2-D marginals of this object, and three variables lying on a curved 2-surface
give three unremarkable-looking pairwise scatters.

The output is per operating point, so a whole record becomes a sequence of local
geometry that a network can consume directly, one vector per timestep, in place
of or alongside the raw channels.
"""

import numpy as np

FIELD_NAMES = ['dim', 'curv', 'tear', 'v_tan', 'v_norm', 'circ', 'density']
FIELD_LONG = {
    'dim': 'local intrinsic dimension (participation ratio of the local spectrum)',
    'curv': 'local curvature (normal extent relative to tangent extent)',
    'tear': 'local discontinuity (neighbourhood gap relative to its scale)',
    'v_tan': 'speed along the manifold',
    'v_norm': 'speed off the manifold',
    'circ': 'circulation of the flow in the tangent plane (signed, odd in time)',
    'density': 'log local density (how much of its life the machine spends here)',
}


def _ranks(X):
    """Column-wise rank transform to (0,1) with ties averaged.

    Kept from the pairwise construction because it is the one thing there that
    was unambiguously right: it makes every quantity below invariant to a
    strictly increasing recalibration of each channel separately, which is what
    a replaced sensor does.
    """
    n, d = X.shape
    R = np.empty((n, d))
    for j in range(d):
        x = X[:, j]
        o = np.argsort(x, kind='stable')
        xs = x[o]
        b = np.flatnonzero(np.concatenate(([True], xs[1:] != xs[:-1], [True])))
        cnt = np.diff(b)
        avg = b[:-1].astype(np.float64) + 0.5 * (cnt - 1) + 1.0
        R[o, j] = np.repeat(avg, cnt)
    return R / (n + 1.0)


def channel_snr(X, win=9, eps=1e-12):
    """Per-channel signal-to-noise, from smooth structure versus white residual.

    This is the fix for the failure that matters most. Normalising each channel
    to unit scale hands a dead sensor exactly the same weight as a live one, so
    appending two channels of pure noise to a Swiss roll moved its measured
    dimension from 1.97 to 4.02. The body had not changed; the embedding had.

    Physically a channel that is reading noise contributes no degree of freedom
    to the machine's operating manifold, and the estimator has to know that. A
    moving-average split separates the part of a channel that moves coherently
    in time from the part that does not, and the ratio is used to weight the
    channel. A live channel keeps its full weight, a dead one is switched off,
    and adding either leaves the geometry of the body alone.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    w = max(3, int(win) | 1)
    pad = w // 2
    Xp = np.pad(X, ((pad, pad), (0, 0)), mode='edge')
    ker = np.ones(w) / w
    S = np.empty_like(X)
    for j in range(d):
        S[:, j] = np.convolve(Xp[:, j], ker, mode='valid')[:n]
    resid = X - S
    sig = S.var(0)
    noi = resid.var(0)
    raw = sig / (sig + noi + eps)
    # A w-point moving average of white noise keeps 1/w of its variance, so a
    # dead channel scores 1/w rather than 0 on the raw ratio. With w = 9 that is
    # 0.111, which sailed past a 0.02 floor and let six channels of pure noise
    # push a 2-dimensional body to 6.22. Referencing the score to its own null
    # sends white noise to exactly 0 and leaves a smooth channel near 1.
    null = 1.0 / w
    return np.clip((raw - null) / (1.0 - null), 0.0, 1.0)


def intrinsic_dim_twonn(Z, frac=0.9, eps=1e-12):
    """Global intrinsic dimension by the two-nearest-neighbour estimator.

    Facco et al. (2017). For each point take the distances to its first and
    second neighbours; on a locally flat manifold of dimension m the ratio
    mu = r2/r1 has a Pareto(m) distribution, so m follows from the slope of the
    empirical log-log survival curve. It uses only the two closest neighbours,
    which is the smallest scale at which the manifold is flat, so it does not
    need a neighbourhood size chosen by hand.

    This replaces a soft eigenvalue count of my own devising, which on a real
    CNC tool path of true dimension 1 returned 1.90, 1.90 and 0.29 as k varied,
    and 0.00 on a command-versus-actual pair. Any estimator that reports a
    dimension below one for a real trajectory is measuring its own noise floor.
    """
    from sklearn.neighbors import NearestNeighbors
    Z = np.asarray(Z, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=3).fit(Z)
    d, _ = nn.kneighbors(Z)
    r1, r2 = d[:, 1], d[:, 2]
    ok = (r1 > eps) & (r2 > r1)
    mu = np.sort(r2[ok] / r1[ok])
    n = len(mu)
    if n < 20:
        return float('nan')
    F = np.arange(1, n + 1) / n
    cut = int(frac * n)                       # drop the heavy tail
    x = np.log(mu[:cut])
    y = -np.log(np.clip(1.0 - F[:cut], eps, None))
    return float((x @ y) / max(x @ x, eps))   # slope through the origin


def local_dim_mle(dist_row, eps=1e-12):
    """Local intrinsic dimension by the Levina-Bickel maximum likelihood
    estimator, from one point's sorted neighbour distances."""
    T = np.asarray(dist_row, dtype=np.float64)
    T = T[T > eps]
    if len(T) < 5:
        return float('nan')
    Tk = T[-1]
    r = np.log(Tk / T[:-1])
    s = r.mean()
    return float(1.0 / s) if s > eps else float('nan')


def local_geometry(X, k=64, n_probe=4000, seed=0, embed='rank', eps=1e-12,
                   snr_weight=True, snr_floor=0.02):
    """Estimate the local geometry of the manifold traced by X.

    X : (n, d) raw channels, in time order.
    k : neighbourhood size. The local geometry is only defined relative to a
        scale, and k sets it. Too small and everything looks 1-D and flat
        because the neighbourhood is inside the noise ball; too large and
        everything looks like the global cloud.
    n_probe : the manifold is probed at this many operating points. The geometry
        is a field over the body, so it is estimated where the machine actually
        went, by subsampling the trajectory itself rather than gridding R^d,
        which would be empty almost everywhere.

    Returns dict of per-probe arrays plus the probe indices and tangent frames.
    """
    X = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(X).all(1)
    X = X[finite]
    n, d = X.shape
    if n < k * 2:
        raise ValueError(f'need at least {2*k} samples, got {n}')

    if embed == 'rank':
        Z = _ranks(X)
    elif embed == 'none':
        Z = X - X.mean(0)          # keeps absolute scale, so curvature is 1/R
    else:
        Z = (X - X.mean(0)) / (X.std(0) + eps)
    if snr_weight and embed != 'none':
        # NOTE: this weighting reads temporal coherence, so it is only
        # meaningful when the rows are a trajectory. On i.i.d. samples of a
        # shape every channel looks white, every weight collapses to zero, and
        # the embedding degenerates to a single point. Fall back rather than
        # silently returning the geometry of the origin.
        w = np.sqrt(np.maximum(channel_snr(X) - snr_floor, 0.0))
        if w.max() < 1e-6:
            w = np.ones(d)
        # A dead channel is not a dimension of the machine's manifold, so it
        # must not be a dimension of the ambient space either. Scaling it to
        # zero is not enough: it leaves an exactly-zero eigenvalue in the local
        # spectrum, which drags the noise floor down and promotes the real
        # manifold's own noise direction to signal. That alone moved a
        # 1.3-dimensional body to 2.60 when four dead channels were appended.
        # Dropping the columns outright makes the geometry exactly invariant to
        # how many dead channels the logger happened to record.
        live = w > 1e-6
        if live.sum() >= 2:
            Z, w, d = Z[:, live], w[live], int(live.sum())
        Z = Z * w
    # velocity in the embedding, central difference so it is not biased in time
    V = np.zeros_like(Z)
    V[1:-1] = 0.5 * (Z[2:] - Z[:-2])
    V[0], V[-1] = Z[1] - Z[0], Z[-1] - Z[-2]

    rng = np.random.default_rng(seed)
    probe = (np.arange(n) if n <= n_probe
             else np.sort(rng.choice(n, n_probe, replace=False)))

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(Z)
    dist, idx = nn.kneighbors(Z[probe])
    dist, idx = dist[:, 1:], idx[:, 1:]          # drop self

    P = len(probe)
    # A neighbourhood of k points spans at most k directions, so the local
    # spectrum has min(k, d) entries and not d. Allocating d and assigning
    # min(k, d) is a straight crash, and quietly padding it with zeros would be
    # worse: those zeros would sink the noise floor and inflate every dimension.
    nl = int(min(k, d))
    if k < d + 2:
        import warnings
        warnings.warn(
            f'k={k} neighbours cannot resolve a {d}-dimensional neighbourhood; '
            f'local dimension will be capped near {nl}. Use k > d.',
            RuntimeWarning, stacklevel=2)
    out = {f: np.zeros(P) for f in FIELD_NAMES}
    frames = np.zeros((P, nl, d))
    spectra = np.zeros((P, nl))

    for a in range(P):
        nb = Z[idx[a]]
        c = nb.mean(0)
        Y = nb - c
        # local PCA: the spectrum IS the local shape
        try:
            _, s, Vt = np.linalg.svd(Y, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        lam = (s ** 2) / max(len(nb) - 1, 1)
        nlam = len(lam)
        tot = lam.sum() + eps
        spectra[a, :nlam] = lam
        frames[a, :len(Vt)] = Vt

        # Intrinsic dimension, counted against the local noise floor rather than
        # by participation ratio. A d-dimensional ball of noise has a flat
        # spectrum, so participation ratio reports d and calls noise structure;
        # that is what inflated a 1-D curve to 2.08. A manifold instead gives m
        # eigenvalues well above the floor and d-m at it, so the floor is
        # estimated from the bottom of the spectrum and eigenvalues are counted
        # softly against it, which keeps the measure continuous as a sheet
        # thickens.
        # Intrinsic dimension by the Levina-Bickel MLE on this point's
        # neighbour distances. The previous soft eigenvalue count was mine and
        # it did not survive contact with real data, swinging from 1.90 to 0.29
        # on a CNC path of true dimension 1 as the neighbourhood size changed.
        m_eff = local_dim_mle(dist[a])
        if not np.isfinite(m_eff):
            m_eff = 1.0
        m_eff = float(np.clip(m_eff, 0.5, nlam))
        out['dim'][a] = m_eff

        # curvature: with m tangent directions, tangent extent is O(r) and
        # normal extent is O(r^2 * kappa), so the ratio of normal to tangent
        # scale, divided by the neighbourhood radius, estimates kappa.
        m = int(max(1, min(nlam - 1, round(m_eff))))
        tan = np.sqrt(lam[:m].sum() + eps)
        nor = np.sqrt(lam[m:].sum() + eps)
        r = dist[a].mean() + eps
        out['curv'][a] = nor / (tan * r + eps)

        # Tear, measured as one-sidedness of the neighbourhood.
        #
        # Looking for a gap in the sorted neighbour distances does not work and
        # reported 0.12 on two disjoint sheets, identical to a single plane: with
        # k neighbours the neighbourhood simply never reaches across the gap, so
        # there is no gap in it to find. What does change at a tear or a boundary
        # is symmetry. Interior to a manifold the neighbours surround the point
        # and their centroid sits on it; at an edge they lie to one side and the
        # centroid is pulled inward. The offset, in units of the neighbourhood
        # radius, is therefore the boundary detector.
        out['tear'][a] = np.linalg.norm(c - Z[probe[a]]) / (dist[a].mean() + eps)

        # flow: split the velocity into along-manifold and off-manifold
        v = V[probe[a]]
        T = Vt[:m]
        v_t = T @ v
        out['v_tan'][a] = np.linalg.norm(v_t)
        out['v_norm'][a] = np.sqrt(max(v @ v - v_t @ v_t, 0.0))

        # circulation: signed area swept by the flow in the leading tangent
        # plane, which is odd under time reversal and so carries lead/lag
        if m >= 2:
            w = Vt[:2] @ (Z[idx[a]] - c).T          # 2 x k local coords
            dw = np.diff(w, axis=1)
            area = (w[0, :-1] * dw[1] - w[1, :-1] * dw[0]).sum()
            denom = np.abs(w[0, :-1] * dw[1] - w[1, :-1] * dw[0]).sum() + eps
            out['circ'][a] = area / denom

        out['density'][a] = -np.log(r)

    return dict(probe=probe, fields=out, frames=frames, spectra=spectra,
                Z=Z, k=k, d=d)


def field_matrix(geo):
    """(n_probe, 7) array a network can consume, one row per operating point."""
    return np.stack([geo['fields'][f] for f in FIELD_NAMES], 1)


def signature(geo, nq=9):
    """Class-level summary: the distribution of local geometry over the body.

    A machine class is not one shape, it is a characteristic way of being
    shaped: how much of its envelope is low dimensional and flat, how much is
    curved, where it tears. Quantiles of each field give that in a fixed number
    of values regardless of how many channels the machine has or how long it
    ran, which is what makes a motor comparable to an actuator.
    """
    qs = np.linspace(0.05, 0.95, nq)
    F = field_matrix(geo)
    return np.concatenate([np.quantile(F[:, j], qs) for j in range(F.shape[1])])
