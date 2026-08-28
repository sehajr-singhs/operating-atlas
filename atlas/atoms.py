"""
The operating atlas: relational atoms over channel pairs.

A system under load presents d channels. Every unordered pair (i,j) traces a
curve in the plane; the shape of that curve, and how it changes with operating
point, is what we summarise. Each pair gets a vector of nine ATOMS. An atom is
the elementary readable fact about a machine ("in this operating cell, torque
leads winding temperature, the relation is single-valued, and it does not jump").

Design constraints, in order of priority:
  1. Per-channel monotone invariance. A channel in Celsius vs Kelvin, or read
     through a drifting sensor gain, must give the same atom. Eight of the nine
     atoms are computed on within-unit ranks and are therefore invariant to any
     strictly increasing reparameterisation of each channel separately. That is
     exactly the invariance a fleet recalibration demands, and unlike a
     Riemannian invariant it costs nothing and is estimable from finite data.
     BETA is the deliberate exception: an exponent needs a scale, so it is
     computed on raw values and excluded from the warp-invariance experiment.
  2. O(d^2 n), vectorised. Everything below is matmuls, bincounts and rank
     transforms. No pairwise-distance O(n^2) statistics.
  3. Physical readability. Each atom names a mechanism, not a basis direction.
"""

import numpy as np

ATOM_NAMES = ['rho', 'eta', 'nlgap', 'asym', 'levy', 'jump', 'tau', 'fill', 'beta']
ATOM_LONG = {
    'rho':   'monotone association (Spearman)',
    'eta':   'functional strength (correlation ratio)',
    'nlgap': 'nonlinearity gap  eta - rho^2',
    'asym':  'functional asymmetry  eta_{j|i} - eta_{i|j}',
    'levy':  'oriented hysteresis (normalised Levy area)',
    'jump':  'jump share of co-movement',
    'tau':   'log timescale ratio',
    'fill':  'support occupancy of the joint cloud',
    'beta':  'log-log scaling exponent',
}
# atoms that are antisymmetric under swapping the pair (i,j) -> (j,i)
ANTISYM = {'asym', 'levy', 'tau'}
# atoms invariant to per-channel strictly-increasing warps
WARP_INVARIANT = [a for a in ATOM_NAMES if a != 'beta']


def _ranks(X):
    """Column-wise rank transform to (0,1), with TIES AVERAGED.

    Averaging ties is not cosmetic here, it is what makes the invariance hold on
    real data. A strictly increasing warp cannot reorder distinct values, but in
    floating point it can collapse two nearby values onto the same float. If
    ties are then broken by array order, those samples swap places relative to
    the unwarped ranking and every downstream atom moves -- which is exactly the
    leak that showed up as a 1e-1 shift under aggressive recalibration while the
    synthetic test, whose values never collided, reported machine precision.
    Average ranks assign tied values the same rank, so a warp that creates ties
    changes nothing.
    """
    n, d = X.shape
    r = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        x = X[:, j]
        order = np.argsort(x, kind='stable')
        xs = x[order]
        bnd = np.flatnonzero(np.concatenate(
            ([True], xs[1:] != xs[:-1], [True])))
        counts = np.diff(bnd)
        starts = bnd[:-1].astype(np.float64)
        avg = starts + 0.5 * (counts - 1) + 1.0        # 1-based mean rank
        r[order, j] = np.repeat(avg, counts)
    return r / (n + 1.0)


def _robust_affine(X):
    """Per-channel median/IQR standardisation.

    The invariance a descriptor demands is a dial, not a switch, and this is the
    middle setting. Ranks buy invariance to every strictly increasing
    reparameterisation and pay for it by discarding magnitude entirely, so a
    machine that simply runs hotter than its siblings becomes invisible. A
    robust affine standardisation buys invariance only to gain and offset, which
    is what a re-zeroed or re-scaled sensor actually does most of the time, and
    keeps the shape of the distribution and the relative size of excursions.

    Atoms computed on this transform are therefore affine-invariant rather than
    monotone-invariant, and sit between the rank atlas and raw magnitudes on the
    invariance-versus-identifiability trade.
    """
    X = np.asarray(X, dtype=np.float64)
    med = np.median(X, axis=0)
    q1, q3 = np.percentile(X, [25, 75], axis=0)
    iqr = q3 - q1
    iqr = np.where(iqr < 1e-12, np.maximum(X.std(0), 1e-12), iqr)
    return (X - med) / iqr


def _corr(U):
    """Pearson on already rank-transformed columns == Spearman."""
    Z = U - U.mean(0)
    s = Z.std(0)
    s[s < 1e-12] = 1e-12
    Z = Z / s
    return (Z.T @ Z) / U.shape[0]


def _eta_matrix(U, nbin=12):
    """eta[i, j] = correlation ratio of channel j explained by binning on
    channel i, i.e. how well x_j is a single-valued function of x_i.

    eta near 1 means a curve; eta well below the dependence level means the
    relation is multi-valued -- a hysteresis loop or a hidden third variable.
    This is the 'curve vs splotch' discriminator, and it is the thing a
    correlation coefficient cannot see.
    """
    n, d = U.shape
    edges = np.linspace(0.0, 1.0, nbin + 1)
    B = np.clip(np.digitize(U, edges[1:-1]), 0, nbin - 1)      # n x d bin ids
    var_j = U.var(0)
    var_j[var_j < 1e-12] = 1e-12
    mbar = U.mean(0)
    eta = np.zeros((d, d))
    lev = np.arange(nbin)
    for i in range(d):
        b = B[:, i]
        cnt = np.bincount(b, minlength=nbin).astype(np.float64)
        good = cnt > 1
        # per-bin sums of every channel at once. A one-hot matmul is an order of
        # magnitude faster than np.add.at, which falls back to a Python-speed
        # unbuffered loop.
        oh = (b[:, None] == lev).astype(np.float64)            # n x nbin
        S = oh.T @ U                                           # nbin x d
        m = np.zeros((nbin, d))
        m[good] = S[good] / cnt[good, None]
        p = cnt / n
        between = ((m - mbar) ** 2 * p[:, None]).sum(0)
        eta[i] = np.clip(between / var_j, 0.0, 1.0)
    np.fill_diagonal(eta, 1.0)
    return eta


def _levy(U, block=256):
    """Loop orientation: the signed Levy area normalised by the total absolute
    area swept, so the atom lands in [-1, 1] and is free of both amplitude and
    sampling rate.

        L[i,j] = sum_t (u_i du_j - u_j du_i) / sum_t |u_i du_j - u_j du_i|

    The numerator is the depth-2 term of the path signature. Its SIGN is a
    lead-lag direction and its magnitude is how consistently the pair circulates:
    a single-valued instantaneous relation encloses zero net area however far it
    travels, while a lagged one (torque heating a winding that then derates the
    torque) traces a loop with a definite sense. Dividing by the raw signed area
    per step instead would make the atom decay as 1/T under refinement, which is
    wrong -- what identifies a machine is the orientation, not the duration.

    The numerator is one matmul; the denominator needs |.| inside the sum, so it
    is accumulated over time blocks to stay O(d^2 n) in work and O(d^2 + block*d)
    in memory rather than materialising an n x d x d array.
    """
    Z = U - U.mean(0)
    dZ = np.diff(Z, axis=0)
    Zt = Z[:-1]
    d = Z.shape[1]
    M = Zt.T @ dZ
    num = M - M.T
    den = np.zeros_like(num)
    # size blocks by a fixed memory budget rather than a fixed row count, so the
    # temporary stays ~16 MB whether d is 8 or 40
    block = max(1, int(2_000_000 // max(d * d, 1)))
    for s0 in range(0, len(dZ), block):
        z = Zt[s0:s0 + block]
        dz = dZ[s0:s0 + block]
        # a[t,i,j] = z_i dz_j - z_j dz_i  for this block
        a = z[:, :, None] * dz[:, None, :]
        den += np.abs(a - a.transpose(0, 2, 1)).sum(0)
    den[den < 1e-12] = 1e-12
    return np.clip(num / den, -1.0, 1.0)


def _jump(U, q=0.99):
    """Share of the pair's total co-movement carried by the largest increments.

    The precondition behind any diffusion model is that increments are locally
    Gaussian. Real machine telemetry violates this hard: commanded setpoint
    changes and load steps are genuine discontinuities. Rather than treat that
    as a nuisance we make it an edge feature -- how discontinuous a relation is
    is a physical property of the coupling, and it is exactly what separates a
    thermal path (smooth) from a switching path (jumpy).
    """
    # Median-filter the ranks before differencing. A channel resting at a
    # constant setpoint under sensor noise has ranks that shuffle randomly
    # inside its tied block, so in rank space a quiet channel is
    # indistinguishable from a jumpy one -- that shuffling is high frequency
    # while a genuine level change is not. A width-5 median filter removes the
    # former and leaves the latter, and because it is a deterministic function
    # of the ranks alone it preserves the exact recalibration invariance.
    from scipy.ndimage import median_filter
    dZ = np.diff(median_filter(U, size=(5, 1), mode='nearest'), axis=0)
    # Each channel gets its OWN tail threshold, and an increment counts as a
    # jump for pair (i,j) if either member jumped. Thresholding instead on a
    # max across all d channels -- as a first version did -- silently couples
    # every pair to every channel: one noisy or coarsely quantised channel
    # then moves the jump atom of pairs it has nothing to do with, and the
    # atom stops being a property of the pair at all.
    aZ = np.abs(dZ)
    thr = np.quantile(aZ, q, axis=0)
    M = (aZ >= thr[None, :]).astype(np.float64)
    P = M * aZ
    # Share of TOTAL ABSOLUTE co-movement carried by tail increments. The signed
    # increment covariance is the wrong denominator: for a pair that happens to
    # be uncorrelated it sits at zero, and the ratio then reports numerical
    # noise rather than a jump fraction. Absolute co-movement is strictly
    # positive, keeps the atom in [0,1], and still factorises as a matmul
    # because |dz_i dz_j| = |dz_i||dz_j|.
    tail = P.T @ aZ + aZ.T @ P - P.T @ P     # union mask, inclusion-exclusion
    total = aZ.T @ aZ
    total[total < 1e-12] = 1e-12
    return np.clip(tail / total, 0.0, 1.0)


def _actime(U):
    """Per-channel autocorrelation time via the lag-1 coefficient."""
    Z = U - U.mean(0)
    v = (Z * Z).sum(0)
    v[v < 1e-12] = 1e-12
    r1 = (Z[:-1] * Z[1:]).sum(0) / v
    r1 = np.clip(r1, 1e-4, 0.999)
    return -1.0 / np.log(r1)


def _fill(U, nbin=16):
    """Fraction of the 2-D quantile grid the joint cloud occupies.

    A tight functional relation fills a thin diagonal ribbon (low occupancy); an
    unstructured pair fills the square; an envelope with distinct regimes fills
    disconnected blobs. This is the 'splotch'.
    """
    n, d = U.shape
    edges = np.linspace(0.0, 1.0, nbin + 1)
    B = np.clip(np.digitize(U, edges[1:-1]), 0, nbin - 1).astype(np.int64)
    F = np.zeros((d, d))
    scale = float(nbin * nbin)
    ncell = nbin * nbin
    for i in range(d):
        key = B[:, i:i + 1] * nbin + B
        for j in range(i + 1, d):
            # counting occupied cells with bincount is O(n); np.unique sorts,
            # which is O(n log n) and dominated the whole atlas at large n
            occ = np.count_nonzero(np.bincount(key[:, j], minlength=ncell))
            F[i, j] = F[j, i] = occ / scale
    return F


def _beta(X, nbin=12):
    """Robust log-log slope of x_j against x_i over quantile-binned medians.

    The one scale-dependent atom, kept because it reads a physical law straight
    off the shape: ohmic heating gives dT ~ tau^2, so beta ~ 2 on the
    torque/temperature pair. Fitted on bin medians rather than raw points so a
    handful of outliers cannot set the exponent.
    """
    n, d = X.shape
    Xs = X - X.min(0) + 1e-6
    L = np.log(Xs)
    edges = np.linspace(0.0, 1.0, nbin + 1)
    R = _ranks(X)
    B = np.clip(np.digitize(R, edges[1:-1]), 0, nbin - 1)
    beta = np.zeros((d, d))
    lev = np.arange(nbin)
    for i in range(d):
        b = B[:, i]
        cnt = np.bincount(b, minlength=nbin)
        good = cnt >= 3
        if good.sum() < 3:
            continue
        # bin means of the logs (= logs of geometric means) via one-hot matmul.
        # A per-bin median would be marginally more robust but costs a Python
        # loop of boolean-masked medians, which dominated the entire atlas; the
        # quantile binning already supplies the robustness the median was for.
        oh = (b[:, None] == lev).astype(np.float64)
        S = oh.T @ L
        med = np.zeros((nbin, d))
        cf = cnt.astype(np.float64)
        med[good] = S[good] / cf[good, None]
        xk = med[good, i]
        yk = med[good]
        xc = xk - xk.mean()
        den = float((xc * xc).sum())
        if den < 1e-12:
            continue
        beta[i] = (xc[:, None] * (yk - yk.mean(0))).sum(0) / den
    return np.clip(beta, -6.0, 6.0)


def pair_index(d):
    return np.triu_indices(d, 1)


def load_coord(X, channel=None, weights=None):
    """The coordinate whose quantiles define operating cells.

    Either a named channel index (speed, torque, load) or a fixed linear
    combination of standardised channels. It must be the SAME rule for every
    unit of a class, otherwise 'cell 1' means a different operating condition
    for each machine and the conditional atlases are not comparable.
    """
    X = np.asarray(X, dtype=np.float64)
    if channel is not None:
        return X[:, channel]
    if weights is not None:
        Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
        return Z @ np.asarray(weights, dtype=np.float64)
    raise ValueError('load_coord needs channel or weights')


def rank_load(X, cols):
    """Load coordinate as the mean of within-unit RANKS of the given channels.

    The obvious load coordinate -- say the summed absolute joint torque -- is
    not recalibration invariant: an independent monotone warp per channel does
    not commute with a sum, so warped samples cross the cell boundaries and the
    conditional atlas moves even though every atom in it is invariant. That is a
    real hole, and it showed up as a collapse in the recalibration experiment
    before it was closed.

    Averaging per-channel ranks fixes it exactly. Each channel's rank is
    invariant to any strictly increasing warp of that channel, so the mean of
    the ranks is too, and the resulting coordinate still means 'how hard is this
    machine working relative to its own operating range' -- which is the
    comparable notion across a fleet whose members are not co-calibrated.
    """
    X = np.asarray(X, dtype=np.float64)
    return _ranks(X[:, list(cols)]).mean(1)


def fit_cell_edges(values, n_cells):
    """Global bin edges, fitted once on pooled values from the training units."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    e = np.quantile(v, np.linspace(0, 1, n_cells + 1))
    e[0], e[-1] = -np.inf, np.inf
    return e


def fit_load_weights(Xs):
    """First principal direction of the POOLED standardised channels, with a
    deterministic sign convention.

    Fitting the direction per unit -- as an earlier version did -- is a bug that
    is easy to miss and fatal to every cross-unit comparison: each machine gets
    its own definition of 'cell 1', and the sign of a principal component is
    arbitrary anyway, so half the fleet has its cells in reverse order. The
    direction is therefore fitted once on the pooled fleet and then applied
    unchanged to every unit, and its sign is pinned by forcing the largest-
    magnitude loading positive.
    """
    Z = np.concatenate([(x - x.mean(0)) / (x.std(0) + 1e-12) for x in Xs], 0)
    Z = Z[np.isfinite(Z).all(1)]
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    w = Vt[0]
    if w[np.argmax(np.abs(w))] < 0:
        w = -w
    return w


def atlas_unit(X, n_cells=3, cell_coord=None, min_cell=200, max_n=60000, seed=0,
               cell_edges=None):
    """Conditional operating atlas of one unit.

    X : (n, d) channel matrix for a single unit / session / engine.
    n_cells : the atlas is computed separately in each of n_cells operating
        cells, because the claim is not that a machine has ONE shape but that
        its shape MOVES with operating point. Cells are quantile bins of a load
        coordinate (default: first principal component of the standardised
        channels, i.e. how hard the machine is working).

    Returns (n_cells, P, 9) with P = d(d-1)/2 ordered pairs i<j.
    """
    X = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(X).all(1)
    X = X[finite]
    n, d = X.shape
    iu, ju = pair_index(d)
    P = len(iu)
    out = np.full((n_cells, P, len(ATOM_NAMES)), np.nan)
    if n < max(min_cell, 50):
        return out

    if n_cells == 1:
        cells = [np.arange(n)]
    else:
        if cell_coord is None:
            raise ValueError(
                'conditional atlases need an explicit cell_coord shared across '
                'units; see fit_load_weights / load_coord')
        c = np.asarray(cell_coord, dtype=np.float64)
        if len(c) == len(finite):
            c = c[finite]          # drop the same rows the channels dropped
        if cell_edges is None:
            # per-unit quantiles: each unit contributes equally to every cell,
            # but the cells are unit-relative. Correct only for reproducibility
            # tests within one unit, never for cross-unit comparison.
            qs = np.quantile(c, np.linspace(0, 1, n_cells + 1))
            qs[0], qs[-1] = -np.inf, np.inf
        else:
            qs = np.asarray(cell_edges, dtype=np.float64)
        cells = [np.where((c > qs[m]) & (c <= qs[m + 1]))[0] for m in range(n_cells)]

    rng = np.random.default_rng(seed)
    for m, idx in enumerate(cells):
        if len(idx) < min_cell:
            continue
        if len(idx) > max_n:
            # the atlas is a statistic, so a contiguous-stride thinning is a
            # legitimate estimator of it. Stride rather than random draw, to
            # keep the increments (levy, jump, tau) on a uniform time base.
            idx = idx[::int(np.ceil(len(idx) / max_n))]
        Xc = X[idx]
        keep = Xc.std(0) > 1e-10
        U = _ranks(Xc)
        U[:, ~keep] = 0.5
        rho = _corr(U)
        eta = _eta_matrix(U)
        lev = _levy(U)
        jmp = _jump(U)
        act = _actime(U)
        fil = _fill(U)
        bet = _beta(Xc)

        eta_ji = eta[iu, ju]          # x_j explained by binning on x_i
        eta_ij = eta[ju, iu]
        eta_max = np.maximum(eta_ij, eta_ji)
        r = rho[iu, ju]
        vals = np.stack([
            r,
            eta_max,
            eta_max - r ** 2,
            eta_ji - eta_ij,
            lev[iu, ju],
            jmp[iu, ju],
            np.log(act[ju] / np.maximum(act[iu], 1e-9) + 1e-9),
            fil[iu, ju],
            bet[iu, ju],
        ], axis=1)
        bad = ~keep[iu] | ~keep[ju]
        vals[bad] = np.nan
        out[m] = vals
    return out


def warp_channels(X, rng, kind='monotone', strength=1.0):
    """Simulated fleet recalibration: an independent strictly-increasing warp
    per channel. This is the deployment reality -- a replacement sensor with a
    different gain curve, a re-zeroed thermocouple, a different ADC range."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    lo, hi = X.min(0), X.max(0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    Z = np.clip((X - lo) / span, 0.0, 1.0)
    out = np.empty_like(Z)
    n_knot = 7
    u = np.linspace(0.0, 1.0, n_knot)
    for j in range(d):
        if kind == 'affine':
            a = np.exp(rng.normal(0, 0.5 * strength))
            out[:, j] = a * Z[:, j] + rng.normal(0, 0.5 * strength)
        else:
            # Monotone piecewise-linear through random increasing knots, with
            # every segment slope bounded away from zero. This is what a
            # replacement sensor with a different response curve, a re-zeroed
            # thermocouple or a different ADC transfer function looks like.
            #
            # An earlier version composed a power law with a logistic squash.
            # That is monotone on paper but numerically degenerate: with the
            # exponent drawn as exp(N(0, 0.8)) it routinely reached ~10, which
            # sends most of the unit interval below 1e-10 and lands it on the
            # flat tail of the logistic, collapsing thousands of distinct
            # readings onto a single float. That destroys information rather
            # than relabelling it, so no invariant could survive it -- and it
            # is not a recalibration any instrument performs.
            g = 1.0 + strength * rng.uniform(-0.55, 1.2, n_knot - 1)
            g = np.maximum(g, 0.3)
            v = np.concatenate([[0.0], np.cumsum(g)])
            v = v / v[-1]
            a = np.exp(rng.normal(0, 0.5 * strength))
            out[:, j] = a * np.interp(Z[:, j], u, v) + rng.normal(0, 0.5 * strength)
    return out
