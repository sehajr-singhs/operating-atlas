"""
The IOO as local geometry on a shared chart.

Two failed attempts sit behind this file and they were both the same mistake.
Parameterising a device by how far the class landmarks MOVE to reach its data
gives a displacement field dominated by sampling noise: on 40 motor sessions and
again on 80 distinct robots, the deformed body was farther from held-out data
than the undeformed base on every single device, and the code retrieved the
device at chance.

What differs between two machines of a kind is not where a centroid drifted. It
is how the body is SHAPED where they both operate: how many degrees of freedom
are live there, how sharply it curves, where it tears. So the chart is fixed by
the class, and what varies per device is the local geometry measured at each
chart location.

    chart       K landmarks covering the class envelope, in a SHARED embedding
    IOO         for one device, the local geometry at each landmark it visits,
                plus a coverage mask saying which parts of the class envelope
                that device actually uses
    operators   the library of charts, one per class

The embedding has to be shared, which the earlier version also got wrong.
Rank-transforming each device against its own history normalises away exactly
the differences between devices. Here the pooled training data defines the
transform and every device is mapped through that same one.
"""

import numpy as np
import manifold_local as ml

GEOM = ['dim', 'curv', 'tear', 'density', 'v_tan', 'v_norm', 'circ']


class SharedEmbedding:
    """Per-channel monotone map fitted once on the class, applied to everyone.

    Keeps the recalibration robustness of a rank transform, since it is still a
    monotone map of each channel, while leaving devices comparable, because it
    is the SAME map for all of them. A per-device rank transform is what erased
    the payload differences in the previous attempt.
    """

    def __init__(self, n_grid=512, snr_floor=0.02):
        self.n_grid = n_grid
        self.snr_floor = snr_floor

    def fit(self, Xs):
        X = np.concatenate([np.asarray(x, dtype=np.float64) for x in Xs])
        X = X[np.isfinite(X).all(1)]
        snr = np.mean([ml.channel_snr(np.asarray(x, dtype=np.float64))
                       for x in Xs], axis=0)
        self.w_ = np.sqrt(np.maximum(snr - self.snr_floor, 0.0))
        self.live_ = self.w_ > 1e-6
        if self.live_.sum() < 2:
            self.live_ = np.ones(X.shape[1], dtype=bool)
            self.w_ = np.ones(X.shape[1])
        q = np.linspace(0, 1, self.n_grid)
        self.knots_ = [np.quantile(X[:, j], q) for j in np.where(self.live_)[0]]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = X[np.isfinite(X).all(1)]
        cols = np.where(self.live_)[0]
        Z = np.empty((len(X), len(cols)))
        q = np.linspace(0, 1, self.n_grid)
        for a, j in enumerate(cols):
            Z[:, a] = np.interp(X[:, j], self.knots_[a], q)
        return Z * self.w_[self.live_]


def _local_fields(Z, V, centres, k=48, eps=1e-12):
    """Local geometry of the body Z, evaluated AT the given chart locations."""
    from sklearn.neighbors import NearestNeighbors
    n, d = Z.shape
    k = int(min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    dist, idx = nn.kneighbors(centres)
    K = len(centres)
    F = np.full((K, len(GEOM)), np.nan)
    for a in range(K):
        nb = Z[idx[a]]
        Y = nb - nb.mean(0)
        try:
            _, s, Vt = np.linalg.svd(Y, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        lam = (s ** 2) / max(len(nb) - 1, 1)
        floor = max(lam[-1], 1e-6 * lam[0]) + eps
        r = lam / floor
        T = 10.0
        s0 = 1.0 / (1.0 + T)
        m_eff = float(np.clip((np.sum(r / (r + T)) - d * s0) / (1 - s0), 0, d))
        m = int(max(1, min(d - 1, round(m_eff))))
        tan = np.sqrt(lam[:m].sum() + eps)
        nor = np.sqrt(lam[m:].sum() + eps)
        rad = dist[a].mean() + eps
        vv = V[idx[a]]
        vt = vv @ Vt[:m].T
        F[a, 0] = m_eff
        F[a, 1] = nor / (tan * rad + eps)
        F[a, 2] = np.linalg.norm(nb.mean(0) - centres[a]) / rad
        F[a, 3] = -np.log(rad)
        F[a, 4] = np.linalg.norm(vt, axis=1).mean()
        F[a, 5] = np.sqrt(np.maximum(
            (vv ** 2).sum(1) - (vt ** 2).sum(1), 0)).mean()
        if m >= 2:
            w = (nb - nb.mean(0)) @ Vt[:2].T
            dw = np.diff(w, axis=0)
            num = (w[:-1, 0] * dw[:, 1] - w[:-1, 1] * dw[:, 0])
            F[a, 6] = num.sum() / (np.abs(num).sum() + eps)
    return F, dist[:, 0]


def _velocity(Z):
    V = np.zeros_like(Z)
    V[1:-1] = 0.5 * (Z[2:] - Z[:-2])
    V[0], V[-1] = Z[1] - Z[0], Z[-1] - Z[-2]
    return V


class OperatorChart:
    """One class: a shared embedding, a chart, and per-device IOOs on it."""

    def __init__(self, n_landmarks=120, k=48, seed=0):
        self.K = n_landmarks
        self.k = k
        self.seed = seed

    def fit_class(self, Xs):
        self.emb_ = SharedEmbedding().fit(Xs)
        Zs = [self.emb_.transform(x) for x in Xs]
        from sklearn.cluster import MiniBatchKMeans
        pool = np.concatenate(Zs)
        sub = pool[::max(1, len(pool) // 60000)]
        km = MiniBatchKMeans(n_clusters=self.K, random_state=self.seed,
                             n_init=5, batch_size=2048).fit(sub)
        self.chart_ = km.cluster_centers_
        spacing = np.median(np.linalg.norm(
            self.chart_[:, None] - self.chart_[None], axis=-1)
            + np.eye(self.K) * 1e9, axis=1)
        self.cover_r_ = float(np.median(spacing))
        return self

    def ioo(self, X):
        """The device's IOO: local geometry on the class chart, plus coverage."""
        Z = self.emb_.transform(X)
        V = _velocity(Z)
        F, near = _local_fields(Z, V, self.chart_, k=self.k)
        covered = near <= self.cover_r_
        return dict(fields=F, covered=covered, Z=Z)

    def descriptor(self, io):
        """Flatten to a fixed-length vector over the chart's COMMON CORE.

        Two mistakes are being corrected here, and the second was severe enough
        to push same-device retrieval below chance, at a percentile rank of 40.5
        against 50 for a coin.

        Coverage is workload, not device. Every episode draws its own duty cycle,
        so two halves of one machine visit different parts of the envelope and
        carry different coverage masks. Putting the mask in the descriptor
        therefore encodes which duty cycle ran, and it makes a machine look less
        like itself than like a stranger. The mask is gone.

        Filling uncovered cells with the class median was the other half of it:
        every sparsely covered device collapses onto the same fill values and so
        onto every other sparsely covered device. Rather than invent geometry
        where a device never went, the descriptor is restricted to the core of
        the chart that essentially all devices visit, where every one of them
        has a real measurement.
        """
        F = io['fields']
        core = getattr(self, 'core_', np.ones(len(F), dtype=bool))
        Fc = np.nan_to_num(F[core], nan=0.0, posinf=0.0, neginf=0.0)
        return Fc.reshape(-1)

    def set_core(self, ioos, min_frac=0.9):
        """Chart cells that nearly every device actually visits."""
        C = np.stack([i['covered'] & np.isfinite(i['fields']).all(1)
                      for i in ioos])
        frac = C.mean(0)
        core = frac >= min_frac
        if core.sum() < 8:                    # fall back to the best cells
            core = frac >= np.quantile(frac, 0.75)
        self.core_ = core
        self.core_frac_ = float(core.mean())
        return self
