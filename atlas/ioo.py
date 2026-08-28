"""
The Index of Operations and the Index of Operators.

    IOO                 every point the device visits, in every live dimension,
                        through space and time. The body itself, held as points
                        plus the local geometry at each of them, not a summary.

    Index of Operators  the library of BASE IOOs, one per kind of machine. You
                        look up the base for your machine's class, and the
                        landmarks shift to fit your particular unit. The shift
                        is the tuning, and its coefficients are the unit's
                        identity.

The deformation is what makes this a usable object rather than a picture. A
base body has K landmarks covering the class envelope. A specific device's data
moves those landmarks; the displacement field, compressed over a fleet, spans a
handful of directions, so a new device is fitted by estimating that handful of
numbers from a short record instead of learning its manifold from scratch.

What the fitted body is for:

    project     snap a noisy reading onto the manifold the machine is actually
                allowed to occupy, which is a physical prior no per-channel
                filter can express
    residual    the part of a reading that leaves the body. On a healthy unit
                this is sensor noise; when it grows the machine is doing
                something its class does not do
    reachable   the local tangent space says which combinations of variables
                can move next, which is the control-relevant statement
"""

import numpy as np
import manifold_local as ml


class IOO:
    """One device's operating body."""

    def __init__(self, Z, weights, live, geo=None, landmarks=None,
                 channels=None, name=''):
        self.Z = Z                  # (n, d_live) embedded points
        self.weights = weights      # per-live-channel weight actually applied
        self.live = live            # boolean mask into the original channels
        self.geo = geo              # local geometry fields at probe points
        self.landmarks = landmarks  # (K, d_live) the chart
        self.channels = channels
        self.name = name

    @property
    def dim_live(self):
        return self.Z.shape[1]

    def __repr__(self):
        m = (np.median(self.geo['fields']['dim']) if self.geo else float('nan'))
        return (f'<IOO {self.name!r} n={len(self.Z)} live={self.dim_live}'
                f'/{len(self.live)} median_dim={m:.2f}>')


def _embed(X, snr_floor=0.02):
    """Rank-embed, weight by channel signal, and drop dead channels.

    Dropping rather than downweighting matters: a zero column leaves a zero
    eigenvalue that drags the local noise floor down and inflates every
    dimension estimate. This is the step that makes the body independent of how
    many unused channels the logger happened to record.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X[np.isfinite(X).all(1)]
    w = np.sqrt(np.maximum(ml.channel_snr(X) - snr_floor, 0.0))
    live = w > 1e-6
    if live.sum() < 2:
        live = np.ones(X.shape[1], dtype=bool)
        w = np.ones(X.shape[1])
    Z = ml._ranks(X)[:, live] * w[live]
    return Z, w[live], live


def build_ioo(X, name='', k=48, n_probe=2500, n_landmarks=160, seed=0,
              channels=None):
    """Build one device's IOO from its raw telemetry."""
    Z, w, live = _embed(X)
    geo = ml.local_geometry(X, k=k, n_probe=n_probe, seed=seed,
                            embed='rank', snr_weight=True)
    L = _landmarks(Z, n_landmarks, seed)
    return IOO(Z, w, live, geo, L, channels, name)


def _landmarks(Z, K, seed=0):
    """K points covering the body, by k-means on the embedded cloud.

    Landmarks are the chart: they are what shifts when the base is fitted to a
    unit, so they need to cover the envelope rather than track density, which
    is why this is k-means and not a random subsample.
    """
    from sklearn.cluster import MiniBatchKMeans
    K = int(min(K, max(2, len(Z) // 10)))
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=5,
                         batch_size=1024).fit(Z)
    return km.cluster_centers_


class OperatorIndex:
    """The library of base IOOs, one per machine class, plus the deformation
    basis that fits a base to a specific device."""

    def __init__(self):
        self.base = {}          # class -> (landmarks, channels)
        self.basis = {}         # class -> (mean_disp, components, sdevs)

    def add_class(self, name, units, n_landmarks=160, seed=0):
        """units: list of raw (n, d) arrays from devices of this class."""
        Zs = [_embed(u)[0] for u in units]
        d = min(z.shape[1] for z in Zs)
        Zs = [z[:, :d] for z in Zs]
        base_L = _landmarks(np.concatenate(Zs), n_landmarks, seed)
        # each unit moves the landmarks; the displacement field is the tuning
        D = np.stack([self._displace(base_L, z) for z in Zs])   # (u, K, d)
        F = D.reshape(len(D), -1)
        mu = F.mean(0)
        Fc = F - mu
        # a fleet gives at most (u-1) independent deformation directions
        r = max(1, min(len(F) - 1, 12))
        U, s, Vt = np.linalg.svd(Fc, full_matrices=False)
        self.base[name] = base_L
        self.basis[name] = (mu, Vt[:r], s[:r] / np.sqrt(max(len(F) - 1, 1)))
        return dict(n_units=len(units), K=len(base_L), d=d, rank=r,
                    var_explained=float((s[:r] ** 2).sum() /
                                        max((s ** 2).sum(), 1e-12)))

    @staticmethod
    def _displace(base_L, Z, cover_q=0.60, return_cover=False):
        """How far each base landmark has to move to sit on this device's body.

        Correspondence is by nearest data point rather than by re-clustering,
        because two independent k-means runs return landmarks in arbitrary and
        unrelated order, and differencing those would measure the labelling
        rather than the machine.

        Landmarks the device never visits have to be masked, and this is not a
        refinement: without it, a landmark sitting in a region the device never
        entered still gets pulled to whatever its nearest points happen to be,
        tens of neighbourhood radii away, and those meaningless displacements
        dominate the field. That is why the first version's fitted body scored
        WORSE than the undeformed base on every held-out device. Coverage is
        judged against the typical landmark spacing, so a device is only
        allowed to move the parts of the body it actually occupies.
        """
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=16).fit(Z)
        dist, idx = nn.kneighbors(base_L)
        near = dist[:, 0]
        scale = np.median(near) + 1e-9
        covered = near <= max(np.quantile(near, cover_q), 2.0 * scale)
        wgt = 1.0 / (dist + 1e-9)
        wgt /= wgt.sum(1, keepdims=True)
        target = np.einsum('kn,knd->kd', wgt, Z[idx])
        disp = target - base_L
        disp[~covered] = 0.0          # unvisited: this device says nothing here
        if return_cover:
            return disp, covered
        return disp

    def fit_unit(self, name, X):
        """Fit the class base to one device. Returns its code and fitted body."""
        if name not in self.base:
            raise KeyError(f'no base IOO for class {name!r}')
        base_L = self.base[name]
        mu, comps, sd = self.basis[name]
        Z, w, live = _embed(X)
        d = base_L.shape[1]
        Z = Z[:, :d] if Z.shape[1] >= d else np.pad(
            Z, ((0, 0), (0, d - Z.shape[1])))
        disp = self._displace(base_L, Z).reshape(-1)
        code = comps @ (disp - mu)
        fitted = base_L + (mu + code @ comps).reshape(base_L.shape)
        return dict(code=code, code_norm=code / (sd + 1e-12),
                    landmarks=fitted, base=base_L, Z=Z,
                    residual=float(np.linalg.norm(disp - mu - code @ comps)
                                   / np.sqrt(len(disp))))

    def project(self, fit, Y):
        """Snap observations onto the fitted body, and report what fell off."""
        from sklearn.neighbors import NearestNeighbors
        L = fit['landmarks']
        nn = NearestNeighbors(n_neighbors=min(8, len(L))).fit(L)
        dist, idx = nn.kneighbors(Y)
        wgt = 1.0 / (dist + 1e-9)
        wgt /= wgt.sum(1, keepdims=True)
        proj = np.einsum('kn,knd->kd', wgt, L[idx])
        return proj, np.linalg.norm(Y - proj, axis=1)
