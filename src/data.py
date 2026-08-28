"""
Dataset loaders and the coordinate-scramble protocol.

PMSM  (Kaggle wkirgsn/electric-motor-temperature): 1.33M samples at 2 Hz from a
      52 kW permanent-magnet synchronous machine on a test bench, grouped into
      69 independent measurement sessions. Coupled electromagnetic -> thermal ->
      mechanical physics with real, non-stationary excitation.
      Manifold state (8 d): the operating condition only -- drive voltages and
      currents, speed, torque, ambient and coolant temperature. The four
      internal temperatures being predicted are never part of the state, so the
      geometric coordinates cannot leak the target.
      Targets (4): rotor magnet temperature pm, stator yoke / tooth / winding.

CMAPSS (Kaggle behrad3d/nasa-cmaps): NASA turbofan run-to-failure simulation.
      FD002 and FD004 carry six discrete flight regimes, which is the closest
      public analogue of a piecewise-charted operating manifold.
      Manifold state (3 + 14 d): operational settings and the informative
      sensors. Target: remaining useful life.
"""

import os
import numpy as np
import pandas as pd
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')

PMSM_STATE = ['u_q', 'u_d', 'i_q', 'i_d', 'motor_speed', 'torque', 'ambient', 'coolant']
PMSM_TARGET = ['pm', 'stator_yoke', 'stator_tooth', 'stator_winding']


# ----------------------------------------------------------------------------
# PMSM
# ----------------------------------------------------------------------------

def load_pmsm(max_profiles=None):
    df = pd.read_csv(os.path.join(ROOT, 'pmsm', 'measures_v2.csv'))
    if max_profiles is not None:
        keep = sorted(df.profile_id.unique())[:max_profiles]
        df = df[df.profile_id.isin(keep)]
    return df


def pmsm_splits(df, seed=0):
    """Session-wise split. Sessions are physically independent bench runs, so
    splitting by profile_id is the only leakage-free protocol."""
    ids = np.array(sorted(df.profile_id.unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n = len(ids)
    return dict(train=ids[:int(.6 * n)], val=ids[int(.6 * n):int(.75 * n)], test=ids[int(.75 * n):])


def pmsm_arrays(df, ids, ewma_spans=(1320, 3360, 6360, 9480), warp=None):
    """Per-session arrays. EWMA features of the state are the standard strong
    baseline for this benchmark -- they give a causal memory of the thermal
    excitation without which no memoryless model can succeed.

    `warp` is applied to the raw state *before* any feature is built, because a
    miscalibrated sensor corrupts the reading at acquisition and every derived
    feature inherits it. Warping the features afterwards would be a strictly
    easier and less honest problem."""
    Xs, Ys, Ss, G = [], [], [], []
    for pid in ids:
        d = df[df.profile_id == pid]
        S = d[PMSM_STATE].to_numpy(np.float64)
        if warp is not None:
            S = warp(S)
        Y = d[PMSM_TARGET].to_numpy(np.float64)
        feats = [S]
        sd = pd.DataFrame(S, columns=PMSM_STATE)
        for sp in ewma_spans:
            feats.append(sd.ewm(span=sp).mean().to_numpy(np.float64))
        for sp in ewma_spans[:2]:
            feats.append(sd.ewm(span=sp).std().fillna(0).to_numpy(np.float64))
        Xs.append(np.concatenate(feats, 1))
        Ys.append(Y)
        Ss.append(S)
        G.append(np.full(len(d), pid))
    return (np.concatenate(Xs), np.concatenate(Ys),
            np.concatenate(Ss), np.concatenate(G))


def pmsm_transitions(S, G, stride=1):
    """(s_t, ds_t) pairs that stay inside a session."""
    same = G[:-1] == G[1:]
    return S[:-1][same], (S[1:] - S[:-1])[same]


# ----------------------------------------------------------------------------
# CMAPSS
# ----------------------------------------------------------------------------

CMAPSS_COLS = (['unit', 'cycle', 'set1', 'set2', 'set3']
               + [f's{i}' for i in range(1, 22)])
# sensors that are not constant across FD002/FD004
CMAPSS_SENSORS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12',
                  's13', 's14', 's15', 's17', 's20', 's21']
CMAPSS_STATE = ['set1', 'set2', 'set3'] + CMAPSS_SENSORS


def load_cmapss(fd='FD004'):
    p = os.path.join(ROOT, 'cmapss', 'CMaps')
    tr = pd.read_csv(os.path.join(p, f'train_{fd}.txt'), sep=r'\s+', header=None, names=CMAPSS_COLS)
    te = pd.read_csv(os.path.join(p, f'test_{fd}.txt'), sep=r'\s+', header=None, names=CMAPSS_COLS)
    rul = pd.read_csv(os.path.join(p, f'RUL_{fd}.txt'), sep=r'\s+', header=None, names=['RUL'])
    tr['RUL'] = tr.groupby('unit').cycle.transform('max') - tr.cycle
    last = te.groupby('unit').cycle.transform('max')
    te['RUL'] = last - te.cycle + rul.RUL.to_numpy()[te.unit.to_numpy() - 1]
    for d in (tr, te):
        d['RUL'] = d.RUL.clip(upper=125)      # standard piecewise-linear RUL cap
    return tr, te


def cmapss_regime(df):
    """The six flight regimes are exactly recoverable by clustering set1..set3."""
    from sklearn.cluster import KMeans
    X = df[['set1', 'set2', 'set3']].to_numpy()
    return KMeans(6, n_init=10, random_state=0).fit_predict(X)


# ----------------------------------------------------------------------------
# coordinate scramble
# ----------------------------------------------------------------------------

class Scramble:
    """A smooth, invertible relabelling of the sensor coordinates.

    Two regimes, because they are different physical situations and, as it
    turns out, different problems:

    'channelwise'  each sensor is independently mis-scaled, offset and warped,
                   x_i -> g_i * sinh(a_i * asinh(x_i)) + o_i. This is ordinary
                   fleet miscalibration. It is a *marginal* distortion, so a
                   per-channel quantile match can undo it whenever the target
                   duty-cycle distribution matches the source.

    'mixing'       additionally couples the channels,
                   T(x) = A (x + c tanh(B x)) + b, before the channel warps.
                   No per-channel correction can undo it. This is the stronger
                   test of whether the routing coordinates are genuinely
                   invariant rather than merely robust to rescaling.

    In both cases T is a diffeomorphism, so the state manifold is unchanged and
    only its coordinates are relabelled. asinh/sinh is used rather than a power
    law because it is smooth, strictly monotone and sign-symmetric through the
    origin, which the state variables (bipolar currents, voltages, torque) need.
    """

    def __init__(self, d, seed=0, mode='mixing', c=0.35, strength=1.0):
        rng = np.random.RandomState(seed)
        self.d, self.mode, self.c = d, mode, c
        self.gain = np.exp(rng.uniform(-0.4, 0.4, d) * strength)
        self.shape = np.exp(rng.uniform(-0.35, 0.35, d) * strength)
        self.off = rng.randn(d) * 0.15 * strength
        if mode == 'mixing':
            self.A = np.eye(d) + strength * rng.randn(d, d) / np.sqrt(d)
            self.B = rng.randn(d, d) / np.sqrt(d)
            self.b = rng.randn(d) * 0.1 * strength
            # I + c B sech^2 is worst-conditioned at sech^2 = 1
            self.min_sv = float(np.linalg.svd(
                self.A @ (np.eye(d) + c * self.B), compute_uv=False).min())
        else:
            self.min_sv = float(self.gain.min())

    def __call__(self, X):
        Y = X
        if self.mode == 'mixing':
            Y = (self.A @ (Y + self.c * np.tanh(Y @ self.B.T)).T).T + self.b
        return self.gain * np.sinh(self.shape * np.arcsinh(Y)) + self.off


# ---- unsupervised defences available to a raw-coordinate router -------------

def coral(Xt, Xs, eps=1e-6):
    """Match target second moments to source (CORAL). Unsupervised."""
    ms, mt = Xs.mean(0), Xt.mean(0)
    Cs = np.cov(Xs - ms, rowvar=False) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt - mt, rowvar=False) + eps * np.eye(Xt.shape[1])

    def isqrt(C):
        w, V = np.linalg.eigh(C)
        return V @ np.diag(w.clip(eps) ** -0.5) @ V.T

    def sqrt(C):
        w, V = np.linalg.eigh(C)
        return V @ np.diag(w.clip(eps) ** 0.5) @ V.T

    return (Xt - mt) @ isqrt(Ct) @ sqrt(Cs) + ms


def quantile_match(Xt, Xs, n_q=512):
    """Per-channel monotone map sending the target marginals onto the source
    marginals. Exactly inverts any channelwise warp *provided* the underlying
    operating distributions agree; conflates calibration with duty cycle when
    they do not, which is the case that matters."""
    q = np.linspace(0, 1, n_q)
    out = np.empty_like(Xt)
    for j in range(Xt.shape[1]):
        out[:, j] = np.interp(Xt[:, j], np.quantile(Xt[:, j], q),
                              np.quantile(Xs[:, j], q))
    return out


def standardize(train, *others):
    m, s = train.mean(0), train.std(0) + 1e-8
    return [(train - m) / s] + [(o - m) / s for o in others]
