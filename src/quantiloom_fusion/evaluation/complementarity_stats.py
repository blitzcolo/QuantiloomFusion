"""Array-level statistics for the complementarity analysis (no dataset access).

* ``pair_frame``: per-frame de-meaning of the target and LWIR features, then a
  seeded pixel subsample. De-meaning uses the full pair set of the frame.
* Regressors: ridge on degree-2 polynomial features, and kNN (scipy cKDTree).
  Both are hand-written; scikit-learn is not a project dependency.
* ``cmi_estimate``: binned plug-in I(Y; L | V) in nats, bias-corrected by a
  stratified permutation null (L shuffled within each V stratum). Plug-in CMI is
  biased upward by roughly (|Y|-1)(|L|-1)|V| / (2N); the null mean estimates that
  bias for the actual cell counts. Neighbouring pixels are autocorrelated, so
  the effective N is smaller than the pixel count and the null under-covers.
* Folds: leave-one-group-out by split_group, filtered per condition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from ..dataset.types import Condition

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

VIS_FEATURES = ("r", "g", "b", "lum_ifov", "lum_coarse", "grad")
LWIR_FEATURES = ("tapp", "tapp_coarse", "tapp_grad")


@dataclass(frozen=True)
class FrameId:
    sample_id: str
    group: str
    condition: Condition


@dataclass(frozen=True)
class PairArrays:
    """Raw pair-pixel arrays of one frame: y (K), x_vis (n, kv), x_lwir (n, kl), label (n,)."""

    y: FloatArray
    x_vis: FloatArray
    x_lwir: FloatArray
    label: IntArray


@dataclass(frozen=True)
class PairFrame:
    ident: FrameId
    y: FloatArray  # surface temperature minus the frame mean over all pair pixels
    x_vis: FloatArray
    x_lwir: FloatArray  # columns minus their frame mean
    label: IntArray


def pair_frame(
    ident: FrameId, arrays: PairArrays, max_px: int, rng: np.random.Generator
) -> PairFrame:
    """De-mean target and LWIR features per frame, then subsample to max_px pixels."""
    n = arrays.y.shape[0]
    y = arrays.y - arrays.y.mean()
    x_lwir = arrays.x_lwir - arrays.x_lwir.mean(axis=0)
    idx = np.sort(rng.choice(n, max_px, replace=False)) if n > max_px else np.arange(n)
    return PairFrame(ident, y[idx], arrays.x_vis[idx], x_lwir[idx], arrays.label[idx])


def _stack(frames: Sequence[PairFrame]) -> PairArrays:
    return PairArrays(
        np.concatenate([f.y for f in frames]),
        np.concatenate([f.x_vis for f in frames]),
        np.concatenate([f.x_lwir for f in frames]),
        np.concatenate([f.label for f in frames]),
    )


def _standardise(train: FloatArray, test: FloatArray) -> tuple[FloatArray, FloatArray]:
    mu = train.mean(axis=0)
    sd = np.where(train.std(axis=0) > 1e-12, train.std(axis=0), 1.0)
    return (train - mu) / sd, (test - mu) / sd


def ridge_poly2_predict(
    x_train: FloatArray, y_train: FloatArray, x_test: FloatArray, alpha: float = 1.0
) -> FloatArray:
    """Ridge regression on [z, z_i z_j] (standardised z); intercept unpenalised."""
    z_tr, z_te = _standardise(x_train, x_test)
    iu = np.triu_indices(z_tr.shape[1])

    def expand(z: FloatArray) -> FloatArray:
        return np.hstack([z, z[:, iu[0]] * z[:, iu[1]]])

    a_tr, a_te = expand(z_tr), expand(z_te)
    mu_a, mu_y = a_tr.mean(axis=0), y_train.mean()
    a_c = a_tr - mu_a
    gram = a_c.T @ a_c + alpha * np.eye(a_c.shape[1])
    w = np.linalg.solve(gram, a_c.T @ (y_train - mu_y))
    return (a_te - mu_a) @ w + mu_y


def knn_predict(
    x_train: FloatArray, y_train: FloatArray, x_test: FloatArray, k: int = 20
) -> FloatArray:
    """Mean target of the k nearest standardised training points (Euclidean)."""
    z_tr, z_te = _standardise(x_train, x_test)
    _, nn = cKDTree(z_tr).query(z_te, k=k, workers=-1)
    return y_train[nn].mean(axis=1)


@dataclass(frozen=True)
class RegressionScores:
    """Held-out mean squared residuals (K^2) of the de-meaned surface temperature."""

    n_test: int
    var_total: float  # mean(y^2): residual of the zero (frame-mean) predictor
    ridge_vis: float
    ridge_both: float
    knn_vis: float
    knn_both: float


def regression_scores(
    train: Sequence[PairFrame], test: Sequence[PairFrame], alpha: float = 1.0, k: int = 20
) -> RegressionScores:
    tr, te = _stack(train), _stack(test)
    both_tr = np.hstack([tr.x_vis, tr.x_lwir])
    both_te = np.hstack([te.x_vis, te.x_lwir])

    def mse(pred: FloatArray) -> float:
        return float(np.mean((te.y - pred) ** 2))

    return RegressionScores(
        int(te.y.size),
        float(np.mean(te.y**2)),
        mse(ridge_poly2_predict(tr.x_vis, tr.y, te.x_vis, alpha)),
        mse(ridge_poly2_predict(both_tr, tr.y, both_te, alpha)),
        mse(knn_predict(tr.x_vis, tr.y, te.x_vis, k)),
        mse(knn_predict(both_tr, tr.y, both_te, k)),
    )


def quantile_codes(train_values: FloatArray, values: FloatArray, bins: int) -> IntArray:
    """Bin index of values using equal-frequency edges estimated on train_values."""
    edges = np.quantile(train_values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    return np.searchsorted(edges, values).astype(np.intp)


def _entropy(*codes: IntArray) -> float:
    key = np.zeros(codes[0].shape, dtype=np.int64)
    for c in codes:
        key = key * (int(c.max()) + 1) + c
    counts = np.bincount(np.unique(key, return_inverse=True)[1])
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def conditional_mutual_information(y: IntArray, x: IntArray, v: IntArray) -> float:
    """Plug-in I(Y; X | V) in nats from integer codes: H(Y,V)+H(X,V)-H(Y,X,V)-H(V)."""
    return _entropy(y, v) + _entropy(x, v) - _entropy(y, x, v) - _entropy(v)


@dataclass(frozen=True)
class CmiEstimate:
    plugin: float  # nats
    null_mean: float  # plug-in on within-stratum shuffled X (bias estimate)
    corrected: float  # plugin - null_mean
    n: int


def cmi_estimate(
    y: IntArray, x: IntArray, v: IntArray, n_perm: int, rng: np.random.Generator
) -> CmiEstimate:
    plugin = conditional_mutual_information(y, x, v)
    strata = [np.flatnonzero(v == u) for u in np.unique(v)]
    nulls = []
    for _ in range(n_perm):
        x_perm = x.copy()
        for idx in strata:
            x_perm[idx] = x[idx][rng.permutation(idx.size)]
        nulls.append(conditional_mutual_information(y, x_perm, v))
    null_mean = float(np.mean(nulls))
    return CmiEstimate(plugin, null_mean, plugin - null_mean, int(y.size))


@dataclass(frozen=True)
class CmiBins:
    y: int = 12
    lwir: int = 12
    vis_fine: int = 8
    vis_coarse: int = 4


def cmi_for_fold(
    train: Sequence[PairFrame],
    test: Sequence[PairFrame],
    bins: CmiBins,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[CmiEstimate, CmiEstimate]:
    """(I(T; LWIR | VIS), I(material; LWIR | VIS)) on the held-out frames.

    VIS is the pair (lum_ifov, lum_coarse) binned jointly; LWIR is the de-meaned
    apparent temperature; bin edges come from the training frames.
    """
    tr, te = _stack(train), _stack(test)
    i_fine, i_coarse = VIS_FEATURES.index("lum_ifov"), VIS_FEATURES.index("lum_coarse")
    i_l = LWIR_FEATURES.index("tapp")

    def codes(col_tr: FloatArray, col_te: FloatArray, n: int) -> IntArray:
        return quantile_codes(col_tr, col_te, n)

    v = codes(tr.x_vis[:, i_fine], te.x_vis[:, i_fine], bins.vis_fine) * bins.vis_coarse + codes(
        tr.x_vis[:, i_coarse], te.x_vis[:, i_coarse], bins.vis_coarse
    )
    x = codes(tr.x_lwir[:, i_l], te.x_lwir[:, i_l], bins.lwir)
    y_temperature = codes(tr.y, te.y, bins.y)
    _, y_material = np.unique(te.label, return_inverse=True)
    return (
        cmi_estimate(y_temperature, x, v, n_perm, rng),
        cmi_estimate(y_material.astype(np.intp), x, v, n_perm, rng),
    )


@dataclass(frozen=True)
class Fold:
    """kind is 'logo' (cross-scene) or 'within_group_blocked' (NOT cross-scene)."""

    kind: str
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


def folds_for_condition(
    logo: Sequence[tuple[tuple[str, ...], tuple[str, ...], str]],
    meta: Mapping[str, tuple[str, Condition]],
    condition: Condition,
) -> list[Fold]:
    """Restrict leave_one_group_out folds to one condition; train/test never share a group."""
    folds = []
    for train_ids, test_ids, group in logo:
        tr = tuple(s for s in train_ids if meta[s][1] == condition)
        te = tuple(s for s in test_ids if meta[s][1] == condition)
        if not tr or not te:
            continue
        train_groups = tuple(sorted({meta[s][0] for s in tr}))
        folds.append(Fold("logo", train_groups, (group,), tr, te))
    return folds


def blocked_folds(ordered_ids: Sequence[str], group: str, n_folds: int) -> list[Fold]:
    """Contiguous blocks of one group's (azimuth-ordered) samples: NOT a cross-scene split."""
    blocks = np.array_split(np.arange(len(ordered_ids)), n_folds)
    folds = []
    for b in blocks:
        test = tuple(ordered_ids[i] for i in b)
        train = tuple(s for s in ordered_ids if s not in test)
        folds.append(Fold("within_group_blocked", (group,), (group,), train, test))
    return folds
