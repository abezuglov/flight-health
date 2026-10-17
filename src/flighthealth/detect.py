"""Multivariate anomaly scores per phase, with the channels that drove each score."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from .features import channel_of

MAX_NAN_FRAC = 0.2
CLIP_Z = 10.0
PCA_VARIANCE = 0.9


@dataclass
class PhaseModel:
    columns: list[str]
    median: np.ndarray
    scale: np.ndarray
    pca: PCA
    forest: IsolationForest
    pca_ref: np.ndarray      # training scores, for percentile scores
    forest_ref: np.ndarray
    pca_max_ref: np.ndarray | None = None


def _standardize(X: pd.DataFrame, median, scale, clip: float | None = None) -> np.ndarray:
    z = (X.to_numpy(float) - median) / scale
    z = np.where(np.isnan(z), 0.0, z)  # missing = typical value
    return np.clip(z, -clip, clip) if clip else z


def fit(features: pd.DataFrame, seed: int = 0) -> PhaseModel:
    """Fit on one phase's feature rows. Robust scaling (median / IQR) so outliers in training
    don't set the scale; constant and mostly-missing features are dropped."""
    X = features.drop(columns=["flight_id", "phase"])
    keep = X.columns[(X.isna().mean() <= MAX_NAN_FRAC)]
    X = X[keep]
    med = X.median().to_numpy(float)
    iqr = (X.quantile(0.75) - X.quantile(0.25)).to_numpy(float)
    spread = X.std().to_numpy(float)
    scale = np.where(iqr > 0, iqr / 1.349, spread)
    ok = scale > 0
    X, med, scale = X.loc[:, ok], med[ok], scale[ok]
    # clip so near-constant features (tiny spread in training) can't explode into noise
    Z = _standardize(X, med, scale, CLIP_Z)
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=seed).fit(Z)
    forest = IsolationForest(n_estimators=300, random_state=seed).fit(Z)
    m = PhaseModel(list(X.columns), med, scale, pca, forest, np.array([]), np.array([]))
    err = _pca_error(m, Z)
    m.pca_ref = err.sum(axis=1)
    m.pca_max_ref = _by_channel(m, err).max(axis=1).to_numpy()
    m.forest_ref = -forest.score_samples(Z)
    return m


def _pca_error(m: PhaseModel, Z: np.ndarray) -> np.ndarray:
    return (Z - m.pca.inverse_transform(m.pca.transform(Z))) ** 2


def _by_channel(m: PhaseModel, err: np.ndarray) -> pd.DataFrame:
    """Reconstruction error summed per sensor channel (all of its features together)."""
    return pd.DataFrame(err, columns=m.columns).T.groupby(channel_of).sum().T


def score(m: PhaseModel, features: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Scores per flight-phase:
    pca_score      total reconstruction error (whole-aircraft unusualness)
    pca_max_score  error of the single worst channel (catches one sensor or engine going off)
    forest_score   isolation forest
    *_pct          percentile among training flights; top_channels = largest shares of error."""
    Z = _standardize(features.reindex(columns=m.columns), m.median, m.scale, CLIP_Z)
    err = _pca_error(m, Z)
    pca_s = err.sum(axis=1)
    forest_s = -m.forest.score_samples(Z)
    by_channel = _by_channel(m, err)
    share = by_channel.div(by_channel.sum(axis=1), axis=0)
    top = share.apply(lambda r: r.nlargest(top_k), axis=1)
    out = features[["flight_id", "phase"]].copy()
    out["pca_score"] = pca_s
    out["pca_max_score"] = by_channel.max(axis=1).to_numpy()
    out["forest_score"] = forest_s
    out["pca_pct"] = [100 * np.mean(m.pca_ref < s) for s in pca_s]
    out["pca_max_pct"] = [100 * np.mean(m.pca_max_ref < s) for s in out.pca_max_score]
    out["forest_pct"] = [100 * np.mean(m.forest_ref < s) for s in forest_s]
    out["combined_pct"] = (out.pca_pct + out.forest_pct) / 2
    out["top_channels"] = [
        "; ".join(f"{c} {v:.0%}" for c, v in r.dropna().sort_values(ascending=False).items()) for _, r in top.iterrows()
    ]
    out["top_channel"] = share.idxmax(axis=1).to_numpy()
    return out


def engine_columns(columns) -> list[str]:
    """Engine-vs-siblings features: each engine's mean deviation from the median of the four."""
    return [c for c in columns if "_dev_" in c and c.endswith("__mean")]


def sibling_z(m: PhaseModel, features: pd.DataFrame) -> pd.DataFrame:
    """No-model baseline: the largest robust z-score among engine deviation features, and its channel."""
    # unclipped: the size of the largest deviation is the whole signal here
    Z = np.abs(_standardize(features.reindex(columns=m.columns), m.median, m.scale))
    z = pd.DataFrame(Z, columns=[channel_of(c) for c in m.columns], index=features.index)
    return pd.DataFrame({"sibling_z": z.max(axis=1), "sibling_channel": z.idxmax(axis=1)})
