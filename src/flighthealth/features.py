"""Per flight-phase features for anomaly detection.

One row per (flight, phase). For each sensor: mean, std, 5th and 95th percentile over the phase.
For each four-engine family: how far each engine sits from the median of the four, which is the
most direct engine-health signal a four-engine aircraft offers. Warnings: seconds active.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_PHASES = ["takeoff", "climb", "cruise", "descent", "approach", "landing"]
MIN_PHASE_S = 20

# Channels that describe where the aircraft is or what the crew selected, not how it is behaving.
EXCLUDE = {"th", "mh", "trk", "trkm", "da", "wd", "latp", "lonp", "crss", "dwpt", "casm",
           "alts", "cass", "hdgs", "mns", "vsps", "fqty_1", "fqty_2", "fqty_4"}
ENGINE_FAMILIES = ["n1", "n2", "egt", "ff", "vib", "oit", "oip", "pla"]
WARNINGS = ["mw", "gpws", "shkr", "push", "wshr", "tocw"]
STATS = ("mean", "std", "p05", "p95")


def feature_channels(catalog: pd.DataFrame) -> list[str]:
    c = catalog[(catalog.kind == "continuous") & catalog.use_for_modeling]
    return [p for p in c.param if p not in EXCLUDE]


def engine_deviation(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """Each engine minus the median of all four, per second."""
    cols = [f"{family}_{k}" for k in range(1, 5)]
    x = df[cols].to_numpy(float)
    with np.errstate(all="ignore"):
        med = np.nanmedian(x, axis=1, keepdims=True)
    return pd.DataFrame(x - med, columns=[f"{family}_dev_{k}" for k in range(1, 5)], index=df.index)


def _stats(block: pd.DataFrame) -> dict:
    x = block.to_numpy(float)
    with np.errstate(all="ignore"), _quiet():
        out = {
            "mean": np.nanmean(x, axis=0), "std": np.nanstd(x, axis=0),
            "p05": np.nanpercentile(x, 5, axis=0), "p95": np.nanpercentile(x, 95, axis=0),
        }
    return {f"{c}__{s}": out[s][i] for s in STATS for i, c in enumerate(block.columns)}


def flight_features(df: pd.DataFrame, channels: list[str]) -> list[dict]:
    """Feature rows for every modeled phase of one flight (aligned 1 Hz table)."""
    dev = pd.concat([engine_deviation(df, f) for f in ENGINE_FAMILIES], axis=1)
    rows = []
    for phase in MODEL_PHASES:
        m = (df.phase == phase).to_numpy()
        if m.sum() < MIN_PHASE_S:
            continue
        row = {"flight_id": df.flight_id.iat[0], "phase": phase, "duration_s": int(m.sum())}
        row |= _stats(df.loc[m, channels])
        d = dev.loc[m]
        with np.errstate(all="ignore"), _quiet():
            row |= {f"{c}__mean": v for c, v in zip(d.columns, np.nanmean(d.to_numpy(float), axis=0))}
            row |= {f"{c}__absmax": v for c, v in zip(d.columns, np.nanmax(np.abs(d.to_numpy(float)), axis=0))}
        for w in WARNINGS:
            if w in df:
                row[f"{w}__on_s"] = int((df.loc[m, w].fillna(0) > 0).sum())
        rows.append(row)
    return rows


def channel_of(feature: str) -> str:
    """'egt_dev_3__mean' -> 'egt_3' ; 'vrtg__p95' -> 'vrtg'."""
    base = feature.split("__")[0]
    if "_dev_" in base:
        fam, k = base.split("_dev_")
        return f"{fam}_{k}"
    return base


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)
