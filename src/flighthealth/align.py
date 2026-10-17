"""Put channels recorded at different rates onto one 1 Hz time axis.

Row t covers the window (t-1, t] seconds, so every value in row t was already
recorded by time t: nothing leaks backward from the future.
"""
from __future__ import annotations

import numpy as np

HIGHRATE_HZ = 16


def n_seconds(lengths_and_rates: list[tuple[int, float]]) -> int:
    """Seconds covered by every channel (the shortest one wins)."""
    return int(min(n / r for n, r in lengths_and_rates))


def window_view(x: np.ndarray, rate: float, n_sec: int) -> np.ndarray:
    """(n_sec, rate) array of the samples in each (t-1, t] window, NaN-padded at the start.

    Sample i is taken at time i / rate. Window t holds samples t*rate-(rate-1) .. t*rate.
    """
    m = int(rate)
    if m != rate or m < 1:
        raise ValueError(f"window_view needs an integer rate >= 1, got {rate}")
    padded = np.concatenate([np.full(m - 1, np.nan), x.astype(float)])
    need = n_sec * m
    if padded.size < need:
        padded = np.concatenate([padded, np.full(need - padded.size, np.nan)])
    return padded[:need].reshape(n_sec, m)


def to_1hz(x: np.ndarray, rate: float, n_sec: int, how: str = "mean") -> dict[str, np.ndarray]:
    """Resample one channel to 1 Hz.

    how: "mean" (continuous), "mean_minmax" (continuous, keep extremes), "max" (discrete flags).
    Slower-than-1 Hz channels hold their last recorded value.
    """
    if rate < 1:
        step = int(round(1 / rate))
        idx = np.minimum(np.arange(n_sec) // step, x.size - 1)
        return {"": x.astype(float)[idx]}
    w = window_view(x, rate, n_sec)
    with np.errstate(all="ignore"), _quiet():
        if how == "max":
            return {"": np.nanmax(w, axis=1)}
        out = {"": np.nanmean(w, axis=1)}
        if how == "mean_minmax":
            out["_min"] = np.nanmin(w, axis=1)
            out["_max"] = np.nanmax(w, axis=1)
        return out


def seconds_touched(t_start: float, t_end: float, n_sec: int) -> slice:
    """1 Hz rows whose (t-1, t] window overlaps the interval [t_start, t_end]."""
    lo = max(int(np.ceil(t_start)), 0)
    hi = min(int(np.ceil(t_end)), n_sec - 1)
    return slice(lo, hi + 1)


def to_highrate(x: np.ndarray, rate: float, n_samples: int) -> np.ndarray:
    """Hold an 8 Hz channel onto the 16 Hz grid (each sample repeated, never shifted earlier)."""
    rep = int(HIGHRATE_HZ / rate)
    return np.repeat(x.astype(float), rep)[:n_samples]


class _quiet:
    """Silence numpy's all-NaN-slice warnings inside a block."""

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)
