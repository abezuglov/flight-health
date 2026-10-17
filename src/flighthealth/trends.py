"""Engine trend monitoring across flights: step changes in each engine's deviation from its
siblings, and the ground runs that come just before them."""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 15          # flights on each side of a candidate step
MIN_STEP_Z = 6.0     # step size in units of the series' robust flight-to-flight noise
MIN_GAP = 30         # flights between two reported steps on the same channel
HIGH_POWER_N1 = 90   # a ground run above this N1 is a run-up, not just a start or taxi


def _robust_noise(x: np.ndarray) -> float:
    d = np.diff(x[np.isfinite(x)])
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) if d.size else np.nan


def find_steps(series: pd.Series) -> pd.DataFrame:
    """Step changes in one chronological series (index = flight order).

    At each point, compare the median of the WINDOW flights before with the WINDOW flights after.
    Report local peaks of that difference that exceed MIN_STEP_Z times the series noise."""
    x = series.to_numpy(float)
    n, w = len(x), WINDOW
    noise = _robust_noise(x)
    if n < 2 * w or not np.isfinite(noise) or noise == 0:
        return pd.DataFrame(columns=["at", "before", "after", "step", "step_z"])
    with np.errstate(all="ignore"):
        before = np.array([np.nanmedian(x[i - w:i]) for i in range(w, n - w + 1)])
        after = np.array([np.nanmedian(x[i:i + w]) for i in range(w, n - w + 1)])
    z = np.abs(after - before) / noise
    rows, taken = [], np.zeros(len(z), bool)
    for j in np.argsort(-z):
        if z[j] < MIN_STEP_Z:
            break
        if taken[max(0, j - MIN_GAP):j + MIN_GAP].any():
            continue
        taken[j] = True
        i = _refine(x, j + w, w)
        with np.errstate(all="ignore"):
            b, a = np.nanmedian(x[i - w:i]), np.nanmedian(x[i:i + w])
        rows.append(dict(at=series.index[i], before=b, after=a, step=a - b, step_z=z[j]))
    return pd.DataFrame(rows)


def _refine(x: np.ndarray, i0: int, w: int) -> int:
    """Window medians give a flat peak several flights wide around a step. Pick the split near
    i0 that best divides the surrounding 2*w flights into two flat segments (least squares)."""
    lo, hi = max(i0 - w, 1), min(i0 + w, len(x) - 1)
    best, best_i = np.inf, i0
    for i in range(max(i0 - w // 2, lo), min(i0 + w // 2, hi) + 1):
        left, right = x[max(i - w, 0):i], x[i:i + w]
        left, right = left[np.isfinite(left)], right[np.isfinite(right)]
        if left.size < 3 or right.size < 3:
            continue
        sse = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
        if sse < best:
            best, best_i = sse, i
    return best_i


def ground_runs_before(flights: pd.DataFrame, run_summary: pd.DataFrame, flight_id: str,
                       flights_back: int = 1) -> pd.DataFrame:
    """Ground-only recordings in the gaps before `flight_id`, reaching back `flights_back` real flights."""
    f = flights.sort_values("start_utc").reset_index(drop=True)
    i = f.index[f.flight_id == flight_id][0]
    prev_air = f.index[(f.index < i) & (f.n_legs > 0)]
    lo = prev_air[-flights_back] + 1 if len(prev_air) >= flights_back else 0
    ids = f.loc[lo:i - 1, "flight_id"]
    return run_summary[run_summary.flight_id.isin(ids)]
