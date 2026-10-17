"""Data-quality checks. Each check returns flag rows; values are never changed here.

Channel checks run at the channel's native rate so a single bad sample is caught
before 1 Hz averaging can hide it. Cross-checks run on the aligned 1 Hz table.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

ANGLE_CHANNELS = {"TH", "MH", "TRK", "TRKM", "WD", "HDGS", "CRSS", "DA"}
DROPOUT_CHANNELS_RE = r"^(N1_\d|N2_\d|EGT_\d|FF_\d|OIP_\d)$"
STUCK_MIN_S = 300
DROPOUT_MAX_S = 20
SPIKE_K = 10.0
CROSS_MIN_S = 10


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of consecutive True values."""
    if not mask.any():
        return []
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def equal_runs(x: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index pairs of runs of identical consecutive values (length >= 2)."""
    same = np.concatenate([[False], x[1:] == x[:-1]])
    return [(s - 1, e) for s, e in runs(same)]


def _flag(fid, param, rate, s, e, check, severity, detail):
    return dict(flight_id=fid, param=param, t_start_s=s / rate, t_end_s=e / rate,
                check=check, severity=severity, detail=detail)


def channel_checks(fid: str, mnemonic: str, x: np.ndarray, rate: float, rule: dict,
                   airborne_at: callable) -> list[dict]:
    """Sentinel, range, dropout, spike and stuck checks for one channel at native rate."""
    x = x.astype(float)
    param = mnemonic.lower()
    flags, used = [], np.zeros(x.size, bool)

    for tok in filter(None, str(rule.get("sentinels", "")).split(";")):
        hit = np.isclose(x, float(tok), atol=1e-3) & ~used
        for s, e in runs(hit):
            flags.append(_flag(fid, param, rate, s, e, "sentinel", "bad", f"no-data code {tok}"))
        used |= hit

    lo, hi = rule.get("valid_min", np.nan), rule.get("valid_max", np.nan)
    if np.isfinite(lo) or np.isfinite(hi):
        tol = 1e-6 * max(1.0, abs(lo) if np.isfinite(lo) else 1.0, abs(hi) if np.isfinite(hi) else 1.0)
        bad = ((x < lo - tol) | (x > hi + tol)) & ~used
        for s, e in runs(bad):
            v = x[s:e + 1]
            flags.append(_flag(fid, param, rate, s, e, "out_of_range", "bad",
                               f"{np.nanmin(v):.4g}..{np.nanmax(v):.4g} outside [{lo:g}, {hi:g}]"))
        used |= bad

    if re.match(DROPOUT_CHANNELS_RE, mnemonic):
        good = x[~used]
        ref = np.percentile(good, 99) if good.size else 0
        zero = (x == 0) & ~used
        for s, e in runs(zero):
            if s == 0 or e == x.size - 1 or (e - s + 1) / rate > DROPOUT_MAX_S:
                continue
            if x[s - 1] > 0.2 * ref and x[e + 1] > 0.2 * ref and airborne_at(s / rate):
                flags.append(_flag(fid, param, rate, s, e, "dropout", "bad",
                                   f"reads 0 for {(e - s + 1) / rate:g}s between {x[s - 1]:.4g} and {x[e + 1]:.4g}"))
                used[s:e + 1] = True

    if mnemonic not in ANGLE_CHANNELS and x.size > 3:
        y = np.where(used, np.nan, x)
        d = np.diff(y)
        ad = np.abs(d[np.isfinite(d)])
        # p95 so the spikes themselves can't inflate the threshold; quantized channels fall back
        # to their smallest real step
        step = np.percentile(ad, 95) if ad.size else 0
        if step == 0 and (ad > 0).any():
            step = ad[ad > 0].min()
        if step > 0:
            j = SPIKE_K * step
            a, b = d[:-1], d[1:]  # jump into sample i, jump out of sample i
            back = np.abs(y[2:] - y[:-2])
            spike = (np.abs(a) > j) & (np.abs(b) > j) & (np.sign(a) != np.sign(b)) & (back < j / 2)
            for i in np.where(spike)[0] + 1:
                flags.append(_flag(fid, param, rate, i, i, "spike", "bad",
                                   f"{y[i]:.4g} between {y[i - 1]:.4g} and {y[i + 1]:.4g}"))

    if rule.get("stuck_check"):
        for s, e in equal_runs(x):
            dur = (e - s + 1) / rate
            if dur >= STUCK_MIN_S and airborne_at((s + e) / 2 / rate) and not used[s]:
                flags.append(_flag(fid, param, rate, s, e, "stuck", "warn",
                                   f"constant {x[s]:.4g} for {dur:.0f}s in flight"))
    return flags


def frame_checks(fid: str, frmc: np.ndarray, rate: float) -> list[dict]:
    """The frame counter should advance by its usual step (or wrap). Anything else means lost frames."""
    f = frmc.astype(np.int64)
    d = np.diff(f)
    if d.size == 0:
        return []
    vals, counts = np.unique(d, return_counts=True)
    step = vals[counts.argmax()]
    # around the wrap the counter steps X -> 0 -> 3 (not 4), so any step out of a value below `step` is normal
    wrap = (d < 0) | (f[:-1] < step)
    gaps = np.where((d != step) & ~wrap)[0]
    return [_flag(fid, "*", rate, i, i + 1, "frame_gap", "warn", f"counter {frmc[i]} -> {frmc[i + 1]}")
            for i in gaps]


def cross_checks(fid: str, df: pd.DataFrame, airborne: np.ndarray) -> list[dict]:
    """Disagreement between channels that measure the same thing (1 Hz, sustained)."""
    pairs = [
        ("bal1", "bal2", 200, False, "baro altitude 1 vs 2 differ by more than 200 ft"),
        ("alt", "bal1", 2500, False, "pressure vs baro-corrected altitude differ by more than 2,500 ft"),
        ("aoa1", "aoa2", 5, True, "angle-of-attack vanes differ by more than 5 deg"),
        ("altr", "ivv", 1500, True, "altitude rate vs inertial vertical speed differ by more than 1,500 ft/min"),
    ]
    flags = []
    for a, b, tol, air_only, text in pairs:
        if a not in df or b not in df:
            continue
        diff = (df[a] - df[b]).abs().to_numpy()
        m = diff > tol
        if air_only:
            m &= airborne
        for s, e in runs(m):
            if e - s + 1 >= CROSS_MIN_S:
                flags.append(dict(flight_id=fid, param=f"{a}|{b}", t_start_s=float(s), t_end_s=float(e),
                                  check="cross_check", severity="warn",
                                  detail=f"{text} (max {np.nanmax(diff[s:e + 1]):.0f})"))
    return flags
