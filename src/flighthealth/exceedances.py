"""Baseline: limit rules in the style of airline flight-data monitoring.

Real programs use the aircraft type's published limits. Those aren't available for this
aircraft, so each rule flags flights beyond the fleet's own 99.5th percentile (or 0.5th for
"too low" rules). The rule list is what an analyst would check first; the thresholds are
deliberately simple and stated in the output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAIL_PCT = 99.5

# name, phase, how to compute the per-flight value, direction ("high" or "low"), description
RULES = [
    ("hard_landing", "landing", lambda d: d["vrtg_max"].max(), "high", "peak vertical load at touchdown (g)"),
    ("high_sink_below_1000ft", "approach", lambda d: d.loc[d.ralt < 1000, "ivv_min"].min(), "low",
     "steepest descent rate below 1,000 ft radio altitude (ft/min)"),
    ("glideslope_dev_below_1000ft", "approach", lambda d: d.loc[d.ralt < 1000, "gls"].abs().max(), "high",
     "largest glideslope deviation below 1,000 ft (DDM)"),
    ("localizer_dev_below_1000ft", "approach", lambda d: d.loc[d.ralt < 1000, "loc"].abs().max(), "high",
     "largest localizer deviation below 1,000 ft (DDM)"),
    ("bank_below_500ft", "approach", lambda d: d.loc[d.ralt < 500, ["roll_min", "roll_max"]].abs().max().max(), "high",
     "largest bank angle below 500 ft (deg)"),
    ("takeoff_pitch", "takeoff", lambda d: d["ptch_max"].max(), "high", "highest pitch during takeoff (deg)"),
    ("landing_pitch", "landing", lambda d: d["ptch_max"].max(), "high", "highest pitch at touchdown (deg)"),
    ("max_egt", "takeoff", lambda d: d[[f"egt_{k}" for k in range(1, 5)]].max().max(), "high",
     "hottest EGT on any engine during takeoff"),
    ("n1_split_cruise", "cruise", lambda d: _split(d, "n1"), "high", "largest N1 gap between engines in cruise (%)"),
    ("egt_split_cruise", "cruise", lambda d: _split(d, "egt"), "high", "largest EGT gap between engines in cruise"),
    ("vibration_cruise", "cruise", lambda d: d[[f"vib_{k}" for k in range(1, 5)]].max().max(), "high",
     "highest engine vibration in cruise (in/s)"),
    ("low_oil_pressure_cruise", "cruise", lambda d: d[[f"oip_{k}" for k in range(1, 5)]].min().min(), "low",
     "lowest oil pressure on any engine in cruise (psi)"),
    ("high_oil_temp_cruise", "cruise", lambda d: d[[f"oit_{k}" for k in range(1, 5)]].max().max(), "high",
     "highest oil temperature on any engine in cruise"),
]


def _split(d: pd.DataFrame, fam: str) -> float:
    x = d[[f"{fam}_{k}" for k in range(1, 5)]].to_numpy(float)
    with np.errstate(all="ignore"):
        gap = np.nanmax(x, axis=1) - np.nanmin(x, axis=1)
    # sustained: the gap held for at least 30 s, not a single-second blip
    return float(pd.Series(gap).rolling(30, min_periods=30).min().max()) if gap.size >= 30 else np.nan


def rule_values(df: pd.DataFrame) -> list[dict]:
    """The value of every rule metric for one flight (aligned 1 Hz table)."""
    out = []
    for name, phase, fn, direction, desc in RULES:
        d = df[df.phase == phase]
        if d.empty:
            continue
        try:
            v = float(fn(d))
        except (KeyError, ValueError):
            v = np.nan
        out.append(dict(flight_id=df.flight_id.iat[0], rule=name, phase=phase, value=v,
                        direction=direction, description=desc))
    return out


def flag_exceedances(values: pd.DataFrame) -> pd.DataFrame:
    """Mark values beyond the fleet's TAIL_PCT percentile for each rule."""
    high = values.direction == "high"
    hi_q = values.groupby("rule").value.transform(lambda v: np.nanpercentile(v, TAIL_PCT))
    lo_q = values.groupby("rule").value.transform(lambda v: np.nanpercentile(v, 100 - TAIL_PCT))
    out = values.assign(threshold=np.where(high, hi_q, lo_q))
    return out.assign(exceeded=np.where(high, out.value > out.threshold, out.value < out.threshold))
