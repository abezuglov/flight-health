"""Split a flight into phases from 1 Hz weight-on-wheels, ground speed and altitude.

Rules (heights are pressure altitude relative to the departure / arrival field):
  takeoff  = start of the ground roll (last GS < 40 kt) until 1,000 ft above the field
  climb    = until first within 1,000 ft of the flight's highest altitude
  cruise   = until last within 1,000 ft of the highest altitude
  descent  = until the last time above 3,000 ft over the arrival field
  approach = until the last time above 50 ft
  landing  = through touchdown until GS < 40 kt
  taxi / parked on the ground before and after: taxi = engines running (N2 > 40%).
  Ground speed can't be used for taxi: this recorder reports 0 below about 25 kt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .quality import runs

ROLL_GS = 40
TAKEOFF_TOP_FT = 1000
CRUISE_BAND_FT = 1000
APPROACH_TOP_FT = 3000
FLARE_FT = 50
MIN_AIRBORNE_S = 60

PHASES = ["parked", "taxi_out", "takeoff", "climb", "cruise", "descent", "approach", "landing", "taxi_in"]


def label_seconds(airborne: np.ndarray, gs: np.ndarray, alt: np.ndarray,
                  engines_on: np.ndarray) -> tuple[np.ndarray, dict]:
    """Phase name for every second, plus flight-level facts (legs, takeoff/landing seen)."""
    n = airborne.size
    lab = np.full(n, "", dtype=object)
    legs = [(s, e) for s, e in runs(airborne.astype(bool)) if e - s + 1 >= MIN_AIRBORNE_S]
    info = dict(n_legs=len(legs), has_takeoff=False, has_landing=False)
    ground = np.where(engines_on, "taxi_out", "parked")
    if not legs:
        lab[:] = ground
        return lab, info

    lo, td = legs[0][0], legs[-1][1]
    info["has_takeoff"] = lo > 0
    info["has_landing"] = td < n - 1

    def med(a, b):
        seg = alt[max(a, 0):max(min(b, n), 1)]
        return float(np.nanmedian(seg)) if np.isfinite(seg).any() else np.nan

    elev_to = med(lo - 10, lo) if info["has_takeoff"] else np.nan
    elev_ld = med(td + 1, td + 11) if info["has_landing"] else np.nan

    # takeoff: ground roll start .. 1,000 ft above departure field
    roll = lo
    if info["has_takeoff"]:
        slow = np.where(gs[:lo] < ROLL_GS)[0]
        roll = slow[-1] + 1 if slow.size else 0
    to_end = lo
    if np.isfinite(elev_to):
        above = np.where(alt[lo:td + 1] - elev_to >= TAKEOFF_TOP_FT)[0]
        to_end = lo + above[0] if above.size else td

    # landing: last time above 50 ft .. GS below 40 kt after touchdown
    flare, stop = td, td
    if np.isfinite(elev_ld):
        hi50 = np.where(alt[:td + 1] - elev_ld > FLARE_FT)[0]
        flare = hi50[-1] + 1 if hi50.size else td
        hi3k = np.where(alt[:flare] - elev_ld > APPROACH_TOP_FT)[0]
        app = hi3k[-1] + 1 if hi3k.size else to_end
        fast = np.where(gs[td + 1:] < ROLL_GS)[0]
        stop = td + 1 + fast[0] if fast.size else n - 1
    else:
        app = td + 1

    app = max(app, to_end)
    top = np.nanmax(alt[lo:td + 1])
    near_top = np.where(alt[to_end:app] >= top - CRUISE_BAND_FT)[0]
    toc = to_end + near_top[0] if near_top.size else app
    tod = to_end + near_top[-1] if near_top.size else app

    lab[:roll] = ground[:roll]
    lab[roll:to_end] = "takeoff"
    lab[to_end:toc] = "climb"
    lab[toc:tod + 1] = "cruise"
    lab[tod + 1:app] = "descent"
    if info["has_landing"]:
        lab[app:flare] = "approach"
        lab[flare:stop + 1] = "landing"
        lab[stop + 1:] = np.where(engines_on[stop + 1:], "taxi_in", "parked")
    else:
        lab[app:] = "descent"
    return lab, info


def segments(fid: str, labels: np.ndarray) -> pd.DataFrame:
    """Run-length encode per-second labels into phase segments."""
    change = np.concatenate([[True], labels[1:] != labels[:-1]])
    starts = np.where(change)[0]
    ends = np.concatenate([starts[1:] - 1, [labels.size - 1]])
    return pd.DataFrame(dict(flight_id=fid, phase=labels[starts], t_start_s=starts, t_end_s=ends, method="derived"))
