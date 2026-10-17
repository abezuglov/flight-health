"""NGAFID maintenance dataset (Cessna 172, 23 channels at 1 Hz): load, clean, segment, featurize.

The same principles as the DASHlink pipeline: checks per channel, bad samples set to missing
(never filled), every change logged. Cylinder channels get the engine-vs-siblings treatment:
each cylinder's CHT/EGT minus the median of the four.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .catalog import apply_rules, load_rules
from .features import _quiet, _stats, engine_deviation
from .quality import channel_checks

# Column order of flight_data.pkl arrays (matches stats.csv), renamed to pipeline style
RAW_COLUMNS = ["volt1", "volt2", "amp1", "amp2", "FQtyL", "FQtyR", "E1 FFlow", "E1 OilT", "E1 OilP", "E1 RPM",
               "E1 CHT1", "E1 CHT2", "E1 CHT3", "E1 CHT4", "E1 EGT1", "E1 EGT2", "E1 EGT3", "E1 EGT4",
               "OAT", "IAS", "VSpd", "NormAc", "AltMSL"]
COLUMNS = ["volt1", "volt2", "amp1", "amp2", "fqty_l", "fqty_r", "ff", "oit", "oip", "rpm",
           "cht_1", "cht_2", "cht_3", "cht_4", "egt_1", "egt_2", "egt_3", "egt_4",
           "oat", "ias", "vspd", "normac", "alt"]
CYLINDER_FAMILIES = ["cht", "egt"]

# Segment rules (1 Hz)
AIRBORNE_IAS = 55       # kt
RUNUP_RPM = (1600, 2000)  # magneto check is done around 1,700-1,800 rpm on the ground
GROUND_IAS = 30
CRUISE_MAX_VSPD = 300   # ft/min
CRUISE_MIN_RPM = 2000
SEGMENTS = ["runup", "climb", "cruise"]


def rules_table(path) -> dict:
    rules = load_rules(path)
    r = apply_rules(pd.Series([c.upper() for c in COLUMNS]), rules)
    r.index = COLUMNS
    return r.to_dict("index")


def clean_flight(fid, arr: np.ndarray, rules: dict) -> tuple[pd.DataFrame, list[dict]]:
    """float16 (T, 23) array -> cleaned float32 DataFrame and quality flags."""
    df = pd.DataFrame(arr.astype(np.float32), columns=COLUMNS)
    airborne = (df.ias > AIRBORNE_IAS).to_numpy()

    def airborne_at(t):
        return bool(airborne[min(int(t), len(airborne) - 1)])

    flags = []
    for c in COLUMNS:
        x = df[c].to_numpy(float)
        f = channel_checks(fid, c.upper(), x, 1.0, rules[c], airborne_at)
        for r in f:
            r["param"] = c
            if r["severity"] == "bad":
                x[int(r["t_start_s"]): int(r["t_end_s"]) + 1] = np.nan
        df[c] = x.astype(np.float32)
        flags += f
    return df, flags


def segment(df: pd.DataFrame) -> np.ndarray:
    """Label each second: runup, climb, cruise, or '' (other)."""
    ias, rpm, vs = df.ias.to_numpy(float), df.rpm.to_numpy(float), df.vspd.to_numpy(float)
    vs_s = pd.Series(vs).rolling(30, center=True, min_periods=10).mean().to_numpy()
    lab = np.full(len(df), "", dtype=object)
    lab[(ias < GROUND_IAS) & (rpm >= RUNUP_RPM[0]) & (rpm <= RUNUP_RPM[1])] = "runup"
    air = ias > AIRBORNE_IAS
    lab[air & (vs_s > CRUISE_MAX_VSPD)] = "climb"
    lab[air & (np.abs(vs_s) <= CRUISE_MAX_VSPD) & (rpm >= CRUISE_MIN_RPM)] = "cruise"
    return lab


def flight_features(fid, df: pd.DataFrame, min_s: int = 20) -> dict:
    """One row per flight: per-segment channel stats and cylinder-vs-siblings deviations."""
    lab = segment(df)
    dev = pd.concat([engine_deviation(df, f) for f in CYLINDER_FAMILIES], axis=1)
    row = {"flight_id": fid, "duration_s": len(df)}
    for seg in SEGMENTS:
        m = lab == seg
        row[f"{seg}_s"] = int(m.sum())
        if m.sum() < min_s:
            continue
        row |= {f"{seg}|{k}": v for k, v in _stats(df.loc[m, COLUMNS]).items()}
        d = dev.loc[m].to_numpy(float)
        with np.errstate(all="ignore"), _quiet():
            means, absmax, sd = np.nanmean(d, 0), np.nanmax(np.abs(d), 0), np.nanstd(d, 0)
        for i, c in enumerate(dev.columns):
            row[f"{seg}|{c}__mean"] = means[i]
            row[f"{seg}|{c}__absmax"] = absmax[i]
            row[f"{seg}|{c}__std"] = sd[i]
        for fam in CYLINDER_FAMILIES:
            x = df.loc[m, [f"{fam}_{k}" for k in range(1, 5)]].to_numpy(float)
            with np.errstate(all="ignore"), _quiet():
                spread = np.nanmax(x, 1) - np.nanmin(x, 1)
            row[f"{seg}|{fam}_spread__mean"] = float(np.nanmean(spread)) if np.isfinite(spread).any() else np.nan
    return row
