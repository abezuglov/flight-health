"""Turn one DASHlink recording into the processed tables, and read them back."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import PIPELINE_VERSION
from .align import HIGHRATE_HZ, n_seconds, to_1hz, to_highrate
from .dashlink_io import flight_id, parse_filename, read_mat, sha256
from .phases import label_seconds, segments
from .quality import channel_checks, cross_checks, frame_checks

MINMAX_PARAMS = {"LATG", "LONG"}  # 4 Hz accelerations; every >= 8 Hz continuous channel also keeps extremes
FRAGMENT_MIN_S = 20 * 60


def _mode(x: np.ndarray) -> float:
    v, c = np.unique(x, return_counts=True)
    return float(v[c.argmax()])


def _start_time(ch, path, rules_by_mn) -> tuple[dt.datetime, str]:
    """Start time from the DATE/GMT channels when they are valid, else from the file name."""
    try:
        y, mo, d = (int(_mode(ch[k].data)) for k in ("DATE_YEAR", "DATE_MONTH", "DATE_DAY"))
        h, mi, s = (int(ch[k].data[0]) for k in ("GMT_HOUR", "GMT_MINUTE", "GMT_SEC"))
        for k, v in (("DATE_YEAR", y), ("DATE_MONTH", mo), ("DATE_DAY", d)):
            r = rules_by_mn.get(k, {})
            if not (r.get("valid_min", -np.inf) <= v <= r.get("valid_max", np.inf)):
                raise ValueError(k)
        return dt.datetime(y, mo, d, h, mi, s), "channels"
    except (KeyError, ValueError):
        return parse_filename(path)[1], "filename"


def process_flight(path: str | Path, catalog: pd.DataFrame) -> dict:
    fid = flight_id(path)
    tail, _ = parse_filename(path)
    ch = read_mat(path)
    cat = catalog.set_index("mnemonic")
    rules = cat[["valid_min", "valid_max", "sentinels", "stuck_check"]].to_dict("index")
    n_sec = n_seconds([(c.data.size, c.rate) for c in ch.values()])
    start, start_src = _start_time(ch, path, rules)

    # WOW is inverted on this aircraft: 1 = in the air
    airborne = to_1hz(ch["WOW"].data, ch["WOW"].rate, n_sec, "max")[""] == 1

    def airborne_at(t: float) -> bool:
        return bool(airborne[min(int(t), n_sec - 1)])

    # Checks run at native rate; samples flagged "bad" are left out of the 1 Hz aggregates,
    # so one dropout doesn't drag a whole second's mean or cost the good samples around it.
    flags, clean = [], {}
    for mn, c in ch.items():
        if mn in cat.index and cat.at[mn, "kind"] == "continuous":
            x = c.data[: int(np.ceil(n_sec * c.rate))].astype(float)
            f = channel_checks(fid, mn, x, c.rate, rules[mn], airborne_at)
            for r in f:
                if r["severity"] == "bad":
                    x[int(round(r["t_start_s"] * c.rate)): int(round(r["t_end_s"] * c.rate)) + 1] = np.nan
            flags += f
            clean[mn] = x
    if "FRMC" in ch:
        flags += frame_checks(fid, ch["FRMC"].data, ch["FRMC"].rate)

    cols = {}
    for mn, c in ch.items():
        if mn not in cat.index or cat.at[mn, "kind"] not in ("continuous", "discrete"):
            continue
        kind = cat.at[mn, "kind"]
        how = "max" if kind == "discrete" else ("mean_minmax" if c.rate >= 8 or mn in MINMAX_PARAMS else "mean")
        for suffix, v in to_1hz(clean.get(mn, c.data), c.rate, n_sec, how).items():
            cols[mn.lower() + suffix] = v
    discrete = {mn.lower() for mn in cat.index[cat.kind == "discrete"]}
    data = pd.DataFrame({
        k: pd.array(np.round(v), dtype="Int32") if k in discrete else v.astype(np.float32)
        for k, v in cols.items()
    })

    n2 = data[[c for c in ("n2_1", "n2_2", "n2_3", "n2_4") if c in data]].to_numpy(float)
    engines_on = (np.nan_to_num(n2, nan=0).max(axis=1) > 40) if n2.shape[1] else np.zeros(n_sec, bool)
    labels, info = label_seconds(airborne, data["gs"].to_numpy(float), data["alt"].to_numpy(float), engines_on)
    keys = pd.DataFrame({
        "flight_id": fid,
        "t_s": np.arange(n_sec, dtype=np.int32),
        "ts_utc": pd.Timestamp(start) + pd.to_timedelta(np.arange(n_sec), unit="s"),
        "phase": pd.Categorical(labels),
        "ph_acms": data.pop("ph") if "ph" in data else pd.array([pd.NA] * n_sec, dtype="Int32"),
    })
    df = pd.concat([keys, data], axis=1)

    flags += cross_checks(fid, df, airborne)
    flags = pd.DataFrame(flags, columns=["flight_id", "param", "t_start_s", "t_end_s", "check", "severity", "detail"])

    hr_mn = [mn for mn, c in ch.items() if c.rate >= 8 and cat.at[mn, "kind"] == "continuous"]
    n_hr = n_sec * HIGHRATE_HZ
    hr = pd.DataFrame({mn.lower(): to_highrate(ch[mn].data, ch[mn].rate, n_hr).astype(np.float32) for mn in hr_mn})
    hr.insert(0, "t_s", (np.arange(len(hr)) / HIGHRATE_HZ).astype(np.float32))
    hr.insert(0, "flight_id", fid)

    nan = pd.Series(np.nan, index=df.index)
    lat = df.get("latp", nan).where(lambda v: v != 0)
    lon = df.get("lonp", nan).where(lambda v: v != 0)
    meta = {}
    for k in [k for k in ch if k.startswith(("EHRS_", "ECYC_", "ESN_"))]:
        meta[k.lower()] = _mode(ch[k].data)
    bad = flags[flags.severity == "bad"]
    flight = dict(
        flight_id=fid, tail=tail, source_file=Path(path).name, source_sha256=sha256(path),
        start_utc=start, start_time_source=start_src, duration_s=n_sec,
        start_lat=lat.first_valid_index() and float(lat.dropna().iloc[0]),
        start_lon=lon.first_valid_index() and float(lon.dropna().iloc[0]),
        end_lat=lat.last_valid_index() and float(lat.dropna().iloc[-1]),
        end_lon=lon.last_valid_index() and float(lon.dropna().iloc[-1]),
        max_alt_ft=float(np.nanmax(df["alt"])),
        **info,
        is_fragment=bool(n_sec < FRAGMENT_MIN_S or not (info["has_takeoff"] and info["has_landing"])),
        n_quality_flags=len(flags), n_bad_flags=len(bad),
        pipeline_version=PIPELINE_VERSION, **meta,
    )
    return dict(aligned=df, highrate=hr, flags=flags, phases=segments(fid, labels), flight=flight)


def load_flight(root: str | Path, fid: str) -> pd.DataFrame:
    """Aligned 1 Hz table for one flight. Bad samples were already left out of each second's values;
    quality_flags.parquet lists every one, and the raw .mat / highrate tables keep the originals."""
    return pd.read_parquet(Path(root) / "aligned_1hz" / f"tail={fid[:3]}" / f"{fid}.parquet")
