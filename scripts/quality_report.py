"""Summarize the processed DASHlink tables into reports/quality_summary.json and a few CSVs."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

cat = pd.read_parquet(root / "catalog" / "parameters.parquet")
flights = pd.read_parquet(root / "catalog" / "flights.parquet")
flags = pd.read_parquet(root / "quality_flags.parquet")
total_s = flights.duration_s.sum()
rate = cat.set_index("param").rate_hz

# Share of native samples dropped as bad, per channel
bad = flags[(flags.severity == "bad") & (flags.param != "*")].copy()
bad["n_samples"] = np.round((bad.t_end_s - bad.t_start_s) * bad.param.map(rate)).astype(int) + 1
dropped = bad.groupby("param").agg(samples_dropped=("n_samples", "sum"), flights_hit=("flight_id", "nunique"))
dropped["pct_samples_dropped"] = 100 * dropped.samples_dropped / (total_s * rate.reindex(dropped.index))

# Share of 1 Hz seconds with no valid value left, plus phase labels for the comparison below
missing, ph_rows = None, []
for fid in flights.flight_id:
    df = pd.read_parquet(root / "aligned_1hz" / f"tail={fid[:3]}" / f"{fid}.parquet")
    m = df.drop(columns=["flight_id", "t_s", "ts_utc", "phase", "ph_acms"]).isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)
    if not flights.set_index("flight_id").at[fid, "is_fragment"]:
        ph_rows.append(df[["phase", "ph_acms"]])
by_param = dropped.join((100 * missing / total_s).rename("pct_seconds_missing"), how="outer").fillna(0)
by_param = by_param.join(cat.set_index("param")[["group", "description"]], how="left")
by_param = by_param[by_param.index.isin(cat.param)]
by_param.sort_values("pct_samples_dropped", ascending=False).to_csv(out / "dropped_by_param.csv")

checks = (flags.merge(cat[["param", "group"]], on="param", how="left").fillna({"group": "all channels"})
          .groupby(["check", "severity", "group"]).size().rename("n").reset_index())
checks.to_csv(out / "flags_by_check_group.csv", index=False)

# Derived phases vs the aircraft's own phase code (full flights only). The ACMS codes don't split
# approach from descent or taxi-in from taxi-out, so also compare on the phases both can express.
ph = pd.concat(ph_rows, ignore_index=True)
ct = pd.crosstab(ph.phase, ph.ph_acms)
ct.to_csv(out / "phase_vs_acms.csv")
code_to_phase = ct.idxmax(axis=0)
exact = float((ph.ph_acms.map(code_to_phase) == ph.phase).mean())
coarse = {"parked": "ground", "taxi_out": "ground", "taxi_in": "ground", "takeoff": "takeoff", "climb": "climb",
          "cruise": "cruise", "descent": "descent", "approach": "descent", "landing": "landing"}
d = ph.phase.astype(str).map(coarse)
a = ph.ph_acms.map(code_to_phase).astype(str).map(coarse)
coarse_agree = float((d == a).mean())
airborne_mix = pd.crosstab(d[d.isin(["climb", "cruise", "descent"])], a[d.isin(["climb", "cruise", "descent"])],
                           normalize="index").round(3)

summary = dict(
    flights=len(flights),
    flight_hours=round(total_s / 3600, 1),
    ground_only_recordings=int((flights.n_legs == 0).sum()),
    fragments=int(flights.is_fragment.sum()),
    multi_leg=int((flights.n_legs > 1).sum()),
    start_time_from_filename=int((flights.start_time_source == "filename").sum()),
    flags_total=len(flags),
    flags_by_check=flags.groupby("check").size().to_dict(),
    channels_checked=int((cat.kind == "continuous").sum()),
    channels_with_bad_samples=int((by_param.samples_dropped > 0).sum()),
    pct_all_samples_dropped=round(float(100 * dropped.samples_dropped.sum()
                                        / (total_s * rate[cat.set_index("param").kind == "continuous"].sum())), 3),
    top_dropped=by_param.sort_values("pct_samples_dropped", ascending=False).head(10)[
        ["group", "pct_samples_dropped", "pct_seconds_missing"]].round(3).reset_index().to_dict("records"),
    acms_code_to_phase={int(k): v for k, v in code_to_phase.items()},
    phase_agreement_exact=round(exact, 3),
    phase_agreement_coarse=round(coarse_agree, 3),
    airborne_confusion=airborne_mix.to_dict("index"),
    phase_hours={k: round(v / 3600, 1) for k, v in ph.phase.value_counts().to_dict().items()},
)
(out / "quality_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
