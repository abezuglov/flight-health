"""Build the small data bundle the Streamlit app runs on (app/data/), from the full processed data.

The full processed data is ~1.3 GB and stays local. The bundle holds every flight-level table plus
the 1 Hz time series of a hand-picked set of flights that illustrate each finding.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flighthealth.pipeline import load_flight  # noqa: E402

DL = Path("data/processed/dashlink")
OUT = Path("app/data")
OUT.mkdir(parents=True, exist_ok=True)


def save(df: pd.DataFrame, name: str) -> None:
    """Write a bundle table; flight IDs are always strings (CSV round-trips turn them into integers)."""
    if "flight_id" in df:
        df = df.assign(flight_id=df.flight_id.astype(str))
    df.to_parquet(OUT / name, index=False, compression="zstd")

# flight id -> why it's in the demo
DEMO = {
    "670200104120549": "Typical flight",
    "670200105172034": "Crew throttles engine 3 back in cruise (top engine score)",
    "670200105050922": "Short flight: 'cruise' is not steady flight (top engine score)",
    "670200104191112": "Engine 1 oil temperature offset",
    "670200104151429": "Last flight before the April engine 4 oil-pressure step",
    "670200104151758": "Ground run: engines 1-3 only, engine 4 off (April)",
    "670200104160401": "Ground run: engine 4 to 96% N1 (April)",
    "670200104160544": "First flight after the April step: engine 4 oil pressure +15 psi",
    "670200107101932": "Last flight before the July step (hottest takeoff EGT in the fleet)",
    "670200107110147": "Ground run: engine 4 alone to 96% N1 (July)",
    "670200107110502": "First flight after the July step: engine 4 oil pressure back to normal",
    "670200105311116": "Speed brakes / spoilers used in climb (top whole-aircraft score)",
    "670200105010602": "Airbrake used in cruise (top whole-aircraft score)",
    "670200105071705": "Hardest landing in the fleet (1.73 g)",
    "670200106201007": "Largest bank and localizer deviation below 1,000 ft",
    "670200104281632": "Corrupt date fields: start time taken from the file name",
}
COLS = (["flight_id", "t_s", "phase", "ph_acms", "alt", "ralt", "cas", "gs", "ivv", "ptch", "roll", "vrtg",
         "vrtg_min", "vrtg_max", "wow", "flap", "abrk", "spl_1", "latp", "lonp"]
        + [f"{fam}_{k}" for fam in ("pla", "n1", "n2", "egt", "ff", "oit", "oip", "vib") for k in range(1, 5)])

series = []
for fid in DEMO:
    df = load_flight(DL, fid)
    series.append(df[[c for c in COLS if c in df]])
save(pd.concat(series, ignore_index=True), "demo_flights.parquet")
save(pd.DataFrame({"flight_id": list(DEMO), "why": list(DEMO.values())}), "demo_index.parquet")

flights = pd.read_parquet(DL / "catalog" / "flights.parquet")
save(flights, "flights.parquet")
save(pd.read_parquet(DL / "catalog" / "parameters.parquet"), "parameters.parquet")
flags = pd.read_parquet(DL / "quality_flags.parquet")
save(flags[flags.flight_id.isin(DEMO)], "demo_flags.parquet")
flag_counts = flags.groupby(["flight_id", "check"]).size().unstack(fill_value=0).reset_index()
save(flag_counts, "flag_counts.parquet")

feats = pd.read_parquet(DL / "features" / "phase_features.parquet")
dev_cols = [c for c in feats.columns if "_dev_" in c and c.endswith("__mean")]
cruise = feats[feats.phase == "cruise"][["flight_id"] + dev_cols].merge(flights[["flight_id", "start_utc"]])
save(cruise, "engine_cruise.parquet")

P2 = Path("reports/phase2")
save(pd.read_csv(P2 / "anomaly_scores.csv"), "anomaly_scores.parquet")
for name in ("engine_scores_cruise", "engine_steps", "ground_runs", "exceedances", "injection_detection"):
    save(pd.read_csv(P2 / f"{name}.csv"), f"{name}.parquet")

# NGAFID: summary tables only (the raw data has its own licence and is 1.1 GB)
P3 = Path("reports/phase3")
for name in ("supervised_cv", "auc_vs_oil_step", "channel_importance", "unsupervised_auc", "early_detection"):
    save(pd.read_csv(P3 / f"{name}.csv"), f"ngafid_{name}.parquet")
ng = pd.read_parquet("data/processed/ngafid/flight_features.parquet",
                     columns=["label", "date_diff", "before_after", "cruise|oip__mean"])
fam = ng.label.str.extract(r"^(baffle|intake|rocker|cylinder|engine|oil)", expand=False).fillna("other")
fam = fam.replace({"rocker": "rocker cover", "engine": "engine / oil", "oil": "engine / oil"})
oil = (ng.assign(family=fam).groupby(["family", "date_diff"])["cruise|oip__mean"]
       .agg(["mean", "std", "count"]).reset_index())
save(oil, "ngafid_oil_step.parquet")

size = sum(p.stat().st_size for p in OUT.iterdir()) / 1e6
print(f"bundle: {len(list(OUT.iterdir()))} files, {size:.1f} MB")
