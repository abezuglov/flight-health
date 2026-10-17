"""Find step changes in each engine's cruise deviation from its siblings, and the ground runs before them."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flighthealth.pipeline import load_flight  # noqa: E402
from flighthealth.trends import HIGH_POWER_N1, find_steps, ground_runs_before  # noqa: E402

ROOT = Path("data/processed/dashlink")
OUT = Path("reports/phase2")
FAMS = ["egt", "n1", "n2", "ff", "vib", "oit", "oip"]

flights = pd.read_parquet(ROOT / "catalog" / "flights.parquet").sort_values("start_utc")
feats = pd.read_parquet(ROOT / "features" / "phase_features.parquet")
cruise = feats[feats.phase == "cruise"].merge(flights[["flight_id", "start_utc"]]).sort_values("start_utc")
cruise = cruise.set_index("flight_id")

# what happened in each ground-only recording
runs = []
for fid in flights.loc[flights.n_legs == 0, "flight_id"]:
    d = load_flight(ROOT, fid)
    r = {"flight_id": fid}
    for k in range(1, 5):
        r[f"n1_max_{k}"] = float(d[f"n1_{k}"].max())
        r[f"run_min_{k}"] = float((d[f"n2_{k}"] > 40).sum() / 60)
    runs.append(r)
runs = pd.DataFrame(runs)

steps = []
for fam in FAMS:
    for k in range(1, 5):
        s = find_steps(cruise[f"{fam}_dev_{k}__mean"])
        for r in s.itertuples():
            g = ground_runs_before(flights, runs, r.at)
            hp = [e for e in range(1, 5) if (g[f"n1_max_{e}"] >= HIGH_POWER_N1).any()] if len(g) else []
            steps.append(dict(channel=f"{fam}_{k}", first_flight_after=r.at,
                              date=cruise.at[r.at, "start_utc"], before=r.before, after=r.after,
                              step=r.step, step_z=r.step_z, ground_runs_before=len(g),
                              high_power_ground_run_engines=",".join(map(str, hp)),
                              same_engine_run_up=k in hp))
steps = pd.DataFrame(steps).sort_values("step_z", ascending=False)
steps.to_csv(OUT / "engine_steps.csv", index=False)
runs.to_csv(OUT / "ground_runs.csv", index=False)
pd.set_option("display.width", 250)
print(steps.round({'before': 2, 'after': 2, 'step': 2, 'step_z': 1}).to_string(index=False))
hp_clusters = runs[[f"n1_max_{k}" for k in range(1, 5)]].max(axis=1).ge(HIGH_POWER_N1).sum()
print(f"\nground-only recordings: {len(runs)}, with a run-up above {HIGH_POWER_N1}% N1: {hp_clusters}")
