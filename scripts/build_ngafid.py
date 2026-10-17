"""Clean every NGAFID 2days flight and build one feature row per flight."""
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import safe_pickle  # noqa: E402
from flighthealth.ngafid import clean_flight, flight_features, rules_table  # noqa: E402

raw = Path(sys.argv[1])
out = Path(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
out.mkdir(parents=True, exist_ok=True)

t0 = time.time()
data = safe_pickle.load(raw / "flight_data.pkl")
header = pd.read_csv(raw / "flight_header.csv").rename(columns={"Master Index": "flight_id"})
rules = rules_table(HERE.parent / "config" / "ngafid_rules.csv")
print(f"loaded {len(data)} flights in {time.time() - t0:.0f}s")

feats, flags = [], []
for i, fid in enumerate(list(data)[:limit]):
    df, f = clean_flight(fid, data[fid], rules)
    feats.append(flight_features(fid, df))
    flags += f
    if i % 2000 == 0:
        print(f"  {i} flights, {time.time() - t0:.0f}s", flush=True)

feats = pd.DataFrame(feats).merge(header, on="flight_id", how="left")
flags = pd.DataFrame(flags)
feats.to_parquet(out / "flight_features.parquet", index=False)
flags.to_parquet(out / "quality_flags.parquet", index=False)
print(f"{len(feats)} flights, {feats.shape[1]} columns, {len(flags)} flags in {time.time() - t0:.0f}s")
print(flags.groupby(["param", "check"]).size().unstack(fill_value=0).to_string())
