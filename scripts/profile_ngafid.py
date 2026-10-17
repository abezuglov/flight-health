"""Data-quality profile of the NGAFID 2days set: sentinel values, impossible ranges, labels."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import safe_pickle

root = sys.argv[1]
cols = pd.read_csv(f"{root}/stats.csv", index_col=0).columns[:23]
hdr = pd.read_csv(f"{root}/flight_header.csv", index_col=0)
data = safe_pickle.load(f"{root}/flight_data.pkl")
print("flights in pickle:", len(data), " in header:", len(hdr), " matched:", len(set(data) & set(hdr.index)))
print("before/after counts:", hdr.before_after.value_counts().to_dict(), " label types:", hdr.label.nunique())
print("top labels:", hdr.label.value_counts().head(6).to_dict())

# plausible physical ranges for a Cessna 172-class piston single
rng = {"volt1": (0, 35), "volt2": (0, 35), "FQtyL": (0, 30), "FQtyR": (0, 30), "E1 FFlow": (0, 25),
       "E1 OilT": (-40, 300), "E1 OilP": (0, 120), "E1 RPM": (0, 3000), "OAT": (-60, 60), "IAS": (0, 200),
       "VSpd": (-6000, 6000), "NormAc": (-4, 6), "AltMSL": (-1500, 20000),
       **{f"E1 CHT{i}": (-40, 600) for i in range(1, 5)}, **{f"E1 EGT{i}": (-40, 1800) for i in range(1, 5)}}
rows = []
allx = np.concatenate([v.astype(np.float32) for v in data.values()])
for j, c in enumerate(cols):
    x = allx[:, j]
    lo, hi = rng.get(c, (-np.inf, np.inf))
    finite = x[np.isfinite(x)]
    rows.append(dict(param=c, nan_pct=100 * np.isnan(x).mean(), inf_pct=100 * np.isinf(x).mean(),
                     out_of_range_pct=100 * ((finite < lo) | (finite > hi)).sum() / x.size,
                     zero_pct=100 * (x == 0).mean(), p01=np.percentile(finite, 0.1), p999=np.percentile(finite, 99.9),
                     min=finite.min(), max=finite.max()))
df = pd.DataFrame(rows).set_index("param")
df.to_csv(sys.argv[2])
pd.set_option("display.width", 200)
print(df.round(2).to_string())
lens = np.array([len(v) for v in data.values()])
print(f"rows/flight min/median/max: {lens.min()}/{int(np.median(lens))}/{lens.max()}  total rows: {lens.sum():,}")
