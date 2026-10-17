"""Per-parameter data-quality profile across all DASHlink flights in a folder."""
import sys, glob, collections
import numpy as np, pandas as pd, scipy.io as sio

folder, out = sys.argv[1], sys.argv[2]
stats = collections.defaultdict(lambda: dict(flights=0, nan=0, n=0, const_flights=0, rate=None, units="", desc=""))
durations, rate_mismatch = [], 0
for f in sorted(glob.glob(folder + "/*.mat")):
    m = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
    params = {k: v for k, v in m.items() if not k.startswith("__")}
    durations.append(params["ALT"].data.size / params["ALT"].Rate / 60)
    for k, s in params.items():
        x = np.atleast_1d(np.asarray(s.data, dtype=float))
        st = stats[k]
        st["flights"] += 1; st["n"] += x.size; st["nan"] += int(np.isnan(x).sum())
        st["rate"] = s.Rate; st["units"] = s.Units if isinstance(s.Units, str) else ""; st["desc"] = s.Description
        if x.size and np.nanmin(x) == np.nanmax(x): st["const_flights"] += 1
        expected = durations[-1] * 60 * s.Rate
        if abs(x.size - expected) > max(2, 0.01 * expected): rate_mismatch += 1
df = pd.DataFrame(stats).T
df["nan_pct"] = 100 * df["nan"] / df["n"]
df["const_pct"] = 100 * df["const_flights"] / df["flights"]
df.drop(columns=["nan", "n"]).to_csv(out)
d = np.array(durations)
print(f"flights: {len(d)}  duration min/median/max (min): {d.min():.0f}/{np.median(d):.0f}/{d.max():.0f}")
print(f"flights under 20 min: {(d < 20).sum()}   length/rate mismatches: {rate_mismatch}")
print("params with any NaN:", int((df.nan_pct > 0).sum()), " always-constant params:", int((df.const_pct == 100).sum()))
print(df.sort_values("const_pct", ascending=False)[["rate", "desc", "const_pct", "nan_pct"]].head(12).to_string())
