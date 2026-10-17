"""Engine trend chart: each engine's cruise deviation from its siblings, flight by flight over time."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path("data/processed/dashlink")
out = Path(sys.argv[1])
fams = ["egt", "n1", "ff", "vib", "oit", "oip"]
f = pd.read_parquet(ROOT / "features" / "phase_features.parquet")
f = f[f.phase == "cruise"].merge(pd.read_parquet(ROOT / "catalog" / "flights.parquet")[["flight_id", "start_utc"]])
f = f.sort_values("start_utc")
fig, axes = plt.subplots(len(fams), 1, figsize=(11, 2.0 * len(fams)), sharex=True)
for ax, fam in zip(axes, fams):
    for k, c in zip(range(1, 5), ["#1f6fb2", "#d9822b", "#2a9d5c", "#c0392b"]):
        s = f.set_index("start_utc")[f"{fam}_dev_{k}__mean"]
        ax.plot(s.index, s, ".", ms=2, color=c, alpha=0.35)
        ax.plot(s.index, s.rolling(25, center=True, min_periods=10).median(), lw=1.4, color=c, label=f"eng {k}")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_ylabel(f"{fam} vs siblings", fontsize=8)
    ax.tick_params(labelsize=7)
axes[0].legend(ncol=4, fontsize=7, loc="upper right")
axes[-1].set_xlabel("flight date (dots = one flight's cruise average, lines = rolling median of 25 flights)", fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=110)
print(f["start_utc"].min(), f["start_utc"].max(), len(f))
