"""Cruise oil pressure by day relative to maintenance, per fault family (NGAFID)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

f = pd.read_parquet("data/processed/ngafid/flight_features.parquet")
fam = np.select([f.label.str.startswith("baffle"), f.label.str.startswith("intake"),
                 f.label.str.startswith("rocker"), f.label.str.startswith(("cylinder",)),
                 f.label.str.startswith(("engine", "oil"))],
                ["baffle (cooling air)", "intake", "rocker cover", "cylinder", "engine / oil"], "other")
f["family"] = fam
col = "cruise|oip__mean"
fig, ax = plt.subplots(figsize=(7, 4.2))
colors = ["#1f6fb2", "#d9822b", "#2a9d5c", "#c0392b", "#7b52ab", "#888888"]
for (name, d), c in zip(f.groupby("family"), colors):
    g = d.groupby("date_diff")[col]
    m, se = g.mean(), g.std() / np.sqrt(g.count())
    ax.errorbar(m.index, m, yerr=1.96 * se, marker="o", ms=4, lw=1.2, capsize=2, color=c,
                label=f"{name} (n={len(d)})")
ax.axvline(0, color="0.4", lw=0.8, ls="--")
ax.text(0.05, ax.get_ylim()[1], " maintenance", va="top", fontsize=8, color="0.3")
ax.set_xticks([-2, -1, 1, 2])
ax.set_xticklabels(["2 days\nbefore", "1 day\nbefore", "1 day\nafter", "2 days\nafter"], fontsize=8)
ax.set_ylabel("mean cruise oil pressure (psi), 95% CI", fontsize=8)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7, loc="lower right")
ax.set_title("Oil pressure steps up at the shop visit for most fault types", fontsize=9)
fig.tight_layout()
Path("reports/phase3/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("reports/phase3/figures/oil_pressure_step.png", dpi=120)
