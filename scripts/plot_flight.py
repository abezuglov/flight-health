"""Engine channels for one flight, all four engines overlaid, phases shaded."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flighthealth.pipeline import load_flight  # noqa: E402

fid, out = sys.argv[1], Path(sys.argv[2])
fams = sys.argv[3].split(",") if len(sys.argv) > 3 else ["pla", "n1", "n2", "egt", "ff", "oit", "oip", "vib"]
df = load_flight("data/processed/dashlink", fid)
t = df.t_s / 60
fig, axes = plt.subplots(len(fams) + 1, 1, figsize=(11, 1.7 * (len(fams) + 1)), sharex=True)
axes[0].plot(t, df.alt, color="0.3", lw=1)
axes[0].set_ylabel("alt ft", fontsize=8)
cruise = df.phase == "cruise"
for ax in axes:
    ax.fill_between(t, 0, 1, where=cruise, transform=ax.get_xaxis_transform(), color="0.92", lw=0)
    ax.tick_params(labelsize=7)
for ax, fam in zip(axes[1:], fams):
    for k, c in zip(range(1, 5), ["#1f6fb2", "#d9822b", "#2a9d5c", "#c0392b"]):
        ax.plot(t, df[f"{fam}_{k}"], lw=0.8, color=c, label=f"eng {k}")
    ax.set_ylabel(fam, fontsize=8)
axes[1].legend(ncol=4, fontsize=7, loc="upper right")
axes[-1].set_xlabel("minutes from recording start (grey = cruise)", fontsize=8)
fig.suptitle(fid, fontsize=9)
fig.tight_layout()
fig.savefig(out, dpi=110)
