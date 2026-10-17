"""Which channels does the before/after classifier rely on? Permutation importance by channel
(all of a channel's features shuffled together), on held-out folds."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path("data/processed/ngafid")
OUT = Path("reports/phase3")
HEADER = ["flight_id", "before_after", "date_diff", "flight_length", "label", "hierarchy", "fold", "class",
          "target_class", "hclass", "number_flights_before"]
f = pd.read_parquet(ROOT / "flight_features.parquet")
cols = [c for c in f.columns if c not in HEADER]


def channel(c):
    base = c.split("|")[-1].split("__")[0]
    return base.split("_dev_")[0] + "_dev" if "_dev_" in base else base


groups = pd.Series(cols).groupby(pd.Series(cols).map(channel)).apply(list).to_dict()
rng = np.random.default_rng(0)
rows = []
for k in range(5):
    tr, te = f.fold != k, f.fold == k
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, l2_regularization=1.0, random_state=0)
    clf.fit(f.loc[tr, cols], f.loc[tr, "before_after"])
    X, y = f.loc[te, cols].copy(), f.loc[te, "before_after"].to_numpy()
    base = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    for g, gc in groups.items():
        drops = []
        for _ in range(3):
            Xp = X.copy()
            idx = rng.permutation(len(Xp))
            Xp[gc] = Xp[gc].to_numpy()[idx]
            drops.append(base - roc_auc_score(y, clf.predict_proba(Xp)[:, 1]))
        rows.append(dict(fold=k, channel=g, auc_drop=np.mean(drops)))
imp = pd.DataFrame(rows).groupby("channel").auc_drop.agg(["mean", "std"]).sort_values("mean", ascending=False)
imp.to_csv(OUT / "channel_importance.csv")
print(imp.round(4).head(15).to_string())
