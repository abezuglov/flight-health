"""Phase 3: do anomaly scores and classifiers separate flights before maintenance from after?

Uses the dataset authors' 5 folds throughout (stratified by fault type, mostly keeping each
maintenance event inside one fold).

A. Unsupervised: fit on post-maintenance flights from the training folds only (repaired = healthy
   reference), score the held-out fold, report ROC AUC for before vs after.
B. Supervised: gradient boosting on three feature groups (all / engine only / operational only),
   accuracy and AUC per fold, plus a shuffled-label control.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flighthealth.detect import fit, score, sibling_z  # noqa: E402

ROOT = Path("data/processed/ngafid")
OUT = Path("reports/phase3")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0

HEADER = ["flight_id", "before_after", "date_diff", "flight_length", "label", "hierarchy", "fold", "class",
          "target_class", "hclass", "number_flights_before"]
f = pd.read_parquet(ROOT / "flight_features.parquet")
y = f.before_after.to_numpy()
feat_cols = [c for c in f.columns if c not in HEADER]

OPERATIONAL = ("ias", "vspd", "alt", "normac", "oat", "fqty_l", "fqty_r")
ENGINE = ("cht", "egt", "oit", "oip", "ff", "rpm")


def channel(col: str) -> str:
    base = col.split("|")[-1].split("__")[0]
    return base.split("_dev_")[0] if "_dev_" in base else base.split("_spread")[0]


def is_group(col, names):
    return channel(col).startswith(names)


groups = {
    "all": feat_cols,
    "engine": [c for c in feat_cols if is_group(c, ENGINE)],
    "operational": [c for c in feat_cols if is_group(c, OPERATIONAL) or c == "duration_s" or c.endswith("_s")],
    "cylinders_vs_siblings": [c for c in feat_cols if "_dev_" in c or "_spread" in c],
    # oil pressure steps up ~1.3 psi at the maintenance date for almost every fault type (an oil
    # service during the shop visit is the likely cause), so test with and without the oil channels
    "oil_only": [c for c in feat_cols if is_group(c, ("oip", "oit"))],
    "all_except_oil": [c for c in feat_cols if not is_group(c, ("oip", "oit"))],
}

# --- A. unsupervised -----------------------------------------------------------
cyl = groups["cylinders_vs_siblings"]
parts = []
for k in range(5):
    tr = f[(f.fold != k) & (f.before_after == 0)]
    te = f[f.fold == k]
    m = fit(tr[["flight_id"] + cyl].assign(phase="flight"))
    s = score(m, te[["flight_id"] + cyl].assign(phase="flight"))
    z = sibling_z(m, te[cyl])
    parts.append(pd.concat([s.reset_index(drop=True), z.reset_index(drop=True),
                            te[["before_after", "label", "number_flights_before", "fold"]].reset_index(drop=True)],
                           axis=1))
u = pd.concat(parts, ignore_index=True)
u.to_csv(OUT / "unsupervised_scores.csv", index=False)


def auc(d, col):
    return roc_auc_score(d.before_after, d[col]) if d.before_after.nunique() == 2 else np.nan


rows = [dict(subset="all flights", n=len(u), **{c: auc(u, c) for c in ("pca_score", "pca_max_score", "sibling_z")})]
for lab, d in u.groupby("label"):
    if len(d) >= 200:
        rows.append(dict(subset=lab, n=len(d), **{c: auc(d, c) for c in ("pca_score", "pca_max_score", "sibling_z")}))
ua = pd.DataFrame(rows)
ua.to_csv(OUT / "unsupervised_auc.csv", index=False)

# --- B. supervised ---------------------------------------------------------------
def cv(cols, shuffle=False):
    accs, aucs, oof = [], [], np.full(len(f), np.nan)
    rng = np.random.default_rng(SEED)
    for k in range(5):
        tr, te = (f.fold != k).to_numpy(), (f.fold == k).to_numpy()
        ytr = rng.permutation(y[tr]) if shuffle else y[tr]
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                             l2_regularization=1.0, random_state=SEED)
        clf.fit(f.loc[tr, cols], ytr)
        p = clf.predict_proba(f.loc[te, cols])[:, 1]
        oof[te] = p
        accs.append(accuracy_score(y[te], p > 0.5))
        aucs.append(roc_auc_score(y[te], p))
    return np.mean(accs), np.std(accs), np.mean(aucs), oof


sup = []
oofs = {}
for name, cols in groups.items():
    a, sd, au, oof = cv(cols)
    oofs[name] = oof
    sup.append(dict(features=name, n_features=len(cols), accuracy=a, accuracy_sd=sd, auc=au))
a, sd, au, _ = cv(groups["all"], shuffle=True)
sup.append(dict(features="all (shuffled labels)", n_features=len(groups["all"]), accuracy=a, accuracy_sd=sd, auc=au))
sup = pd.DataFrame(sup)
sup.to_csv(OUT / "supervised_cv.csv", index=False)

# out-of-fold predictions for every feature set, and AUC per fault type
pred = f[["flight_id", "before_after", "label", "number_flights_before", "fold"]].copy()
for name, oof in oofs.items():
    pred[f"p_{name}"] = oof
pred.to_parquet(OUT / "oof_predictions.parquet", index=False)
per_label = pred.groupby("label").apply(
    lambda d: pd.Series({"n": len(d), **{n: roc_auc_score(d.before_after, d[f"p_{n}"]) for n in oofs}}),
    include_groups=False).sort_values("n", ascending=False)
per_label.to_csv(OUT / "auc_by_fault_type.csv")

# early detection: share of "before" flights caught, by how many flights before maintenance
f["p_all"] = oofs["all"]
early = (f[f.before_after == 1].groupby("number_flights_before")
         .apply(lambda d: pd.Series({"n": len(d), "recall_at_0.5": (d.p_all > 0.5).mean()}), include_groups=False))
early.to_csv(OUT / "early_detection.csv")

pd.set_option("display.width", 200)
print("A. unsupervised, trained on post-maintenance flights only (ROC AUC, 0.5 = chance)")
print(ua.round(3).to_string(index=False))
print("\nB. supervised, authors' 5 folds (paper best: ConvMHSA 76.0% accuracy)")
print(sup.round(3).to_string(index=False))
print("\nrecall of 'before' flights by flights-before-maintenance (0 = last flight before):")
print(early.round(3).to_string())
print("\nAUC by fault type:")
print(per_label.round(3).to_string())
