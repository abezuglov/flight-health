"""Score every full flight-phase (PCA + isolation forest) and apply the limit rules."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flighthealth.detect import fit, score  # noqa: E402
from flighthealth.exceedances import flag_exceedances  # noqa: E402

ROOT = Path("data/processed/dashlink")
OUT = Path("reports/phase2")
OUT.mkdir(parents=True, exist_ok=True)

feats = pd.read_parquet(ROOT / "features" / "phase_features.parquet")
scores = pd.concat([score(fit(g), g) for _, g in feats.groupby("phase")], ignore_index=True)
scores.sort_values("combined_pct", ascending=False).to_csv(OUT / "anomaly_scores.csv", index=False)

rules = flag_exceedances(pd.read_parquet(ROOT / "features" / "rule_values.parquet"))
rules.to_csv(OUT / "rule_values.csv", index=False)
ex = rules[rules.exceeded]
ex.to_csv(OUT / "exceedances.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print("rule thresholds:\n", rules.groupby("rule")[["threshold"]].first().round(3).to_string())
print(f"\nexceedances: {len(ex)} on {ex.flight_id.nunique()} flights")
top = scores[scores.combined_pct >= 99].sort_values(["combined_pct", "pca_pct"], ascending=False)
print(f"\nflight-phases above the 99th percentile on both models combined: {len(top)}")
print(top.head(25)[["flight_id", "phase", "pca_pct", "forest_pct", "top_channels"]].to_string(index=False))
corr = scores[["pca_score", "forest_score"]].corr(method="spearman").iat[0, 1]
print(f"\nSpearman correlation PCA vs forest: {corr:.2f}")
hi = set(map(tuple, scores.loc[scores.combined_pct >= 95, ["flight_id", "phase"]].to_numpy()))
exset = set(map(tuple, ex[["flight_id", "phase"]].to_numpy()))
print(f"rule exceedances whose flight-phase is also in the models' top 5%: {len(hi & exset)} of {len(exset)}")

# --- engine health on the real flights -------------------------------------
# Each flight is scored by a model fit on the other 80% of flights (5-fold cross-fitting),
# so no flight is judged by a model that has already seen it.
import numpy as np  # noqa: E402
from flighthealth.detect import engine_columns, sibling_z  # noqa: E402

rng = np.random.default_rng(0)
cruise = feats[feats.phase == "cruise"].reset_index(drop=True)
fold = rng.permutation(len(cruise)) % 5
eng_cols = ["flight_id", "phase"] + engine_columns(cruise.columns)
parts = []
for k in range(5):
    m = fit(cruise.loc[fold != k, eng_cols])
    test = cruise.loc[fold == k, eng_cols]
    parts.append(pd.concat([score(m, test).reset_index(drop=True),
                            sibling_z(m, test).reset_index(drop=True)], axis=1))
eng = pd.concat(parts, ignore_index=True)
rank = lambda c: eng[c].rank(pct=True)  # noqa: E731
eng["engine_combined"] = np.maximum(rank("pca_max_score"), rank("sibling_z"))
eng["blamed_channel"] = np.where(rank("pca_max_score") >= rank("sibling_z"), eng.top_channel, eng.sibling_channel)
eng = eng.sort_values("engine_combined", ascending=False)
eng.to_csv(OUT / "engine_scores_cruise.csv", index=False)
print("\ntop engine anomalies in cruise (cross-fitted):")
print(eng.head(15)[["flight_id", "engine_combined", "blamed_channel", "sibling_z", "top_channels"]].to_string(index=False))
print("\nblamed channel among the top 5%:", eng.head(int(0.05 * len(eng))).blamed_channel.value_counts().to_dict())
