"""How well do the detectors find engine faults of known size?

Fit on 70% of flights. Into the cruise phase of each held-out flight, inject one fault on one
random engine, rebuild that flight's cruise features, and score it. Detection = score above the
95th percentile of the same held-out flights without a fault (a 5% false-alarm rate). Localization
= the channel the model blames most is the one we changed. Baseline = the cruise limit rules with
thresholds set on the training flights.
"""
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flighthealth.detect import engine_columns, fit, score, sibling_z  # noqa: E402
from flighthealth.exceedances import RULES, TAIL_PCT, rule_values  # noqa: E402
from flighthealth.features import feature_channels, flight_features  # noqa: E402
from flighthealth.pipeline import load_flight  # noqa: E402

ROOT = Path("data/processed/dashlink")
OUT = Path("reports/phase2")
SEED = 0

# fault name -> (channel family, sizes, how to apply to the cruise values of one engine)
FAULTS = {
    "control": ("egt", [0], lambda x, s: x),  # nothing changed: detection must stay at the 5% false-alarm rate
    "egt_offset": ("egt", [5, 10, 20, 40], lambda x, s: x + s),
    "egt_drift": ("egt", [10, 20, 40, 80], lambda x, s: x + np.linspace(0, s, len(x))),
    "n1_offset": ("n1", [0.25, 0.5, 1, 2], lambda x, s: x - s),
    "vib_offset": ("vib", [0.025, 0.05, 0.1, 0.2], lambda x, s: x + s),
    "oip_drop": ("oip", [2.5, 5, 10, 20], lambda x, s: x - s),
    "oit_rise": ("oit", [2.5, 5, 10, 20], lambda x, s: x + s),
}
CRUISE_RULES = [r[0] for r in RULES if r[1] == "cruise"]


def _inject(args):
    fid, channels = args
    rng = np.random.default_rng(zlib.crc32(fid.encode()))  # stable across runs, unlike hash()
    df = load_flight(ROOT, fid)
    cruise = df[df.phase == "cruise"].copy()
    rows, rules = [], []
    for fault, (fam, sizes, fn) in FAULTS.items():
        k = int(rng.integers(1, 5))
        for s in sizes:
            d = cruise.copy()
            col = f"{fam}_{k}"
            d[col] = fn(d[col].to_numpy(float), s).astype(np.float32)
            meta = dict(flight_id=fid, fault=fault, size=s, channel=col)
            rows += [r | meta for r in flight_features(d, channels)]
            rules += [r | meta for r in rule_values(d) if r["rule"] in CRUISE_RULES]
    return rows, rules


if __name__ == "__main__":
    feats = pd.read_parquet(ROOT / "features" / "phase_features.parquet")
    rv = pd.read_parquet(ROOT / "features" / "rule_values.parquet")
    flights = feats.flight_id.unique()
    rng = np.random.default_rng(SEED)
    test = set(rng.choice(flights, size=int(0.3 * len(flights)), replace=False))
    train_c = feats[(feats.phase == "cruise") & ~feats.flight_id.isin(test)]
    test_c = feats[(feats.phase == "cruise") & feats.flight_id.isin(test)]
    model = fit(train_c)
    eng_cols = ["flight_id", "phase"] + engine_columns(train_c.columns)
    eng_model = fit(train_c[eng_cols])

    # rule thresholds from training flights only
    tr = rv[(rv.phase == "cruise") & ~rv.flight_id.isin(test)]
    thr = {r: (np.nanpercentile(g.value, TAIL_PCT) if g.direction.iat[0] == "high"
               else np.nanpercentile(g.value, 100 - TAIL_PCT), g.direction.iat[0]) for r, g in tr.groupby("rule")}

    def rule_hit(df):
        hit = df.apply(lambda r: (r.value > thr[r.rule][0]) if thr[r.rule][1] == "high" else (r.value < thr[r.rule][0]),
                       axis=1)
        return df.assign(hit=hit)

    clean = score(model, test_c)
    # Thresholds on raw scores, calibrated on the clean held-out flights. (Percentiles of the
    # training scores saturate: PCA reconstructs its own training flights better than new ones.)
    def all_scores(f):
        g = score(model, f)
        e = score(eng_model, f[eng_cols]).add_prefix("eng_")
        z = sibling_z(eng_model, f[eng_cols])
        return pd.concat([g.reset_index(drop=True), e.reset_index(drop=True), z.reset_index(drop=True)], axis=1)

    DETECTORS = {"global_pca_sum": "pca_score", "global_pca_max": "pca_max_score", "global_forest": "forest_score",
                 "engine_pca_sum": "eng_pca_score", "engine_pca_max": "eng_pca_max_score",
                 "engine_forest": "eng_forest_score", "sibling_z": "sibling_z"}
    clean_raw = all_scores(test_c)

    def add_engine_combined(f):
        # whichever of the two engine detectors is more extreme, as a rank among clean held-out flights
        r = lambda col: np.array([np.mean(clean_raw[col] < v) for v in f[col]])  # noqa: E731
        return f.assign(engine_combined=np.maximum(r("eng_pca_max_score"), r("sibling_z")))

    DETECTORS["engine_combined"] = "engine_combined"
    clean = add_engine_combined(clean_raw)
    cut = {d: np.percentile(clean[c], 95) for d, c in DETECTORS.items()}
    clean_rules = rule_hit(rv[(rv.phase == "cruise") & rv.flight_id.isin(test)])
    rule_fpr = clean_rules.groupby("flight_id").hit.any().mean()

    channels = feature_channels(pd.read_parquet(ROOT / "catalog" / "parameters.parquet"))
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(_inject, [(f, channels) for f in sorted(test)], chunksize=2))
    inj = pd.DataFrame([r for rows, _ in res for r in rows])
    inj_rules = rule_hit(pd.DataFrame([r for _, rr in res for r in rr]))

    meta = inj[["flight_id", "fault", "size", "channel"]]
    s = pd.concat([meta.reset_index(drop=True),
                   add_engine_combined(all_scores(inj.drop(columns=["fault", "size", "channel"])))
                   .drop(columns=["flight_id", "phase"])],
                  axis=1)
    for d, c in DETECTORS.items():
        s[f"det_{d}"] = s[c] > cut[d]
    s["global_names_channel"] = s.top_channel == s.channel
    s["engine_names_channel"] = s.eng_top_channel == s.channel
    s["sibling_names_channel"] = s.sibling_channel == s.channel
    rule_det = inj_rules.groupby(["flight_id", "fault", "size"]).hit.any().rename("det_rules")
    s = s.merge(rule_det.reset_index(), on=["flight_id", "fault", "size"], how="left")

    agg = {"n": ("flight_id", "size"), "rules": ("det_rules", "mean")}
    agg |= {d: (f"det_{d}", "mean") for d in DETECTORS}
    agg |= {k: (k, "mean") for k in ("global_names_channel", "engine_names_channel", "sibling_names_channel")}
    summary = s.groupby(["fault", "size"]).agg(**agg).round(2)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "injection_detection.csv")
    s.to_csv(OUT / "injection_scores.csv", index=False)
    print(f"train flights {len(flights) - len(test)}, test flights {len(test)}")
    print(f"false-alarm rate on clean test flights: models 5% by construction, rules {rule_fpr:.1%}")
    pd.set_option("display.width", 250)
    print(summary.to_string())
