import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flighthealth.ngafid import COLUMNS, clean_flight, flight_features, rules_table, segment  # noqa: E402


def _flight(n=900):
    t = np.arange(n)
    d = {c: np.full(n, 0.0) for c in COLUMNS}
    d["rpm"] = np.where(t < 100, 1000, np.where(t < 200, 1800, 2400.0))
    d["ias"] = np.where(t < 300, 0, 100.0)
    d["vspd"] = np.where((t >= 300) & (t < 500), 700, 0.0)
    for k, off in zip(range(1, 5), (0, 5, -5, 40)):  # cylinder 4 runs 40 F hot
        d[f"cht_{k}"] = 350 + off + np.sin(t / 50)
        d[f"egt_{k}"] = 1300 + off + np.cos(t / 40)
    d["volt1"] = d["volt2"] = np.full(n, 28.0)
    d["oip"] = np.full(n, 70.0) + np.sin(t / 30)
    return pd.DataFrame(d)


def test_segments():
    lab = segment(_flight())
    assert set(lab[100:200]) == {"runup"}
    assert (lab[320:480] == "climb").mean() > 0.9
    assert (lab[550:900] == "cruise").all()


def test_clean_flight_drops_bad_values_and_logs_them():
    df = _flight()
    df.loc[400, "oip"] = 500.0          # impossible oil pressure
    df.loc[[10, 11], "fqty_l"] = 54.0   # full-scale fuel reading
    rules = rules_table(ROOT / "config" / "ngafid_rules.csv")
    clean, flags = clean_flight(7, df.to_numpy(np.float16), rules)
    assert np.isnan(clean.loc[400, "oip"]) and np.isnan(clean.loc[10, "fqty_l"])
    got = {(f["param"], f["check"]) for f in flags}
    assert ("oip", "out_of_range") in got and ("fqty_l", "out_of_range") in got


def test_cylinder_vs_siblings_feature_finds_hot_cylinder():
    row = flight_features(1, _flight())
    devs = {k: row[f"cruise|cht_dev_{k}__mean"] for k in range(1, 5)}
    assert max(devs, key=lambda k: abs(devs[k])) == 4 and devs[4] > 30
