import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flighthealth.detect import engine_columns, fit, score, sibling_z  # noqa: E402
from flighthealth.exceedances import flag_exceedances  # noqa: E402
from flighthealth.features import channel_of, engine_deviation  # noqa: E402
from flighthealth.trends import find_steps, ground_runs_before  # noqa: E402


def test_engine_deviation_is_distance_from_median_of_four():
    df = pd.DataFrame({"egt_1": [500.0], "egt_2": [510.0], "egt_3": [520.0], "egt_4": [560.0]})
    assert engine_deviation(df, "egt").iloc[0].tolist() == [-15, -5, 5, 45]


def test_channel_of():
    assert channel_of("egt_dev_3__mean") == "egt_3"
    assert channel_of("vrtg__p95") == "vrtg"
    assert channel_of("mw__on_s") == "mw"


def test_flag_exceedances_uses_direction():
    v = pd.DataFrame({"flight_id": [f"f{i}" for i in range(400)] * 2,
                      "rule": ["hi"] * 400 + ["lo"] * 400,
                      "phase": "cruise",
                      "value": list(np.arange(400.0)) * 2,
                      "direction": ["high"] * 400 + ["low"] * 400})
    out = flag_exceedances(v)
    hi = out[(out.rule == "hi") & out.exceeded].value
    lo = out[(out.rule == "lo") & out.exceeded].value
    assert hi.min() > 390 and lo.max() < 9


def _engine_table(n=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = {"flight_id": [f"f{i}" for i in range(n)], "phase": "cruise"}
    for fam in ("egt", "oip"):
        for k in range(1, 5):
            rows[f"{fam}_dev_{k}__mean"] = rng.normal(0, 1, n)
    return pd.DataFrame(rows)


def test_engine_model_flags_and_names_an_offset_engine():
    train = _engine_table()
    m = fit(train[["flight_id", "phase"] + engine_columns(train.columns)])
    test = _engine_table(50, seed=1)
    test.loc[0, "oip_dev_2__mean"] += 12
    s = score(m, test)
    assert s.pca_max_score.idxmax() == 0 and s.top_channel.iat[0] == "oip_2"
    z = sibling_z(m, test)
    assert z.sibling_channel.iat[0] == "oip_2" and z.sibling_z.iat[0] > 8


def test_find_steps_locates_a_step_and_ignores_noise():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 100), rng.normal(10, 1, 100)])
    s = find_steps(pd.Series(x, index=[f"f{i}" for i in range(200)]))
    assert len(s) == 1 and abs(int(s["at"].iat[0][1:]) - 100) <= 2 and abs(s.step.iat[0] - 10) < 1.5
    assert find_steps(pd.Series(rng.normal(0, 1, 200))).empty


def test_ground_runs_before():
    t = pd.date_range("2001-01-01", periods=6, freq="h")
    flights = pd.DataFrame({"flight_id": list("abcdef"), "start_utc": t, "n_legs": [1, 0, 0, 1, 1, 1]})
    runs = pd.DataFrame({"flight_id": ["b", "c"], "n1_max_4": [95.0, 20.0]})
    assert ground_runs_before(flights, runs, "d").flight_id.tolist() == ["b", "c"]
    assert ground_runs_before(flights, runs, "e").empty
    assert ground_runs_before(flights, runs, "e", flights_back=2).flight_id.tolist() == ["b", "c"]
