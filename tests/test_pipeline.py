import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from flighthealth.align import seconds_touched, to_1hz, to_highrate  # noqa: E402
from flighthealth.catalog import classify  # noqa: E402
from flighthealth.phases import label_seconds  # noqa: E402
from flighthealth.quality import channel_checks, cross_checks, frame_checks, runs  # noqa: E402
import safe_pickle  # noqa: E402

ALWAYS_AIR = lambda t: True  # noqa: E731


# --- alignment -----------------------------------------------------------

@pytest.mark.parametrize("rate", [1, 2, 4, 8, 16])
def test_impulse_lands_in_window_that_contains_it_never_earlier(rate):
    n_sec = 10
    for i in range(n_sec * rate):
        x = np.zeros(n_sec * rate)
        x[i] = 1.0
        out = to_1hz(x, rate, n_sec, "mean_minmax")["_max"]
        row = int(np.ceil(i / rate))  # sample at time i/rate belongs to window (t-1, t]
        assert np.flatnonzero(out == 1).tolist() == ([row] if row < n_sec else [])


def test_slow_channel_holds_last_value():
    x = np.array([10.0, 20.0, 30.0])  # 0.25 Hz: samples at t = 0, 4, 8
    out = to_1hz(x, 0.25, 10)[""]
    assert out.tolist() == [10, 10, 10, 10, 20, 20, 20, 20, 30, 30]


def test_discrete_max_keeps_short_flag():
    x = np.array([0, 0, 1, 0, 0, 0, 0, 0], float)  # 2 Hz
    assert to_1hz(x, 2, 4, "max")[""].tolist() == [0, 1, 0, 0]


def test_highrate_repeats_8hz_onto_16hz():
    assert to_highrate(np.array([1.0, 2.0]), 8, 4).tolist() == [1, 1, 2, 2]


def test_seconds_touched():
    assert seconds_touched(3.25, 3.25, 10) == slice(4, 5)
    assert seconds_touched(3.0, 5.5, 10) == slice(3, 7)


# --- catalog ---------------------------------------------------------------

def test_classify():
    assert classify("EGT_3", "EXHAUST GAS TEMPERATURE 3", "DEG") == ("continuous", "engine", 3)
    assert classify("WOW", "WEIGHT ON WHEELS", "") == ("discrete", "gear_brakes", None)
    assert classify("DATE_YEAR", "Date (Year)", "Year")[0] == "metadata"
    assert classify("DWPT", "DISTANCE TO WAYPOINT LSP", "")[0] == "continuous"


# --- quality checks --------------------------------------------------------

def test_runs():
    assert runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 2), (4, 4)]


def _checks(name, x, rate=1, **rule):
    rule = {"valid_min": np.nan, "valid_max": np.nan, "sentinels": "", "stuck_check": False} | rule
    return pd.DataFrame(channel_checks("f", name, np.asarray(x, float), rate, rule, ALWAYS_AIR))


def test_sentinel_flagged_and_not_double_counted_as_spike():
    x = np.full(50, 1.0) + np.sin(np.arange(50) / 5) * 0.1
    x[20] = -3.375
    f = _checks("VRTG", x, 8, valid_min=-1.5, valid_max=4, sentinels="-3.375")
    assert f.check.tolist() == ["sentinel"]
    assert f.t_start_s.iloc[0] == 20 / 8


def test_out_of_range_with_float_tolerance():
    x = np.array([0, 90, -180.0000005, 180, 181.0])
    f = _checks("WD", x, valid_min=-180, valid_max=180)
    assert f.check.tolist() == ["out_of_range"] and f.t_start_s.iloc[0] == 4


def test_single_sample_spike():
    x = np.linspace(100, 110, 60)
    x[30] = 400
    f = _checks("CAS", x)
    assert f.check.tolist() == ["spike"] and f.t_start_s.iloc[0] == 30


def test_engine_dropout_to_zero():
    x = np.full(60, 90.0)
    x[10:13] = 0
    f = _checks("N2_1", x)
    assert f.check.tolist() == ["dropout"] and (f.t_start_s.iloc[0], f.t_end_s.iloc[0]) == (10, 12)


def test_stuck_sensor_only_when_long_enough():
    x = np.concatenate([np.linspace(50, 60, 100), np.full(400, 60.0), np.linspace(60, 70, 100)])
    f = _checks("EGT_1", x, stuck_check=True)
    assert f.check.tolist() == ["stuck"]
    assert _checks("EGT_1", x[:300], stuck_check=True).empty


def test_frame_counter_learns_its_step_and_wraps():
    f = np.array([1, 5, 9, 13, 21, 25, 29, 4092, 0, 3, 7, 11])  # step 4, one real gap, then a wrap 4092 -> 0 -> 3
    flags = frame_checks("f", f, 0.25)
    assert [x["detail"] for x in flags] == ["counter 13 -> 21", "counter 29 -> 4092"]


def test_cross_check_needs_sustained_disagreement():
    a = np.zeros(40)
    b = np.zeros(40)
    b[5:8] = 500     # 3 s: ignored
    b[20:35] = 500   # 15 s: flagged
    flags = cross_checks("f", pd.DataFrame({"bal1": a, "bal2": b}), np.ones(40, bool))
    assert [(f["t_start_s"], f["t_end_s"]) for f in flags] == [(20, 34)]


# --- phases ----------------------------------------------------------------

def test_phases_on_simple_profile():
    gs = np.concatenate([np.zeros(100), np.linspace(0, 140, 30), np.full(1000, 300), np.linspace(140, 0, 40),
                         np.zeros(100)])
    alt = np.concatenate([np.zeros(130), np.linspace(0, 20000, 300), np.full(400, 20000),
                          np.linspace(20000, 0, 300), np.zeros(140)])
    air = np.zeros(gs.size, bool)
    air[130:1130] = True
    eng = np.ones(gs.size, bool)
    eng[:20] = False
    lab, info = label_seconds(air, gs, alt, eng)
    order = [p for i, p in enumerate(lab) if i == 0 or p != lab[i - 1]]
    assert order == ["parked", "taxi_out", "takeoff", "climb", "cruise", "descent", "approach", "landing", "taxi_in"]
    assert info == dict(n_legs=1, has_takeoff=True, has_landing=True)


# --- safe pickle -----------------------------------------------------------

class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_safe_pickle_loads_arrays_and_blocks_code(tmp_path):
    good = tmp_path / "good.pkl"
    # protocol 4 = the array format the NGAFID file was written in (5 uses a different rebuilder)
    good.write_bytes(pickle.dumps({1: np.arange(3, dtype=np.float16)}, protocol=4))
    assert safe_pickle.load(good)[1].tolist() == [0, 1, 2]
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(pickle.dumps(_Evil()))
    with pytest.raises(pickle.UnpicklingError, match="blocked"):
        safe_pickle.load(bad)


def test_filename_with_part_suffix():
    from flighthealth.dashlink_io import parse_filename
    assert parse_filename("670200105081116_1.mat")[0] == 670
    assert str(parse_filename("670200104120549.mat")[1]) == "2001-04-12 05:49:00"


# --- end to end on a synthetic recording ----------------------------------

def _write_synthetic_mat(path):
    import scipy.io as sio
    n = 1200  # seconds
    t1 = np.arange(n)
    air = (t1 >= 300) & (t1 < 900)
    alt = np.where(air, np.minimum((t1 - 300) * 50, 15000), 0.0)
    alt = np.where(t1 >= 750, np.maximum(15000 - (t1 - 750) * 100, 0), alt) * air
    vrtg = np.ones(n * 8)
    vrtg[8 * 500 + 3] = -3.375  # one dropout sample inside second 501's window

    def ch(data, rate, units="", desc=""):
        return {"data": np.asarray(data, float), "Rate": float(rate), "Units": units, "Description": desc, "Alpha": ""}

    sio.savemat(path, {
        "WOW": ch(air.astype(float), 1, "", "WEIGHT ON WHEELS"),
        "GS": ch(np.repeat(np.where(air, 250, 0.0), 4), 4, "KNOTS", "GROUND SPEED LSP"),
        "ALT": ch(np.repeat(alt, 4), 4, "FEET", "PRESSURE ALTITUDE LSP"),
        "N2_1": ch(np.repeat(np.where(t1 < 60, 0.0, 80.0), 4), 4, "%RPM", "CORE SPEED 1 LSP"),
        "LATP": ch(45.0 + t1 * 1e-4, 1, "DEG", "LATITUDE POSITION LSP"),
        "LONP": ch(-93.0 + t1 * 1e-4, 1, "DEG", "LONGITUDE POSITION LSP"),
        "PH": ch(np.zeros(n), 1, "", "FLIGHT PHASE FROM ACMS"),
        "VRTG": ch(vrtg, 8, "G", "VERTICAL ACCELERATION"),
        "FRMC": ch(np.arange(0, n // 4) * 4 % 4096, 0.25, "", "FRAME COUNTER"),
    })


def test_end_to_end_drops_bad_sample_before_averaging(tmp_path):
    from flighthealth.catalog import build_catalog, load_rules
    from flighthealth.pipeline import process_flight
    path = tmp_path / "670200101010000.mat"
    _write_synthetic_mat(path)
    cat = build_catalog([str(path)], load_rules(ROOT / "config" / "param_rules.csv"))
    r = process_flight(path, cat)
    df, flags = r["aligned"], r["flags"]
    assert len(df) == 1200
    assert df.loc[501, "vrtg"] == pytest.approx(1.0)        # mean of the 7 good samples, not dragged down
    assert df.loc[501, "vrtg_min"] == pytest.approx(1.0)
    # the dropout sample, plus N2 held at exactly 80% through the flight (a stuck sensor)
    assert sorted(zip(flags.param, flags.check)) == [("n2_1", "stuck"), ("vrtg", "sentinel")]
    assert r["flight"]["has_takeoff"] and r["flight"]["has_landing"] and r["flight"]["start_time_source"] == "filename"
    assert set(df.phase.unique()) >= {"taxi_out", "takeoff", "climb", "cruise", "descent", "taxi_in"}
