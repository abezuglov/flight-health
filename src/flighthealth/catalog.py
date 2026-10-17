"""Parameter catalog: what each DASHlink channel is and how the pipeline treats it."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .dashlink_io import read_mat

METADATA_RE = re.compile(
    r"SYNC WORD|SERIAL NUMBER|DATABASE ID|AIRCRAFT NUMBER|ENGINE HOURS|ENGINE CYCLE|^DATE|GREENWICH|ACMS TIMING",
    re.IGNORECASE,
)
# Hand-checked notes that a reader of the data needs and the file does not say.
NOTES = {
    "WOW": "inverted: 1 = airborne, 0 = on ground (checked against GS and RALT)",
    "PH": "aircraft's own flight-phase code; meanings undocumented",
    "VRTG": "-3.375 g appears as single-sample dropouts mid-flight",
}
# Channels with no units that are still continuous measurements.
CONTINUOUS_WITHOUT_UNITS = {"DWPT"}
COUNTERS = {"FRMC"}

# Group by mnemonic prefix; first match wins.
GROUPS = [
    ("engine", r"^(N1_|N2_|EGT_|FF_|VIB_|OIT_|OIP_|PLA_|N1T$|N1C$|N1CO$|FADF$|FADS$|OIPL$|POVT$|FIRE_|ESN_|ECYC_|EHRS_)"),
    ("fuel", r"^FQTY_"),
    ("air_data", r"^(ALT$|ALTR$|BAL\d$|CAS$|CASM$|TAS$|MACH$|PSA$|PI$|PS$|PT$|SAT$|TAT$|AOA|RALT$|IVV$)"),
    ("attitude", r"^(PTCH|ROLL|VRTG|LATG|LONG$|FPAC|BLAC|CTAC|TH$|MH$)"),
    ("nav", r"^(GS$|TRK|DA$|WS$|WD$|LATP|LONP|LOC$|GLS$|MRK$|DWPT|ILSF|CRSS)"),
    ("controls", r"^(RUDD|RUDP|CCP|CWP|AIL_|SPL_|ELEV_|FLAP|PTRM|ABRK)"),
    ("autopilot", r"^(APFD|VMODE|LMOD|A_T$|ATEN|TMODE|ALTS|HDGS|CASS|VSPS|MNS$|DFGS|FGC3|TMAG)"),
    ("gear_brakes", r"^(LGDN|LGUP|WOW|NSQT|MSQT_|BPGR_|BPYR_)"),
    ("systems", r"^(BLV|EAI|PACK|TAI|WAI_|HYD|SPLY|SPLG)"),
    ("warnings", r"^(MW$|SMOK|SMKB|PUSH|SHKR|APUF|TOCW|WSHR|GPWS|TCAS|CALT)"),
    ("crew_events", r"^(VHF|HF\d|EVNT|SNAP|PH$)"),
]


def to_snake(mnemonic: str) -> str:
    return mnemonic.lower()


def classify(mnemonic: str, description: str, units: str) -> tuple[str, str, int | None]:
    """Return (kind, group, engine_idx) from the channel's own metadata."""
    if mnemonic in COUNTERS:
        kind = "counter"
    elif METADATA_RE.search(description) or mnemonic.startswith("VAR_"):
        kind = "metadata"
    elif units == "" and mnemonic not in CONTINUOUS_WITHOUT_UNITS:
        kind = "discrete"
    else:
        kind = "continuous"
    group = "time_id" if kind in ("metadata", "counter") else "other"
    for name, pattern in GROUPS:
        if re.match(pattern, mnemonic):
            group = name if kind not in ("metadata", "counter") else group
            break
    engine_idx = None
    m = re.search(r"_(\d)$", mnemonic)
    if m and group in ("engine", "fuel"):
        engine_idx = int(m.group(1))
    return kind, group, engine_idx


def observe(files: list[str]) -> pd.DataFrame:
    """One pass over all flights: per-channel metadata plus observed value ranges."""
    meta, per_flight = {}, []
    for f in files:
        for k, c in read_mat(f).items():
            x = c.data.astype(float)
            meta.setdefault(k, (c.description, c.units, c.rate))
            per_flight.append((k, np.nanmin(x), np.nanmax(x), np.percentile(x, 1), np.percentile(x, 99)))
    pf = pd.DataFrame(per_flight, columns=["mnemonic", "lo", "hi", "p01", "p99"])
    g = pf.groupby("mnemonic")
    obs = pd.DataFrame({
        "obs_min": g.lo.min(), "obs_max": g.hi.max(),
        "obs_p01_median": g.p01.median(), "obs_p99_median": g.p99.median(),
        "constant_everywhere": g.apply(lambda d: bool((d.lo == d.hi).all() and d.lo.nunique() == 1), include_groups=False),
    })
    m = pd.DataFrame.from_dict(meta, orient="index", columns=["description", "units", "rate_hz"])
    return m.join(obs).rename_axis("mnemonic").reset_index()


def load_rules(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sentinels": str, "stuck_check": str}).fillna({"sentinels": "", "stuck_check": ""})


def apply_rules(mnemonics: pd.Series, rules: pd.DataFrame) -> pd.DataFrame:
    """First matching rule per channel -> valid_min, valid_max, sentinels, stuck_check."""
    rows = []
    for mn in mnemonics:
        hit = next((r for r in rules.itertuples() if re.match(r.mnemonic_regex, mn)), None)
        rows.append(dict(
            valid_min=hit.valid_min if hit else np.nan,
            valid_max=hit.valid_max if hit else np.nan,
            sentinels=hit.sentinels if hit else "",
            stuck_check=bool(hit and hit.stuck_check == "y"),
        ))
    return pd.DataFrame(rows, index=mnemonics.index)


def build_catalog(files: list[str], rules: pd.DataFrame | None = None) -> pd.DataFrame:
    df = observe(files)
    cls = df.apply(lambda r: classify(r.mnemonic, r.description, r.units), axis=1, result_type="expand")
    df[["kind", "group", "engine_idx"]] = cls
    df.loc[df.constant_everywhere & (df.kind != "metadata"), "kind"] = "constant"
    df["engine_idx"] = df.engine_idx.astype("Int8")
    df.insert(0, "param", df.mnemonic.map(to_snake))
    rules = rules if rules is not None else pd.DataFrame(columns=["mnemonic_regex"])
    df = df.join(apply_rules(df.mnemonic, rules))
    df["use_for_modeling"] = df.kind.isin(["continuous", "discrete"])
    df["note"] = df.mnemonic.map(NOTES).fillna("")
    cols = ["param", "mnemonic", "description", "units", "rate_hz", "kind", "group", "engine_idx",
            "valid_min", "valid_max", "sentinels", "stuck_check", "use_for_modeling", "note",
            "obs_min", "obs_max", "obs_p01_median", "obs_p99_median"]
    return df[cols].sort_values(["group", "param"]).reset_index(drop=True)
