"""Read NASA DASHlink .mat flight recordings."""
from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

# e.g. 670200104120549.mat -> tail 670, 2001-04-12 05:49; a _1 suffix marks a second recording in the same minute
FILENAME_RE = re.compile(r"^(?P<tail>\d{3})(?P<stamp>\d{12})(?:_(?P<part>\d+))?\.mat$")


@dataclass
class Channel:
    mnemonic: str
    data: np.ndarray
    rate: float
    units: str
    description: str


def flight_id(path: str | Path) -> str:
    return Path(path).stem


def parse_filename(path: str | Path) -> tuple[int, dt.datetime]:
    m = FILENAME_RE.match(Path(path).name)
    if not m:
        raise ValueError(f"unexpected DASHlink file name: {Path(path).name}")
    return int(m["tail"]), dt.datetime.strptime(m["stamp"], "%Y%m%d%H%M")


def read_mat(path: str | Path) -> dict[str, Channel]:
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    out = {}
    for key, s in m.items():
        if key.startswith("__"):
            continue
        units = s.Units if isinstance(s.Units, str) else ""
        out[key] = Channel(key, np.atleast_1d(np.asarray(s.data)), float(s.Rate), units.strip(), str(s.Description).strip())
    return out


def start_utc(ch: dict[str, Channel], path: str | Path) -> dt.datetime:
    """Recording start from the DATE/GMT channels; falls back to the file name."""
    try:
        d = [int(ch[k].data[0]) for k in ("DATE_YEAR", "DATE_MONTH", "DATE_DAY", "GMT_HOUR", "GMT_MINUTE", "GMT_SEC")]
        return dt.datetime(*d)
    except (KeyError, ValueError, IndexError):
        return parse_filename(path)[1]


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
