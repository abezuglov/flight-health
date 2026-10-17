"""Build the processed DASHlink tables: catalog, aligned 1 Hz, high-rate, quality flags, phases."""
import argparse, glob, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flighthealth.catalog import build_catalog, load_rules  # noqa: E402
from flighthealth.pipeline import process_flight  # noqa: E402


def _one(args):
    path, catalog, out = args
    r = process_flight(path, catalog)
    tail = r["flight"]["tail"]
    for name in ("aligned", "highrate"):
        d = out / ("aligned_1hz" if name == "aligned" else "highrate") / f"tail={tail}"
        d.mkdir(parents=True, exist_ok=True)
        r[name].to_parquet(d / f"{r['flight']['flight_id']}.parquet", index=False)
    return r["flight"], r["flags"], r["phases"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out_dir)
    files = sorted(glob.glob(f"{a.raw_dir}/*.mat"))[: a.limit]
    t0 = time.time()
    (out / "catalog").mkdir(parents=True, exist_ok=True)
    catalog = build_catalog(files, load_rules(Path(__file__).resolve().parents[1] / "config" / "param_rules.csv"))
    catalog.to_parquet(out / "catalog" / "parameters.parquet", index=False)
    print(f"catalog: {len(catalog)} channels ({time.time() - t0:.0f}s)")
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_one, [(f, catalog, out) for f in files], chunksize=4))
    pd.DataFrame([r[0] for r in res]).to_parquet(out / "catalog" / "flights.parquet", index=False)
    pd.concat([r[1] for r in res], ignore_index=True).to_parquet(out / "quality_flags.parquet", index=False)
    pd.concat([r[2] for r in res], ignore_index=True).to_parquet(out / "phases.parquet", index=False)
    print(f"{len(res)} flights in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
