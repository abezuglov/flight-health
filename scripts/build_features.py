"""Phase features and rule values for every full flight (fragments skipped)."""
import sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flighthealth.exceedances import rule_values  # noqa: E402
from flighthealth.features import feature_channels, flight_features  # noqa: E402
from flighthealth.pipeline import load_flight  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/dashlink")


def _one(args):
    fid, channels = args
    df = load_flight(ROOT, fid)
    return flight_features(df, channels), rule_values(df)


if __name__ == "__main__":
    t0 = time.time()
    flights = pd.read_parquet(ROOT / "catalog" / "flights.parquet")
    full = flights.loc[~flights.is_fragment, "flight_id"].tolist()
    channels = feature_channels(pd.read_parquet(ROOT / "catalog" / "parameters.parquet"))
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(_one, [(f, channels) for f in full], chunksize=4))
    (ROOT / "features").mkdir(exist_ok=True)
    feats = pd.DataFrame([r for f, _ in res for r in f])
    rules = pd.DataFrame([r for _, v in res for r in v])
    feats.to_parquet(ROOT / "features" / "phase_features.parquet", index=False)
    rules.to_parquet(ROOT / "features" / "rule_values.parquet", index=False)
    print(f"{len(full)} flights, {len(feats)} flight-phases, {feats.shape[1] - 3} features, "
          f"{len(rules)} rule values in {time.time() - t0:.0f}s")
    print(feats.phase.value_counts().to_dict())
