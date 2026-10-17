# Flight Health

Cleaning many-channel flight recorder data and detecting anomalies, then testing whether those anomalies show up in the flights before real maintenance events.

**Status:** done: data pipeline and anomaly detection on NASA DASHlink tail 670, validation against real maintenance events on NGAFID, and an interactive Streamlit app.

## Data

| Dataset | What it is | Use here |
|---|---|---|
| [NASA DASHlink flight data](https://catalog.data.gov/dataset/flight-data-for-tail-670), tail 670 set 1 | 637 recordings, 821 hours, 186 channels at 0.25–16 Hz, 2001. Four-engine regional jet (most likely BAe 146 / Avro RJ; NASA doesn't name the type). | Pipeline and anomaly detection |
| [NGAFID maintenance dataset](https://zenodo.org/record/6624956) (CC-BY-4.0) | 11,446 general-aviation flights, 23 channels at 1 Hz, labeled before/after 2,111 unplanned maintenance events | Validation (Phase 3) |

Raw files go in `data/raw/` and are never modified.

## Pipeline

```
python scripts/build_dashlink.py data/raw/dashlink/tail670 data/processed/dashlink
python scripts/quality_report.py data/processed/dashlink reports
python -m pytest tests
```

Outputs under `data/processed/dashlink/`:

| Table | Grain | Notes |
|---|---|---|
| `catalog/parameters.parquet` | one row per channel | kind (continuous / discrete / metadata / constant / counter), group, engine index, valid range, no-data codes, observed ranges, notes |
| `catalog/flights.parquet` | one row per recording | start time and its source, duration, takeoff/landing seen, fragment flag, max altitude, engine hours/cycles, flag counts, source SHA-256 |
| `aligned_1hz/tail=670/<flight_id>.parquet` | one row per second | every continuous and discrete channel; fast channels as the mean of the second (plus min/max for accelerations and attitude); derived `phase` and the aircraft's own `ph_acms` |
| `highrate/tail=670/<flight_id>.parquet` | 16 rows per second | the eight 8/16 Hz channels at native rate, uncleaned |
| `quality_flags.parquet` | one row per problem | check, severity, channel, time span, detail |
| `phases.parquet` | one row per phase segment | parked, taxi_out, takeoff, climb, cruise, descent, approach, landing, taxi_in |

Every cleaning step and the reasoning behind it: [docs/cleaning_decisions.md](docs/cleaning_decisions.md).

**Alignment.** Row `t` summarizes the window (t−1, t] seconds, so nothing in a row was recorded after time `t`. Channels slower than 1 Hz hold their last value.

**Cleaning.** Checks run on each channel at its native rate. Samples flagged `bad` are left out of the 1 Hz values (a second goes missing only if all its samples were bad). Every dropped sample is listed in `quality_flags`; original values stay in the raw files and the `highrate` table. Rules live in [config/param_rules.csv](config/param_rules.csv).

| Check | Severity | What it catches |
|---|---|---|
| sentinel | bad | no-data codes: VRTG −3.375 g, LATG/LONG −1.083 g, oil temp −390, position 0 |
| out_of_range | bad | values outside physical limits (e.g. CAS 511.9 kt saturation, oil pressure 9,999) |
| dropout | bad | engine channels reading 0 for up to 20 s in flight between normal values |
| spike | bad | single-sample out-and-back jumps larger than 10× the channel's typical step |
| stuck | warn | engine and air-temperature channels frozen for 5+ minutes in flight |
| frame_gap | warn | frame counter skipping (none found after handling its wrap-around) |
| cross_check | warn | redundant sensors disagreeing for 10+ s (baro altitudes, AOA vanes, altitude rate vs inertial vertical speed) |

## Things this data gets wrong that you'd otherwise trust

- `WOW` (weight on wheels) is inverted: 1 = airborne.
- Ground speed reads 0 below about 25 kt, so taxi is detected from engines running, not speed.
- 2.9% of vertical-acceleration samples are a −3.375 g no-data code scattered through every flight.
- 82 of 637 recordings have corrupt date fields (year up to 2165); start time falls back to the file name.
- 89 recordings never leave the ground (likely engine runs or aborted recordings).
- Some files carry a `_1` suffix: a second recording starting in the same minute.

## Phase 1 results (tail 670)

- 0.14% of all samples dropped as bad; 81 of 103 continuous channels had at least one bad sample.
- Worst channel: VRTG, 2.9% of samples dropped but only 0.16% of seconds left without a value.
- Derived phases agree with the aircraft's own phase code on 88% of seconds when compared on the phases both can express. Most of the gap is definitional: the aircraft calls any level flight "cruise", while this pipeline calls level-offs below the top altitude part of climb or descent.

Full numbers: [reports/quality_summary.json](reports/quality_summary.json), [reports/dropped_by_param.csv](reports/dropped_by_param.csv), [reports/phase_vs_acms.csv](reports/phase_vs_acms.csv).

## Phase 2 results: anomaly detection

Full write-up: [docs/phase2_findings.md](docs/phase2_findings.md).

- **Comparing each engine with its siblings works; a whole-aircraft score doesn't.** On faults injected into held-out flights, at a 5% false-alarm rate, the engine detector catches a 20 °C EGT offset 99% of the time (limit rules: 6%), a 0.25% N1 offset 98% (rules: 6%), a 5 psi oil-pressure drop 75% (rules: 12%) and a 0.1 in/s vibration rise 80% (rules: 30%), and names the right engine channel for most of them. A PCA over all 406 features misses most of these; isolation forest misses all of them.
- **Engine trend monitoring found two maintenance-like events without labels.** Engine 4 oil pressure steps up 15 psi on 16 April and back down 11 psi on 11 July. Each step comes right after overnight ground runs of engine 4 at about 95% N1, the usual pattern of maintenance followed by a run-up check. No maintenance log is available to confirm.
- **The top-scored real flights are mostly operational, not faults:** a crew throttling one engine back in cruise, speed brakes used in climb. A useful reminder that "unusual" and "broken" are different questions.

```
python scripts/build_features.py      # phase features + rule values
python scripts/run_detection.py       # rules, global models, cross-fitted engine scores
python scripts/eval_injection.py      # fault-injection evaluation
python scripts/run_trends.py          # engine step changes + ground runs before them
python scripts/plot_trends.py reports/phase2/figures/engine_trends.png
```

## Phase 3 results: validation against real maintenance (NGAFID)

Full write-up: [docs/phase3_findings.md](docs/phase3_findings.md).

- **A gradient-boosted classifier on hand-built features matches the published deep-learning benchmark** for telling flights before maintenance from flights after: 76.3% accuracy vs 76.0% (ConvMHSA), on the authors' folds, with a shuffled-label control at 49.7%.
- **Most of that accuracy is a shop-visit effect, not the faults.** Cruise oil pressure is ~1.3 psi higher right after maintenance for almost every fault type, including cooling-baffle parts unrelated to oil. Oil channels alone give 72%; everything except oil gives 57%. Per fault type, accuracy tracks the size of the oil step (Spearman 0.56, p = 0.013).
- **The faults that matter most are at chance:** engine run rough, engine failure, overspeed. Unsupervised scores are at chance for every fault type, as the dataset authors predicted.
- **The small real signal sits where physics says it should:** cylinder-vs-siblings features do best on cylinder cracks, compression problems and rough running (AUC 0.61–0.65).

```
python scripts/build_ngafid.py data/raw/ngafid/2days data/processed/ngafid
python scripts/eval_ngafid.py         # unsupervised + supervised, per fault type
python scripts/explain_ngafid.py      # permutation importance by channel
python scripts/plot_oil_step.py
```

## App

```
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

Five pages: an overview, a **flight explorer** (all four engines per channel, phases, every dropped sample with its reason, how each detector scored the flight), **engine trends** (the engine 4 oil-pressure steps and the ground runs before them), the **detector test** (fault-injection results) and **maintenance validation** (the NGAFID oil-pressure finding).

The app runs on a 4.3 MB bundle in `app/data/` (every flight-level table plus 16 hand-picked flights), so it needs none of the raw data. Rebuild it with `python scripts/build_app_data.py` after rerunning the pipeline. To deploy on Streamlit Community Cloud, point it at `app/streamlit_app.py`.

## Safety note

The NGAFID data ships as a Python pickle. It was scanned with `pickletools` first (it only builds numpy arrays) and is loaded only through [scripts/safe_pickle.py](scripts/safe_pickle.py), which refuses any other type.

## Setup

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # .venv/bin/pip on macOS/Linux
```

Download [Tail_670_1.zip](https://c3.nasa.gov/dashlink/static/media/dataset/Tail_670_1.zip) and unzip into `data/raw/dashlink/tail670/`; download `2days.tar.gz` from the [Zenodo record](https://zenodo.org/record/6624956) and extract into `data/raw/ngafid/`.
