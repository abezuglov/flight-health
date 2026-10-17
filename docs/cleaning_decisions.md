# Cleaning steps and decisions

A record of how the raw flight data is turned into analysis tables, and why. Each entry says what was found, what was decided, and what it costs. Later phases should read this before trusting any column.

Scope so far: NASA DASHlink tail 670, set 1 (637 recordings, 821 flight hours, 186 channels). NGAFID findings are listed separately at the end; that dataset is not processed yet.

## 1. Inputs are never modified

- Raw `.mat` files stay as downloaded in `data/raw/`. Each processed flight records the source file's SHA-256 and the pipeline version.
- The `highrate` table keeps the eight 8/16 Hz channels at native rate with **no** cleaning, so any cleaning decision can be re-examined against the original samples.

## 2. Channel catalog

Every channel is classified once, from its own metadata plus a pass over all flights (`catalog/parameters.parquet`).

| Kind | Count | Rule | Treatment |
|---|---|---|---|
| continuous | 103 | has units, or listed as continuous by hand (`DWPT`) | checked and averaged |
| discrete | 50 | no units (flags, modes, switch states) | not checked; 1 Hz value = max over the second, so a short "on" is never lost |
| metadata | 26 | sync words, serial numbers, engine hours/cycles, date/time | kept out of the 1 Hz table; engine hours, cycles and serials go to the flights table |
| constant | 6 | never changes in any recording (e.g. smoke warning, tank 3 fuel) | dropped from modeling |
| counter | 1 | frame counter `FRMC` | used only to check for lost frames |

Valid ranges and no-data codes are set by hand in `config/param_rules.csv`, from physical limits and from the observed value distributions, not fitted to the data.

## 3. Time alignment

- **Decision:** one main table at 1 Hz. Row `t` summarizes the window (t−1, t] seconds.
- **Why:** most channels are 1–4 Hz, and anomaly windows of interest are seconds to minutes long. A 4 Hz base would quadruple the table, mostly by repeating slow channels.
- **No look-ahead:** nothing in row `t` was recorded after time `t`. Channels slower than 1 Hz hold their last value. A test sends a single-sample impulse through every rate and checks it lands in exactly the right row.
- Fast channels keep the mean of the second; accelerations, attitude and the other ≥8 Hz channels also keep `_min` and `_max` so short peaks survive.
- **Cost:** 1 Hz averages blur sub-second events such as touchdown loads. Use the `highrate` table for those.

## 4. Quality checks

Checks run on each channel **at its native rate**, before averaging, so one bad sample can't hide inside a 1-second mean.

| Check | Severity | Rule | Flags (tail 670) |
|---|---|---|---|
| sentinel | bad | exact no-data codes: VRTG −3.375 g, LATG/LONG −1.083 g, oil temp −390, latitude/longitude 0 | 938,231 |
| spike | bad | one sample jumps away and straight back, by more than 10× the channel's 95th-percentile step | 320,321 |
| dropout | bad | engine channel (N1, N2, EGT, fuel flow, oil pressure) reads exactly 0 for ≤20 s in flight between normal values | 23,449 |
| out_of_range | bad | outside physical limits, e.g. CAS 511.9 kt (saturation), oil pressure 9,999 | 487 |
| cross_check | warn | redundant sensors disagree for 10+ s: baro altitude 1 vs 2 (>200 ft), pressure vs baro altitude (>2,500 ft), AOA vanes (>5°), altitude rate vs inertial vertical speed (>1,500 ft/min) | 86 |
| stuck | warn | engine or air-temperature channel exactly constant for 5+ minutes in flight | 36 |
| frame_gap | warn | frame counter skips a step | 0 |

Decisions behind the thresholds:

- **Spike threshold uses the 95th percentile of step size, not the 99th.** With the 99th, a channel with many spikes raises its own threshold and hides them. Heavily quantized channels whose 95th-percentile step is 0 fall back to their smallest real step.
- **Spikes are treated as bad, not just suspicious.** Spot checks found them to be single samples dropping to 0 or to a channel's floor value (rudder −41.9°, spoiler 36.54°) and returning. Real motion doesn't do that in one sample.
- **Stuck is only checked on engine and air-temperature channels.** Altitude, selected values and similar legitimately hold constant for long periods.
- **Cross-check failures are warnings.** Without a third sensor we can't tell which of the two is wrong.

## 5. What "clean" means in the 1 Hz table

- **Decision:** samples flagged `bad` are left out of each second's mean/min/max. A second becomes missing only if *every* sample in it was bad.
- **What it replaced:** the original plan kept raw values and blanked any second containing a bad sample when the data was loaded. For VRTG (2.9% bad samples at 8 Hz) that lost **13% of seconds**, mostly good data. Dropping samples instead loses **0.16%**.
- **Traceability:** every dropped sample is a row in `quality_flags.parquet` (channel, time span, check, original value in `detail`). Overall 0.14% of samples were dropped.
- **Not done:** no interpolation or filling. Missing stays missing; filling is a modeling decision, made per model.

## 6. Data problems that would mislead a reader

| Problem | Evidence | Decision |
|---|---|---|
| `WOW` (weight on wheels) is inverted: 1 = airborne | WOW=1 rows average 328 kt ground speed and 4,787 ft radio altitude | Treat 1 as airborne; noted in the catalog |
| Ground speed reads 0 below ~25 kt | Heading turns and position moves during taxi while GS stays 0 | Taxi = on ground with any engine N2 > 40% |
| Corrupt date fields in 82 recordings | Year up to 2165, month up to 25, day up to 45 | Use the DATE/GMT channels only when in range; otherwise the start time from the file name (`start_time_source` records which) |
| Frame counter wraps X → 0 → 3, not 0 → 4 | All 188 apparent gaps had this exact pattern | Treat as a wrap; zero real frame gaps remain |
| `ALT` stored as an unsigned integer in some files | Values still go negative (min −271 ft) | Storage detail only; no correction needed |
| 89 recordings never become airborne | No WOW=1 segment of 60+ s | Kept and marked `is_fragment`; possibly engine ground runs, to look at in Phase 2 |
| Seven files with a `_1` suffix | Second recording starting in the same minute | Accepted by the file-name parser; `flight_id` keeps the suffix |
| Aircraft type not given | Four engines (N1_1 to N1_4) | Most likely BAe 146 / Avro RJ; stated as an inference, not a fact |

## 7. Flight phases

- Rule-based, from weight on wheels, ground speed, pressure altitude and engines running (rules in `src/flighthealth/phases.py`): parked, taxi_out, takeoff, climb, cruise, descent, approach, landing, taxi_in.
- **Cruise** = from first to last time within 1,000 ft of the flight's highest altitude. Level-offs below that count as climb or descent.
- **Validation:** compared with the aircraft's own phase code (`PH`), whose meanings were inferred from how each code lines up with the derived phases. Agreement is 88% of seconds on the phases both can express (74% exact). Most disagreement is definitional: the aircraft calls any level flight cruise, so 24% of the pipeline's descent seconds are "cruise" to the aircraft.
- **Open option:** add a separate "level" phase if Phase 2 needs it.

## 8. NGAFID (not processed yet)

- Data ships as a Python pickle. It was scanned with `pickletools` without loading (only numpy arrays) and is read only through `scripts/safe_pickle.py`, which refuses any other type.
- Stored as float16: altitude is rounded to 8 ft steps near 10,000 ft.
- Impossible values to handle: cylinder-head and exhaust temps near −300 °F, vertical speed ±13,000–17,000 ft/min, altitude −2,882 ft.
- volt2, amp2 and CHT2–4 missing in 4.7% of rows, probably aircraft with fewer sensors.
- Fuel quantity stuck at 54 for long stretches, above a 172's capacity.
- No aircraft IDs in the download: use the authors' `fold` column for cross-validation instead of splitting by aircraft.
