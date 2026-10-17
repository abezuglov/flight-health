"""Flight Health: cleaning flight recorder data, finding engine anomalies, and checking them
against real maintenance events. Runs on the small bundle in app/data (scripts/build_app_data.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from charts import base_layout, chrome, end_labels, engine_lines, phase_bounds, slots

DATA = Path(__file__).parent / "data"
REPO = "https://github.com/abezuglov/flight-health"
FAMILIES = {"n1": ("N1 fan speed", "%"), "n2": ("N2 core speed", "%"), "egt": ("Exhaust gas temperature", "°C"),
            "ff": ("Fuel flow", "lb/h"), "oip": ("Oil pressure", "psi"), "oit": ("Oil temperature", "°C"),
            "vib": ("Vibration", "in/s"), "pla": ("Thrust lever angle", "°")}

st.set_page_config(page_title="Flight Health", page_icon="✈️", layout="wide")


@st.cache_data
def load(name: str) -> pd.DataFrame:
    p = DATA / name
    return pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})


# --------------------------------------------------------------------------------------------
def overview():
    st.title("Flight Health")
    st.markdown(
        "Cleaning many-channel flight recorder data, finding engine anomalies without labels, and "
        "checking the methods against real maintenance events. Two public datasets: "
        "**NASA DASHlink** (a four-engine regional jet, 637 recordings, 186 channels at 0.25–16 Hz) and "
        "the **NGAFID maintenance dataset** (11,446 Cessna 172 flights before or after 19 kinds of "
        f"unplanned repair). Code and write-ups: [{REPO.split('//')[1]}]({REPO})."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("EGT fault caught, engine vs siblings", "99%",
              help="20 °C EGT offset injected on one engine, 164 held-out flights, 5% false-alarm rate")
    c1.caption("Limit rules catch the same fault 6% of the time.")
    c2.metric("Maintenance events found without labels", "2")
    c2.caption("Engine 4 oil-pressure steps, each right after engine-4 ground run-ups.")
    c3.metric("Benchmark accuracy without the oil channels", "57.4%",
              help="NGAFID before/after-maintenance task; the published best is 76.0%")
    c3.caption("76.3% with them: most of the skill is a shop-visit oil-pressure step.")

    st.subheader("What the project found")
    st.markdown(
        "1. **Data cleaning is most of the work, and the data lies in specific ways.** Weight-on-wheels is inverted, "
        "ground speed reads 0 below 25 kt, 2.9% of vertical-acceleration samples are a no-data code, 82 recordings "
        "have corrupt dates. Checks run at each channel's native rate; bad samples are dropped before averaging "
        "and every one is logged. → *Flight explorer*\n"
        "2. **Compare each engine with its siblings.** A model over all 406 features drowns out a single-engine "
        "fault; one built on engine-vs-siblings deviations catches 75–99% of injected faults where limit rules "
        "catch 6–30%. → *Detector test*\n"
        "3. **Trends find maintenance.** Engine 4 oil pressure steps up 15 psi in April and back in July, each time "
        "right after overnight ground runs of engine 4 alone at ~95% N1. → *Engine trends*\n"
        "4. **A benchmark can measure the wrong thing.** On NGAFID, a classifier matches the published 76% accuracy, "
        "but most of its skill is a ~1.3 psi oil-pressure step at every shop visit, whatever the fault. Rough "
        "running and engine failures stay at chance. → *Maintenance validation*"
    )
    st.caption("This app runs on a 4.5 MB bundle: every flight-level table plus 16 hand-picked flights. "
               "Run the pipeline locally for all 637.")


# --------------------------------------------------------------------------------------------
def flight_explorer():
    st.title("Flight explorer")
    idx = load("demo_index.parquet")
    flights = load("flights.parquet").set_index("flight_id")
    label = {r.flight_id: f"{r.why}  ·  {r.flight_id}" for r in idx.itertuples()}
    fid = st.selectbox("Flight", idx.flight_id, format_func=lambda f: label[f])
    df = load("demo_flights.parquet")
    df = df[df.flight_id == fid].copy()
    df["minutes"] = df.t_s / 60
    info = flights.loc[fid]

    a, b, c, d = st.columns(4)
    start = pd.Timestamp(info.start_utc)
    a.metric("Date", start.strftime("%Y-%m-%d"))
    a.caption(f"Start {start:%H:%M} UTC, " + ("from the file name (date fields corrupt)"
                                              if info.start_time_source == "filename" else "from the date channels"))
    b.metric("Duration", f"{info.duration_s / 60:.0f} min")
    b.caption("Ground only" if info.n_legs == 0 else ("Fragment" if info.is_fragment else "Full flight"))
    c.metric("Max altitude", f"{info.max_alt_ft:,.0f} ft")
    flags = load("demo_flags.parquet")
    flags = flags[flags.flight_id == fid]
    rate = load("parameters.parquet").set_index("param").rate_hz
    bad = flags[(flags.severity == "bad") & flags.param.isin(rate.index)]
    n_bad = int((((bad.t_end_s - bad.t_start_s) * bad.param.map(rate)).round() + 1).sum())
    d.metric("Samples dropped as bad", f"{n_bad:,}")
    d.caption(f"{len(bad):,} bad-sample flags, {int((flags.severity == 'warn').sum())} warnings")

    phases = df.groupby((df.phase != df.phase.shift()).cumsum()).agg(phase=("phase", "first"), t_start_s=("t_s", "min"))
    fig = go.Figure(go.Scattergl(x=df.minutes, y=df.alt, mode="lines", name="Pressure altitude",
                                 line=dict(width=2, color=slots()[0]),
                                 hovertemplate="%{y:,.0f} ft<extra></extra>"))
    base_layout(fig, "Pressure altitude", 220, "ft", legend=False)
    phase_bounds(fig, phases)
    fig.update_layout(margin=dict(t=56))
    show(fig)

    fams = st.multiselect("Engine channels (one chart each, all four engines)", list(FAMILIES),
                          default=["pla", "n1", "egt", "oip"], format_func=lambda f: FAMILIES[f][0])
    cruise = df.phase == "cruise"
    for fam in fams:
        name, unit = FAMILIES[fam]
        fflags = flags[flags.param.str.match(rf"^{fam}_\d$") & (flags.severity == "bad")]
        show(engine_lines(df, fam, "minutes", unit, name, "minutes from recording start (shaded = cruise)",
                          shade=cruise, flags=fflags))
    st.caption("Ticks along the bottom mark samples dropped as bad; hover for the reason. "
               "Lines are 1 Hz values after cleaning.")

    st.subheader("How the detectors scored this flight")
    eng = load("engine_scores_cruise.parquet")
    e = eng[eng.flight_id == fid]
    if len(e):
        r = e.iloc[0]
        rank = int((eng.engine_combined > r.engine_combined).sum()) + 1
        st.markdown(f"**Engine vs siblings (cruise):** ranked **{rank} of {len(eng)}** flights, "
                    f"channel blamed: **{r.blamed_channel}** (sibling z = {r.sibling_z:.1f}; "
                    f"PCA error shares: {r.top_channels}).")
    else:
        st.markdown("No cruise phase, so no engine score.")
    g = load("anomaly_scores.parquet")
    g = g[g.flight_id == fid][["phase", "pca_pct", "forest_pct", "combined_pct", "top_channels"]]
    if len(g):
        st.dataframe(g.rename(columns={"pca_pct": "PCA percentile", "forest_pct": "forest percentile",
                                       "combined_pct": "combined", "top_channels": "channels driving the score"})
                     .round(1), hide_index=True, width="stretch")
    ex = load("exceedances.parquet")
    ex = ex[ex.flight_id == fid]
    if len(ex):
        st.markdown("**Limit-rule exceedances:** " + "; ".join(
            f"{r.description} = {r.value:.3g} (fleet threshold {r.threshold:.3g})" for r in ex.itertuples()))
    with st.expander("Quality flags for this flight"):
        st.dataframe(flags.groupby(["check", "severity"]).size().rename("flags").reset_index(), hide_index=True)
        st.dataframe(flags.head(500), hide_index=True, width="stretch")


# --------------------------------------------------------------------------------------------
def engine_trends():
    st.title("Engine trends")
    st.markdown("Each dot is one flight's cruise average of an engine's deviation from the median of all four; "
                "lines are rolling medians over 25 flights. Steps are found automatically; triangles mark nights "
                "with a ground run above 90% N1.")
    fam = st.radio("Channel", ["oip", "oit", "egt", "ff", "vib", "n1"], horizontal=True,
                   format_func=lambda f: FAMILIES[f][0])
    name, unit = FAMILIES[fam]
    cr = load("engine_cruise.parquet").sort_values("start_utc")
    colors = slots()[:4]
    fig = go.Figure()
    lasts = []
    for k, col in zip(range(1, 5), colors):
        s = cr.set_index("start_utc")[f"{fam}_dev_{k}__mean"]
        fig.add_trace(go.Scattergl(x=s.index, y=s, mode="markers", marker=dict(size=4, color=col, opacity=0.35),
                                   showlegend=False, hoverinfo="skip"))
        roll = s.rolling(25, center=True, min_periods=10).median()
        fig.add_trace(go.Scatter(x=roll.index, y=roll, mode="lines", name=f"Eng {k}", line=dict(width=2, color=col),
                                 hovertemplate=f"Eng {k}: %{{y:.2f}} {unit}<extra></extra>"))
        lasts.append(roll.dropna().iloc[-1])
    steps = load("engine_steps.parquet")
    steps = steps[steps.channel.str.startswith(fam + "_")]
    c = chrome()
    for r in steps.itertuples():
        fig.add_vline(x=pd.Timestamp(r.date), line_width=1, line_dash="dot", line_color=c["axis"])
        fig.add_annotation(x=pd.Timestamp(r.date), y=0.98, yref="paper", showarrow=False, yanchor="top",
                           xanchor="left", xshift=4, bgcolor=c["surface"],
                           text=f"step: {r.channel.replace('_', ' ')} {r.step:+.1f}", font=dict(size=10, color=c["ink2"]))
    runs = load("ground_runs.parquet")
    hi = runs[runs[[f"n1_max_{k}" for k in range(1, 5)]].max(axis=1) >= 90]
    hi = hi.merge(load("flights.parquet")[["flight_id", "start_utc"]], on="flight_id")
    hi["engines"] = hi.apply(lambda r: ", ".join(str(k) for k in range(1, 5) if r[f"n1_max_{k}"] >= 90), axis=1)
    fig.add_trace(go.Scatter(x=hi.start_utc, y=[0] * len(hi), yaxis="y2", mode="markers", name="run-up on the ground",
                             marker=dict(symbol="triangle-up", size=9, color=c["muted"]),
                             customdata=hi.engines, hovertemplate="ground run-up, engine(s) %{customdata}<extra></extra>"))
    fig.update_layout(yaxis2=dict(overlaying="y", range=[0, 20], visible=False, fixedrange=True))
    base_layout(fig, f"{name}: each engine minus the median of the four (cruise)", 420, unit)
    fig.update_layout(margin=dict(t=64))
    end_labels(fig, cr.start_utc.max(), lasts, [f"Eng {k}" for k in range(1, 5)], colors)
    show(fig)
    if len(steps):
        st.dataframe(steps[["channel", "date", "before", "after", "step", "step_z", "ground_runs_before",
                            "high_power_ground_run_engines", "same_engine_run_up"]].round(2),
                     hide_index=True, width="stretch")
    else:
        st.caption("No steps detected on this channel.")
    st.info("Both engine 4 oil-pressure steps come right after overnight ground runs of engine 4 at ~95% N1, "
            "the usual pattern of maintenance followed by a run-up check. Engine serial numbers never change. "
            "No maintenance log is available to confirm what was done.")


# --------------------------------------------------------------------------------------------
def detector_test():
    st.title("Detector test")
    st.markdown("No labels exist on DASHlink, so faults of known size were injected into the cruise phase of "
                "164 held-out flights (one engine each). Detection is counted at a **5% false-alarm rate**, "
                "set on the same flights without a fault. A size-0 control stays at 5%.")
    d = load("injection_detection.parquet")
    faults = {"egt_offset": "EGT offset (°C)", "egt_drift": "EGT drift over cruise (°C)",
              "n1_offset": "N1 offset (%)", "oip_drop": "Oil pressure drop (psi)",
              "oit_rise": "Oil temperature rise", "vib_offset": "Vibration rise (in/s)"}
    fault = st.radio("Injected fault", list(faults), horizontal=True, format_func=lambda f: faults[f])
    sub = d[d.fault == fault].sort_values("size")
    pos = list(range(len(sub)))  # evenly spaced positions; sizes double each step, so show them as labels
    detectors = [("rules", "Limit rules"), ("global_pca_max", "Whole-aircraft PCA (worst channel)"),
                 ("sibling_z", "Sibling z-score"), ("engine_combined", "Engine vs siblings (combined)")]
    fig = go.Figure()
    colors = slots()[:4]
    for (col, name), color in zip(detectors, colors):
        fig.add_trace(go.Scatter(x=pos, y=100 * sub[col], name=name, mode="lines+markers",
                                 line=dict(width=2, color=color), marker=dict(size=8, color=color,
                                 line=dict(width=2, color=chrome()["surface"])),
                                 hovertemplate=f"{name}: %{{y:.0f}}%<extra></extra>"))
    fig.add_hline(y=5, line_width=1, line_dash="dot", line_color=chrome()["axis"],
                  annotation_text="5% false-alarm rate", annotation_position="bottom left",
                  annotation_font=dict(size=10, color=chrome()["muted"]))
    base_layout(fig, f"Share of faulty flights detected: {faults[fault]}", 380, "% detected", "fault size")
    fig.update_yaxes(range=[0, 105])
    fig.update_xaxes(tickvals=pos, ticktext=[f"{v:g}" for v in sub["size"]], range=[-0.3, len(pos) - 0.4])
    end_labels(fig, pos[-1], [100 * sub[c].iloc[-1] for c, _ in detectors],
               [n for _, n in detectors], colors)
    show(fig)
    with st.expander("All detectors, all faults"):
        st.dataframe(d.round(2), hide_index=True, width="stretch")
    st.markdown("A score over all 406 features drowns out one engine's change. Comparing each engine with the "
                "other three works: the plain z-score catches N1 faults (N1 is held very tight across engines), "
                "and PCA catches EGT faults (it learns how EGT normally varies between engines). Isolation forest "
                "missed every fault and is left out of the chart.")


# --------------------------------------------------------------------------------------------
def maintenance_validation():
    st.title("Maintenance validation (NGAFID)")
    st.markdown("Task: tell Cessna 172 flights **before** an unplanned repair from flights **after** it. "
                "11,446 flights, 19 fault types, the dataset authors' 5 folds "
                "([Yang & Desell 2022](https://arxiv.org/abs/2210.07317), CC-BY-4.0).")
    cv = load("ngafid_supervised_cv.parquet")
    names = {"all": "All features", "engine": "Engine channels", "oil_only": "Oil pressure + temperature only",
             "all_except_oil": "Everything except oil", "cylinders_vs_siblings": "Cylinders vs siblings",
             "operational": "Operational (speed, altitude, fuel, OAT)", "all (shuffled labels)": "Shuffled labels (control)"}
    cv = cv.assign(name=cv.features.map(names)).sort_values("accuracy")
    c = chrome()
    fig = go.Figure(go.Bar(x=100 * cv.accuracy, y=cv.name, orientation="h", marker=dict(color=slots()[0]),
                           text=[f"{v:.1%}" for v in cv.accuracy], textposition="outside",
                           textfont=dict(color=c["ink2"]),
                           hovertemplate="%{y}: %{x:.1f}% accuracy<extra></extra>"))
    for x, txt in ((50, "chance"), (76.0, "paper's best (76.0%)")):
        fig.add_vline(x=x, line_width=1, line_dash="dot", line_color=c["axis"])
        fig.add_annotation(x=x, y=1, yref="paper", text=txt, showarrow=False, yanchor="bottom",
                           font=dict(size=10, color=c["muted"]))
    base_layout(fig, "Gradient boosting accuracy by feature set", 330, legend=False, x_title="accuracy (%)")
    fig.update_layout(hovermode="closest", margin=dict(t=56))
    fig.update_xaxes(range=[40, 85], showspikes=False)
    show(fig)

    oil = load("ngafid_oil_step.parquet")
    order = ["baffle", "intake", "rocker cover", "cylinder", "engine / oil", "other"]
    labels = {-2: "2 days before", -1: "1 day before", 1: "1 day after", 2: "2 days after"}
    fig = go.Figure()
    for fam_name, color in zip(order, slots()[:6]):
        s = oil[oil.family == fam_name].sort_values("date_diff")
        se = 1.96 * s["std"] / np.sqrt(s["count"])
        fig.add_trace(go.Scatter(x=s.date_diff.map(labels), y=s["mean"], name=f"{fam_name} (n={int(s['count'].sum())})",
                                 mode="lines+markers", line=dict(width=2, color=color),
                                 marker=dict(size=8, color=color, line=dict(width=2, color=c["surface"])),
                                 error_y=dict(type="data", array=se, thickness=1, width=3, color=color),
                                 hovertemplate=f"{fam_name}: %{{y:.2f}} psi<extra></extra>"))
    # six series: legend only (direct end labels are for four or fewer)
    base_layout(fig, "Mean cruise oil pressure around the maintenance date (95% CI)", 440, "psi")
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0))
    show(fig)

    a = load("ngafid_auc_vs_oil_step.parquet")
    fig = go.Figure(go.Scatter(x=a.oil_step_psi, y=a["all"], mode="markers", name="fault type",
                               marker=dict(size=np.clip(np.sqrt(a.n) / 2, 8, 30), color=slots()[0], opacity=0.85,
                                           line=dict(width=2, color=c["surface"])),
                               customdata=np.stack([a.label, a.n], axis=1),
                               hovertemplate="%{customdata[0]}<br>oil step %{x:+.2f} psi · AUC %{y:.2f} · n=%{customdata[1]}<extra></extra>"))
    callouts = {"engine run rough": (40, -30), "intake gasket leak/damage": (-70, 34),
                "engine failure/fire/time out": (50, 28), "baffle seal loose/damage": (-10, 46)}
    for lab, (ax, ay) in callouts.items():
        r = a[a.label == lab].iloc[0]
        fig.add_annotation(x=r.oil_step_psi, y=r["all"], text=lab, showarrow=True, arrowwidth=1, arrowcolor=c["axis"],
                           ax=ax, ay=ay, font=dict(size=10, color=c["ink2"]), bgcolor=c["surface"])
    fig.add_hline(y=0.5, line_width=1, line_dash="dot", line_color=c["axis"], annotation_text="chance",
                  annotation_font=dict(size=10, color=c["muted"]))
    base_layout(fig, "Per fault type: classifier AUC tracks the oil-pressure step (Spearman 0.56)", 400,
                "AUC, before vs after", "oil pressure after minus before (psi)", legend=False)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(showspikes=False)
    show(fig)

    st.markdown("Shuffling oil pressure alone drops the classifier's AUC from 0.85 to about 0.54. The step "
                "appears for cooling-baffle parts that have nothing to do with oil, so the most likely cause is "
                "an oil change during the shop visit (the maintenance text is withheld, so this can't be confirmed). "
                "Rough running, engine failure and overspeed, the faults that matter most, show no step and stay "
                "at chance. Unsupervised scores are at chance for every fault type.")
    with st.expander("Tables"):
        st.dataframe(load("ngafid_supervised_cv.parquet").round(3), hide_index=True)
        st.dataframe(a.round(3), hide_index=True, width="stretch")
        st.dataframe(load("ngafid_channel_importance.parquet").round(4).head(12), hide_index=True)


pg = st.navigation([
    st.Page(overview, title="Overview", default=True),
    st.Page(flight_explorer, title="Flight explorer"),
    st.Page(engine_trends, title="Engine trends"),
    st.Page(detector_test, title="Detector test"),
    st.Page(maintenance_validation, title="Maintenance validation"),
])
pg.run()
