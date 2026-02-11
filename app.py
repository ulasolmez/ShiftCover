"""
ShiftCover – Streamlit front-end
================================
Supports 1-3 occupation workload curves with shared shift structures.
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from solver import (
    INTERVALS_PER_DAY, INTERVALS_PER_HOUR, TOTAL_INTERVALS,
    DAY_NAMES, OCC_COLORS,
    SolverParams, MultiCurveResult,
    solve_multi,
    coverage_dataframe, shifts_to_dataframe, schedules_to_dataframe,
    shift_type_summary, build_weekly_report_xlsx,
)
from sample_data import generate_sample_demand

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="ShiftCover", layout="wide")
st.title("🕐 ShiftCover – Weekly Shift Optimiser")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")

    n_curves = st.number_input("Number of occupation curves", 1, 3, 1)
    occ_names = []
    for i in range(n_curves):
        default_name = ["Technician", "Labourer", "Helper"][i]
        name = st.text_input(f"Occupation {i+1} name",
                             value=default_name, key=f"occ_name_{i}")
        occ_names.append(name)

    st.subheader("Shift settings")
    min_shift = st.slider("Min shift (h)", 3.0, 8.0, 3.0, 0.5)
    max_shift = st.slider("Max shift (h)", 6.0, 12.0, 12.0, 0.5)
    max_unique = st.number_input("Max unique shifts (0 = unlimited)", 0, 200, 0)

    st.subheader("Granularity")
    start_step = st.selectbox("Shift-start step (min)",
                              [5, 10, 15, 30, 60], index=2)
    dur_step = st.selectbox("Duration step (min)", [15, 30, 60], index=1)

    st.subheader("Weekly hours")
    min_wh = st.number_input("Min weekly hours", 20.0, 60.0, 40.0, 1.0)
    max_wh = st.number_input("Max weekly hours", 20.0, 60.0, 50.0, 1.0)

    st.subheader("Worker constraints")
    min_rest = st.number_input("Min rest between shifts (h)", 8.0, 24.0, 12.0, 1.0)

    st.subheader("Solver")
    time_limit = st.number_input("Time limit (s)", 10, 600, 120, 10)
    t_penalty = st.number_input("Transition penalty", 0, 500, 50, 10)

params = SolverParams(
    min_shift_hours=min_shift,
    max_shift_hours=max_shift,
    shift_start_granularity_min=start_step,
    shift_duration_step_min=dur_step,
    min_weekly_hours=min_wh,
    max_weekly_hours=max_wh,
    min_rest_hours=min_rest,
    max_unique_shifts=max_unique,
    transition_penalty=t_penalty,
    solver_time_limit_sec=time_limit,
)


# ── Helper: parse uploaded demand file ───────────────────────────────────────
def _parse_demand_file(uploaded, required_len=TOTAL_INTERVALS):
    """Return np.ndarray of shape (required_len,) from an uploaded CSV/XLSX."""
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    # Find the numeric demand column
    for col in ("Required", "required", "Demand", "demand"):
        if col in df.columns:
            arr = df[col].values
            break
    else:
        # pick first numeric column
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) == 0:
            raise ValueError("No numeric column found")
        arr = df[num_cols[0]].values

    if len(arr) < required_len:
        raise ValueError(f"Need {required_len} rows, got {len(arr)}")
    return np.round(arr[:required_len]).astype(int)


# ── Demand input ─────────────────────────────────────────────────────────────
st.header("📂 Demand input")

tab_upload, tab_sample = st.tabs(["⬆ Upload files", "🎲 Generate sample"])

with tab_upload:
    st.info(f"Upload **{n_curves} file(s)** — one per occupation. "
            "Each must have {TOTAL_INTERVALS} rows with a numeric column "
            "(e.g. 'Required').")
    uploaded_files = []
    for i in range(n_curves):
        f = st.file_uploader(
            f"{occ_names[i]} demand (CSV / XLSX)",
            type=["csv", "xlsx"],
            key=f"upload_{i}",
        )
        uploaded_files.append(f)

    if st.button("Load uploaded files", key="btn_load"):
        try:
            demands = []
            for i, f in enumerate(uploaded_files):
                if f is None:
                    st.error(f"Missing file for {occ_names[i]}")
                    st.stop()
                demands.append(_parse_demand_file(f))
            st.session_state["demands"] = demands
            st.session_state["occ_names"] = occ_names[:]
            st.success(f"Loaded {len(demands)} curve(s)")
        except Exception as exc:
            st.error(str(exc))

with tab_sample:
    cols_sample = st.columns(n_curves)
    peaks = []
    bases = []
    for i, col in enumerate(cols_sample):
        with col:
            st.markdown(f"**{occ_names[i]}**")
            pk = st.slider(f"Peak agents", 5, 60, max(5, 25 - i * 8),
                           key=f"peak_{i}")
            bs = st.slider(f"Base agents", 0, 10, max(1, 3 - i),
                           key=f"base_{i}")
            peaks.append(pk)
            bases.append(bs)
    sample_seed = st.number_input("Random seed", 0, 9999, 42)

    if st.button("🎲 Generate sample", key="btn_sample"):
        demands = []
        for i in range(n_curves):
            d = generate_sample_demand(
                peak_agents=peaks[i],
                base_agents=bases[i],
                seed=sample_seed + i * 7,
            )
            demands.append(d)
        st.session_state["demands"] = demands
        st.session_state["occ_names"] = occ_names[:]
        st.success(f"Generated {n_curves} sample curve(s)")

# ── Preview demand chart ─────────────────────────────────────────────────────
demands = st.session_state.get("demands")
stored_names = st.session_state.get("occ_names", occ_names)

if demands is not None:
    # ensure curve count matches current n_curves
    if len(demands) != n_curves:
        st.warning("Number of curves changed — please regenerate / re-upload.")
        demands = None

if demands is not None:
    st.subheader("Demand preview")
    x_vals = list(range(TOTAL_INTERVALS))
    fig_d = go.Figure()
    for i, (d, name) in enumerate(zip(demands, stored_names)):
        color = OCC_COLORS[i % len(OCC_COLORS)]
        fig_d.add_trace(go.Scatter(
            x=x_vals, y=d, mode="lines", name=name,
            fill="tozeroy",
            line=dict(color=color, width=1),
            opacity=0.55,
        ))
    # add total if >1 curve
    if len(demands) > 1:
        total_d = sum(demands)
        fig_d.add_trace(go.Scatter(
            x=x_vals, y=total_d, mode="lines", name="Total",
            line=dict(color="black", width=2, dash="dot"),
        ))
    # day separators
    for d in range(1, 7):
        fig_d.add_vline(x=d * INTERVALS_PER_DAY, line_dash="dash",
                        line_color="grey", opacity=0.4)
    tick_vals = [d * INTERVALS_PER_DAY for d in range(7)]
    fig_d.update_xaxes(tickvals=tick_vals, ticktext=DAY_NAMES)
    fig_d.update_layout(height=300, margin=dict(l=40, r=20, t=30, b=30),
                        legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig_d, use_container_width=True)


# ── Solve ────────────────────────────────────────────────────────────────────
if demands is not None:
    if st.button("▶ Run Optimiser", type="primary"):
        log_box = st.empty()
        log_lines: list[str] = []

        def _cb(msg):
            log_lines.append(msg)
            log_box.code("\n".join(log_lines), language="text")

        with st.spinner("Solving …"):
            result = solve_multi(demands, stored_names, params, callback=_cb)

        st.session_state["result"] = result
        st.session_state["result_params"] = params


# ── Results ──────────────────────────────────────────────────────────────────
result: MultiCurveResult | None = st.session_state.get("result")

if result is not None:
    st.divider()
    st.header("📊 Results")

    cp1 = result.combined_phase1
    n_occ = len(result.occupations)

    # ── Phase 1 summary metrics ──────────────────────────────────────────
    st.subheader("Phase 1 – Set Covering")
    total_headcount = sum(o.phase2.num_workers for o in result.occupations)
    peak_sim = int(cp1.coverage.max())
    total_wh = cp1.total_worker_intervals / INTERVALS_PER_HOUR
    n_active = sum(1 for c in cp1.counts if c > 0)

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Status", cp1.status)
    c2.metric("Active shifts", n_active)
    c3.metric("Total worker-h", f"{total_wh:,.0f}")
    c4.metric("Headcount", total_headcount)
    c5.metric("Peak Simultaneous", peak_sim)
    c6.metric("FTE (÷45)", f"{total_wh / 45:.1f}")
    c7.metric("Solve time", f"{cp1.elapsed_sec:.1f}s")

    # per-occupation headcount
    if n_occ > 1:
        occ_cols = st.columns(n_occ)
        for i, occ in enumerate(result.occupations):
            with occ_cols[i]:
                occ_wh = occ.phase1.total_worker_intervals / INTERVALS_PER_HOUR
                st.metric(f"{occ.name} headcount", occ.phase2.num_workers)
                st.metric(f"{occ.name} worker-h", f"{occ_wh:,.0f}")

    # ── Weekly coverage chart (combined + sub-curves) ────────────────────
    st.subheader("Weekly coverage")
    x_vals = list(range(TOTAL_INTERVALS))
    fig_w = go.Figure()

    # total demand + coverage (thick lines)
    fig_w.add_trace(go.Scatter(
        x=x_vals, y=result.combined_demand, mode="lines",
        name="Total demand",
        line=dict(color="red", width=2),
        fill="tozeroy", fillcolor="rgba(255,0,0,0.08)",
    ))
    fig_w.add_trace(go.Scatter(
        x=x_vals, y=cp1.coverage, mode="lines",
        name="Total coverage",
        line=dict(color="green", width=2),
        fill="tozeroy", fillcolor="rgba(0,128,0,0.08)",
    ))

    # per-occupation sub-curves (thin, dashed)
    if n_occ > 1:
        for i, occ in enumerate(result.occupations):
            clr = OCC_COLORS[i % len(OCC_COLORS)]
            fig_w.add_trace(go.Scatter(
                x=x_vals, y=occ.demand, mode="lines",
                name=f"{occ.name} demand",
                line=dict(color=clr, width=1, dash="dash"),
                visible="legendonly",
            ))
            fig_w.add_trace(go.Scatter(
                x=x_vals, y=occ.phase1.coverage, mode="lines",
                name=f"{occ.name} coverage",
                line=dict(color=clr, width=1),
                visible="legendonly",
            ))

    for d in range(1, 7):
        fig_w.add_vline(x=d * INTERVALS_PER_DAY, line_dash="dash",
                        line_color="grey", opacity=0.4)
    tick_vals = [d * INTERVALS_PER_DAY for d in range(7)]
    fig_w.update_xaxes(tickvals=tick_vals, ticktext=DAY_NAMES)
    fig_w.update_layout(height=400, margin=dict(l=40, r=20, t=30, b=30),
                        legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig_w, use_container_width=True)

    # ── Daily coverage chart (dropdown) ──────────────────────────────────
    st.subheader("Daily coverage")
    sel_day = st.selectbox("Day", DAY_NAMES, key="sel_day")
    day_idx = DAY_NAMES.index(sel_day)
    day_start = day_idx * INTERVALS_PER_DAY
    day_end = day_start + INTERVALS_PER_DAY
    day_x = list(range(INTERVALS_PER_DAY))

    # per-occupation day metrics
    if n_occ > 1:
        day_met_cols = st.columns(n_occ + 1)
        for i, occ in enumerate(result.occupations):
            sl = slice(day_start, day_end)
            with day_met_cols[i]:
                d_dem = occ.demand[sl]
                d_cov = occ.phase1.coverage[sl]
                st.metric(f"{occ.name} peak demand", int(d_dem.max()))
                st.metric(f"{occ.name} peak coverage", int(d_cov.max()))
        with day_met_cols[-1]:
            td = result.combined_demand[day_start:day_end]
            tc = cp1.coverage[day_start:day_end]
            st.metric("Total peak demand", int(td.max()))
            st.metric("Total peak coverage", int(tc.max()))
    else:
        occ = result.occupations[0]
        sl = slice(day_start, day_end)
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        d_dem = occ.demand[sl]
        d_cov = occ.phase1.coverage[sl]
        mc1.metric("Peak demand", int(d_dem.max()))
        mc2.metric("Peak coverage", int(d_cov.max()))
        mc3.metric("Avg demand", f"{d_dem.mean():.1f}")
        mc4.metric("Avg coverage", f"{d_cov.mean():.1f}")
        surplus = d_cov.astype(int) - d_dem.astype(int)
        mc5.metric("Max surplus", int(surplus.max()))

    fig_day = go.Figure()
    # total
    td = result.combined_demand[day_start:day_end]
    tc = cp1.coverage[day_start:day_end]
    fig_day.add_trace(go.Scatter(
        x=day_x, y=td, mode="lines", name="Total demand",
        line=dict(color="red", width=2),
        fill="tozeroy", fillcolor="rgba(255,0,0,0.08)",
    ))
    fig_day.add_trace(go.Scatter(
        x=day_x, y=tc, mode="lines", name="Total coverage",
        line=dict(color="green", width=2),
        fill="tozeroy", fillcolor="rgba(0,128,0,0.08)",
    ))
    if n_occ > 1:
        for i, occ in enumerate(result.occupations):
            clr = OCC_COLORS[i % len(OCC_COLORS)]
            fig_day.add_trace(go.Scatter(
                x=day_x, y=occ.demand[day_start:day_end],
                mode="lines", name=f"{occ.name} demand",
                line=dict(color=clr, width=1, dash="dash"),
            ))
            fig_day.add_trace(go.Scatter(
                x=day_x, y=occ.phase1.coverage[day_start:day_end],
                mode="lines", name=f"{occ.name} coverage",
                line=dict(color=clr, width=1),
                fill="tozeroy", fillcolor=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.10)",
            ))

    hourly_ticks = list(range(0, INTERVALS_PER_DAY, INTERVALS_PER_HOUR))
    hourly_labels = [f"{h:02d}:00" for h in range(24)]
    fig_day.update_xaxes(tickvals=hourly_ticks, ticktext=hourly_labels,
                         tickangle=45)
    fig_day.update_layout(height=400, margin=dict(l=40, r=20, t=30, b=50),
                          legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig_day, use_container_width=True)

    # ── Phase 2 summary ─────────────────────────────────────────────────
    st.subheader("Phase 2 – Worker Assignment")

    for occ in result.occupations:
        p2 = occ.phase2
        if n_occ > 1:
            st.markdown(f"**{occ.name}**")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Headcount", p2.num_workers)
        avg_h = (sum(p2.worker_hours) / max(1, p2.num_workers))
        mc2.metric("Avg weekly hours", f"{avg_h:.1f}")
        mc3.metric("FTE (÷45)", f"{sum(p2.worker_hours)/45:.1f}")
        mc4.metric("Total assigned h",
                    f"{sum(p2.worker_hours):,.0f}")
        mc5.metric("Solve time", f"{p2.elapsed_sec:.1f}s")

    # ── Gantt chart (per occupation) ─────────────────────────────────────
    st.subheader("Gantt chart")
    if n_occ > 1:
        gantt_occ = st.selectbox("Occupation",
                                 [o.name for o in result.occupations],
                                 key="gantt_occ")
        sel_occ = next(o for o in result.occupations
                       if o.name == gantt_occ)
    else:
        sel_occ = result.occupations[0]

    p2 = sel_occ.phase2
    prefix = sel_occ.name[:3].upper()
    if p2.worker_schedules:
        gantt_rows = []
        for w_idx, sched in enumerate(p2.worker_schedules):
            emp_id = f"{prefix}-{w_idx+1:03d}"
            for s in sched:
                gantt_rows.append({
                    "Worker": emp_id,
                    "Day": DAY_NAMES[s.day],
                    "Start": s.global_start,
                    "End": s.global_end,
                    "Shift": s.shift_code,
                })
        gdf = pd.DataFrame(gantt_rows)
        day_color = {d: c for d, c in zip(
            DAY_NAMES,
            ["#636EFA", "#EF553B", "#00CC96", "#AB63FA",
             "#FFA15A", "#19D3F3", "#FF6692"])}
        fig_g = go.Figure()
        for _, row in gdf.iterrows():
            fig_g.add_trace(go.Bar(
                y=[row["Worker"]], x=[row["End"] - row["Start"]],
                base=row["Start"], orientation="h",
                marker_color=day_color.get(row["Day"], "#999"),
                name=row["Day"], showlegend=False,
                hovertext=f'{row["Day"]} {row["Shift"]}',
                hoverinfo="text",
            ))
        tick_vals = [d * INTERVALS_PER_DAY for d in range(7)]
        fig_g.update_xaxes(tickvals=tick_vals, ticktext=DAY_NAMES)
        n_w = p2.num_workers
        fig_g.update_layout(
            height=max(300, 22 * n_w + 80),
            barmode="stack",
            margin=dict(l=80, r=20, t=30, b=30),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_g, use_container_width=True)

    # ── Weekly-hours histogram ───────────────────────────────────────────
    st.subheader("Weekly hours distribution")
    all_hrs = []
    for occ in result.occupations:
        for h in occ.phase2.worker_hours:
            all_hrs.append({"Hours": h, "Occupation": occ.name})
    if all_hrs:
        hdf = pd.DataFrame(all_hrs)
        fig_h = go.Figure()
        for i, occ in enumerate(result.occupations):
            clr = OCC_COLORS[i % len(OCC_COLORS)]
            sub = hdf[hdf["Occupation"] == occ.name]["Hours"]
            fig_h.add_trace(go.Histogram(
                x=sub, name=occ.name,
                marker_color=clr, opacity=0.7,
                nbinsx=20,
            ))
        fig_h.add_vline(x=params.min_weekly_hours, line_dash="dash",
                        line_color="orange",
                        annotation_text=f"Min {params.min_weekly_hours}h")
        fig_h.add_vline(x=params.max_weekly_hours, line_dash="dash",
                        line_color="red",
                        annotation_text=f"Max {params.max_weekly_hours}h")
        fig_h.update_layout(
            height=300, barmode="overlay",
            margin=dict(l=40, r=20, t=30, b=30))
        st.plotly_chart(fig_h, use_container_width=True)

    # ── Shift Types ──────────────────────────────────────────────────────
    st.subheader("Shift types")
    all_st = []
    for occ in result.occupations:
        st_df = shift_type_summary(occ.phase1, occ.name)
        all_st.append(st_df)
    if all_st:
        combined_st = pd.concat(all_st, ignore_index=True)
        st.dataframe(combined_st, use_container_width=True, hide_index=True)

    # ── Full Roster ──────────────────────────────────────────────────────
    with st.expander("📋 Full Roster"):
        all_roster = []
        emp_off = 0
        for occ in result.occupations:
            rdf = schedules_to_dataframe(occ.phase2, occ.name, emp_off)
            prefix = occ.name[:3].upper()
            if not rdf.empty:
                rdf["Worker"] = rdf["Worker"].apply(
                    lambda w: f"{prefix}-{w:03d}")
            all_roster.append(rdf)
            emp_off += occ.phase2.num_workers
        if all_roster:
            st.dataframe(pd.concat(all_roster, ignore_index=True),
                         use_container_width=True, hide_index=True)

    # ── Per-Worker Summary ───────────────────────────────────────────────
    with st.expander("👤 Per-Worker Summary"):
        ws_rows = []
        for occ in result.occupations:
            prefix = occ.name[:3].upper()
            for w_idx, (sched, hrs) in enumerate(
                    zip(occ.phase2.worker_schedules,
                        occ.phase2.worker_hours)):
                row = {
                    "Occupation": occ.name,
                    "Worker": f"{prefix}-{w_idx+1:03d}",
                    "TotalHours": round(hrs, 1),
                    "FTE(÷45)": round(hrs / 45, 2),
                    "Shifts": len(sched),
                }
                for d in range(7):
                    ds = [s for s in sched if s.day == d]
                    row[DAY_NAMES[d]] = (
                        ", ".join(s.shift_code for s in ds) if ds else "OFF")
                ws_rows.append(row)
        if ws_rows:
            st.dataframe(pd.DataFrame(ws_rows),
                         use_container_width=True, hide_index=True)

    # ── XLSX download ────────────────────────────────────────────────────
    st.subheader("📥 Download report")
    xlsx_bytes = build_weekly_report_xlsx(result)
    st.download_button(
        "⬇ Download XLSX report",
        data=xlsx_bytes,
        file_name="shiftcover_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
