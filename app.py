"""
ShiftCover – Streamlit UI
=========================
Upload a weekly demand curve (2 016 × 5-min intervals) and let OR-Tools
find optimal shift assignments that cover every interval.

Launch:  streamlit run app.py
"""

import io
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from solver import (
    TOTAL_INTERVALS, INTERVALS_PER_DAY, INTERVALS_PER_HOUR, DAY_NAMES,
    SolverParams, solve, shifts_to_dataframe, schedules_to_dataframe,
    coverage_dataframe, shift_type_summary, build_weekly_report_xlsx,
)
from sample_data import generate_sample_demand, demand_to_dataframe

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShiftCover – Weekly Shift Optimiser",
    page_icon="📊",
    layout="wide",
)

st.title("📊 ShiftCover – Weekly Shift Optimiser")
st.markdown(
    "Upload a **weekly demand curve** (2 016 five-minute intervals, 288 per "
    "day, Monday → Sunday) and the solver will find the optimal shifts "
    "(3–12 h) to cover every interval while respecting weekly-hour targets."
)

# ── Sidebar – parameters ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Solver Parameters")

    st.subheader("Shift Settings")
    col1, col2 = st.columns(2)
    min_shift = col1.number_input("Min shift (h)", 1.0, 12.0, 3.0, 0.5)
    max_shift = col2.number_input("Max shift (h)", 1.0, 16.0, 12.0, 0.5)

    st.subheader("Start-Time / Duration Granularity")
    start_gran = st.selectbox(
        "Shift start-time step (min)",
        [5, 10, 15, 30, 60],
        index=2,
        help="Smaller = more candidate shifts = slower but finer grained",
    )
    dur_step = st.selectbox(
        "Duration step (min)", [15, 30, 60], index=1
    )

    st.subheader("Weekly Hours per Worker")
    col3, col4 = st.columns(2)
    min_weekly = col3.number_input("Min weekly (h)", 0.0, 80.0, 40.0, 1.0)
    max_weekly = col4.number_input("Max weekly (h)", 0.0, 80.0, 50.0, 1.0)

    st.subheader("Worker Constraints")
    max_shifts_day = st.number_input(
        "Max shifts per day per worker", 1, 3, 1
    )
    min_rest = st.number_input(
        "Min rest between shifts (h)", 0.0, 24.0, 12.0, 1.0,
        help="Legal minimum rest period between consecutive shifts",
    )

    st.subheader("Solver")
    time_limit = st.slider("Time limit per phase (s)", 10, 600, 120, 10)

    st.divider()
    run_phase2 = st.checkbox("Run Phase 2 (worker assignment)", value=True,
                              help="Bin-pack shifts into individual workers")

# ── Demand upload / sample ───────────────────────────────────────────────────
st.header("1 – Demand Curve")

tab_upload, tab_sample = st.tabs(["📁 Upload file", "🎲 Use sample data"])

# Persist demand across reruns
if "demand" not in st.session_state:
    st.session_state["demand"] = None

with tab_upload:
    st.markdown(textwrap.dedent("""\
        Upload a **CSV** or **Excel** file.  The file must contain a column
        of **2 016 integer values** representing the number of personnel
        required per 5-minute interval across the whole week
        (Mon 00:00 → Sun 23:55).

        Accepted column names: `Required`, `Demand`, `Headcount`, `Agents`,
        or simply the **first numeric column** will be used.
    """))
    uploaded = st.file_uploader(
        "Choose CSV or XLSX", type=["csv", "xlsx", "xls"]
    )
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)

            # find the demand column
            target_names = {"required", "demand", "headcount", "agents"}
            demand_col = None
            for c in df_up.columns:
                if c.strip().lower() in target_names:
                    demand_col = c
                    break
            if demand_col is None:
                # fall back to first numeric column
                for c in df_up.columns:
                    if pd.api.types.is_numeric_dtype(df_up[c]):
                        demand_col = c
                        break

            if demand_col is None:
                st.error("❌ No numeric column found in the uploaded file.")
            else:
                vals = df_up[demand_col].dropna().values
                if len(vals) < TOTAL_INTERVALS:
                    st.error(
                        f"❌ Need {TOTAL_INTERVALS} rows, got {len(vals)}."
                    )
                else:
                    st.session_state["demand"] = np.round(vals[:TOTAL_INTERVALS]).astype(int)
                    d_arr = st.session_state["demand"]
                    st.success(
                        f"✅ Loaded **{demand_col}** column — "
                        f"min {d_arr.min()}, max {d_arr.max()}, "
                        f"mean {d_arr.mean():.1f}"
                    )
        except Exception as exc:
            st.error(f"❌ Error reading file: {exc}")

with tab_sample:
    st.markdown("Generate a synthetic demand curve for testing.")
    scol1, scol2 = st.columns(2)
    peak = scol1.slider("Peak agents", 5, 100, 25)
    base = scol2.slider("Base agents", 0, 20, 2)
    if st.button("Generate sample"):
        st.session_state["demand"] = generate_sample_demand(
            peak_agents=peak, base_agents=base
        )
        st.success("✅ Sample demand generated")

# ── Show demand chart ────────────────────────────────────────────────────────
demand = st.session_state["demand"]
if demand is not None:
    with st.expander("📈 Preview demand curve", expanded=True):
        x_labels = []
        for t in range(TOTAL_INTERVALS):
            day = t // INTERVALS_PER_DAY
            intra = t % INTERVALS_PER_DAY
            h, m_ = divmod(intra * 5, 60)
            x_labels.append(f"{DAY_NAMES[day][:3]} {h:02d}:{m_:02d}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(TOTAL_INTERVALS)), y=demand,
            mode="lines", name="Required",
            line=dict(color="#1f77b4"),
            hovertext=x_labels,
        ))
        # day separators
        for d in range(1, 7):
            fig.add_vline(x=d * INTERVALS_PER_DAY, line_dash="dot",
                          line_color="grey", opacity=0.5)
        fig.update_layout(
            height=300, margin=dict(l=40, r=20, t=30, b=40),
            xaxis=dict(
                tickvals=[d * INTERVALS_PER_DAY + 144 for d in range(7)],
                ticktext=[DAY_NAMES[d][:3] for d in range(7)],
            ),
            yaxis_title="Agents required",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Download template ────────────────────────────────────────────────
    with st.expander("⬇️ Download this demand as CSV"):
        csv_buf = demand_to_dataframe(demand).to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv_buf,
            file_name="demand_curve.csv",
            mime="text/csv",
        )

# ── Solve ────────────────────────────────────────────────────────────────────
st.header("2 – Optimise")

if demand is None:
    st.info("⬆️ Upload or generate a demand curve first.")
    st.stop()

if st.button("🚀 Run Optimiser", type="primary"):
    params = SolverParams(
        min_shift_hours=min_shift,
        max_shift_hours=max_shift,
        shift_start_granularity_min=start_gran,
        shift_duration_step_min=dur_step,
        min_weekly_hours=min_weekly,
        max_weekly_hours=max_weekly,
        max_shifts_per_day_per_worker=max_shifts_day,
        min_rest_hours=min_rest,
        solver_time_limit_sec=time_limit,
    )

    log_area = st.empty()
    log_lines: list[str] = []

    def log_cb(msg: str):
        log_lines.append(msg)
        log_area.code("\n".join(log_lines), language="text")

    with st.spinner("Running OR-Tools CP-SAT solver …"):
        result = solve(demand, params, callback=log_cb)

    st.session_state["result"] = result
    st.session_state["params"] = params

# ── Results ──────────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.stop()

result = st.session_state["result"]
params = st.session_state["params"]

st.header("3 – Results")

# ---- Phase 1 summary ----
p1 = result.phase1
st.subheader("Phase 1 – Set Covering")

mcol1, mcol2, mcol3, mcol4, mcol5, mcol6, mcol7 = st.columns(7)
mcol1.metric("Status", p1.status)
mcol2.metric("Active shifts",
             sum(1 for c in p1.counts if c > 0) if p1.counts else 0)
total_wh = p1.total_worker_intervals / INTERVALS_PER_HOUR if p1.counts else 0
mcol3.metric("Total worker-hours", f"{total_wh:,.0f}")
# Headcount = total unique workers from Phase 2
p2_preview = result.phase2
headcount = p2_preview.num_workers if p2_preview.num_workers else "–"
mcol4.metric("Headcount", headcount)
# Peak simultaneous workers needed (max coverage value)
peak_headcount = int(p1.coverage.max()) if p1.counts else 0
mcol5.metric("Peak Simultaneous", peak_headcount)
# FTE = total worker-hours / 45
fte = total_wh / 45.0
mcol6.metric("FTE (÷45 h)", f"{fte:.1f}")
mcol7.metric("Solve time", f"{p1.elapsed_sec:.1f} s")

if p1.status not in ("OPTIMAL", "FEASIBLE"):
    st.error(f"Solver returned **{p1.status}** – try relaxing constraints.")
    st.stop()

# ---- Coverage chart (weekly overview – filled area) ----
cov_df = coverage_dataframe(result)
with st.expander("📈 Demand vs. Coverage – Full Week", expanded=True):
    fig2 = go.Figure()
    # Coverage filled area (drawn first so demand line sits on top)
    fig2.add_trace(go.Scatter(
        x=cov_df["Interval"], y=cov_df["Coverage"],
        mode="lines", name="Coverage",
        line=dict(color="#2ca02c", width=1),
        fill="tozeroy",
        fillcolor="rgba(44,160,44,0.25)",
    ))
    # Demand filled area
    fig2.add_trace(go.Scatter(
        x=cov_df["Interval"], y=cov_df["Demand"],
        mode="lines", name="Demand",
        line=dict(color="#1f77b4", width=2),
        fill="tozeroy",
        fillcolor="rgba(31,119,180,0.18)",
    ))
    for d in range(1, 7):
        fig2.add_vline(x=d * INTERVALS_PER_DAY, line_dash="dot",
                       line_color="grey", opacity=0.4)
    fig2.update_layout(
        height=350, margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(
            tickvals=[d * INTERVALS_PER_DAY + 144 for d in range(7)],
            ticktext=[DAY_NAMES[d][:3] for d in range(7)],
        ),
        yaxis_title="Agents",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # over / under coverage
    over = (cov_df["Coverage"] - cov_df["Demand"]).clip(lower=0).sum()
    under = (cov_df["Demand"] - cov_df["Coverage"]).clip(lower=0).sum()
    st.markdown(
        f"**Over-coverage:** {over:,} interval-slots &nbsp;|&nbsp; "
        f"**Under-coverage:** {under:,} interval-slots"
    )

# ---- Per-day coverage chart (dropdown selector) ----
with st.expander("📅 Daily Demand vs. Coverage", expanded=True):
    # build time labels for a single day (00:00 – 23:55)
    day_time_labels = []
    for i in range(INTERVALS_PER_DAY):
        h, m_ = divmod(i * 5, 60)
        day_time_labels.append(f"{h:02d}:{m_:02d}")

    selected_day_name = st.selectbox(
        "Select day", DAY_NAMES, index=0, key="day_selector"
    )
    day = DAY_NAMES.index(selected_day_name)

    start_t = day * INTERVALS_PER_DAY
    end_t = start_t + INTERVALS_PER_DAY
    day_demand = cov_df["Demand"].values[start_t:end_t]
    day_coverage = cov_df["Coverage"].values[start_t:end_t]

    # daily metrics row
    day_peak_hc = int(day_coverage.max())
    day_total_hrs = float(day_coverage.sum()) / INTERVALS_PER_HOUR
    day_demand_hrs = float(day_demand.sum()) / INTERVALS_PER_HOUR
    day_over = int(np.clip(day_coverage - day_demand, 0, None).sum())
    day_under = int(np.clip(day_demand - day_coverage, 0, None).sum())

    dc1, dc2, dc3, dc4, dc5 = st.columns(5)
    dc1.metric("Peak Headcount", day_peak_hc)
    dc2.metric("Coverage hrs", f"{day_total_hrs:.0f}")
    dc3.metric("Demand hrs", f"{day_demand_hrs:.0f}")
    dc4.metric("Over-coverage", f"{day_over:,} slots")
    dc5.metric("Under-coverage", f"{day_under:,} slots")

    fig_day = go.Figure()
    fig_day.add_trace(go.Scatter(
        x=list(range(INTERVALS_PER_DAY)),
        y=day_coverage,
        mode="lines", name="Coverage",
        line=dict(color="#2ca02c", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(44,160,44,0.25)",
        hovertext=day_time_labels,
    ))
    fig_day.add_trace(go.Scatter(
        x=list(range(INTERVALS_PER_DAY)),
        y=day_demand,
        mode="lines", name="Demand",
        line=dict(color="#1f77b4", width=2),
        fill="tozeroy",
        fillcolor="rgba(31,119,180,0.18)",
        hovertext=day_time_labels,
    ))
    fig_day.update_layout(
        title=dict(
            text=f"{selected_day_name}",
            font=dict(size=16),
        ),
        height=420,
        margin=dict(l=50, r=20, t=45, b=50),
        xaxis=dict(
            tickvals=list(range(0, INTERVALS_PER_DAY, 12)),  # every hour
            ticktext=[day_time_labels[i]
                      for i in range(0, INTERVALS_PER_DAY, 12)],
            tickangle=-45,
            title="Time of day",
        ),
        yaxis_title="Agents",
        legend=dict(orientation="h", y=1.10),
    )
    st.plotly_chart(fig_day, use_container_width=True)

# ---- Shift table ----
with st.expander("📋 Active Shifts (Phase 1)"):
    shift_df = shifts_to_dataframe(p1)
    st.dataframe(shift_df, use_container_width=True, hide_index=True)
    csv1 = shift_df.to_csv(index=False)
    st.download_button("Download shifts CSV", csv1,
                       file_name="shifts.csv", mime="text/csv")

# ---- Phase 2 ----
p2 = result.phase2
if p2.status == "SKIPPED":
    st.info("Phase 2 was skipped.")
elif p2.status == "NO_SHIFTS":
    st.warning("No shift instances to assign.")
else:
    st.subheader("Phase 2 – Worker Assignment")
    wcol1, wcol2, wcol3, wcol4, wcol5 = st.columns(5)
    wcol1.metric("Headcount (workers)", p2.num_workers)
    avg_wh = np.mean(p2.worker_hours) if p2.worker_hours else 0
    wcol2.metric(
        "Avg weekly hours",
        f"{avg_wh:.1f}" if p2.worker_hours else "–"
    )
    total_assigned_hrs = sum(p2.worker_hours) if p2.worker_hours else 0
    fte_p2 = total_assigned_hrs / 45.0
    wcol3.metric("FTE (÷45 h)", f"{fte_p2:.1f}")
    wcol4.metric("Total assigned hours", f"{total_assigned_hrs:,.0f}")
    wcol5.metric("Solve time", f"{p2.elapsed_sec:.1f} s")

    if "BELOW" in p2.status.upper() or "below" in p2.status:
        st.warning(f"⚠️ {p2.status}")

    sched_df = schedules_to_dataframe(p2)

    # ---- Gantt-style chart ----
    with st.expander("📅 Worker Schedule (Gantt view)", expanded=True):
        fig3 = go.Figure()
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        for _, row in sched_df.iterrows():
            w = int(row["Worker"])
            d = int(row["DayNum"])
            sh, sm = map(int, row["Start"].split(":"))
            eh, em = map(int, row["End"].split(":"))
            start_h = d * 24 + sh + sm / 60
            end_h = d * 24 + eh + em / 60
            fig3.add_trace(go.Bar(
                x=[end_h - start_h],
                y=[f"EMP-{w:03d}"],
                base=start_h,
                orientation="h",
                marker_color=colors[d % len(colors)],
                name=DAY_NAMES[d],
                showlegend=False,
                hovertemplate=(
                    f"EMP-{w:03d}<br>{DAY_NAMES[d]} "
                    f"{row['Start']}–{row['End']} "
                    f"({row['DurationHrs']:.1f} h)<extra></extra>"
                ),
            ))
        fig3.update_layout(
            barmode="stack",
            height=max(300, p2.num_workers * 28 + 80),
            margin=dict(l=70, r=20, t=30, b=50),
            xaxis=dict(
                title="Hour of week",
                tickvals=[d * 24 + 12 for d in range(7)],
                ticktext=[DAY_NAMES[d][:3] for d in range(7)],
            ),
            yaxis=dict(autorange="reversed"),
        )
        for d in range(1, 7):
            fig3.add_vline(x=d * 24, line_dash="dot",
                           line_color="grey", opacity=0.4)
        st.plotly_chart(fig3, use_container_width=True)

    # ---- Worker hours histogram ----
    with st.expander("📊 Weekly hours distribution"):
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(
            x=p2.worker_hours, nbinsx=20,
            marker_color="#1f77b4",
        ))
        fig4.add_vline(x=params.min_weekly_hours, line_dash="dash",
                       line_color="red", annotation_text="Min")
        fig4.add_vline(x=params.max_weekly_hours, line_dash="dash",
                       line_color="red", annotation_text="Max")
        fig4.update_layout(
            height=280,
            xaxis_title="Weekly hours",
            yaxis_title="Workers",
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ---- Shift Type Summary ----
    with st.expander("📋 Shift Types (HHMM-HHMM)", expanded=True):
        st_df = shift_type_summary(p1)
        st.dataframe(st_df, use_container_width=True, hide_index=True)

    # ---- Full Roster table ----
    with st.expander("📋 Full Roster"):
        # build roster with EMP IDs and shift codes
        roster_rows = []
        for w_idx, (schedule, hrs) in enumerate(
                zip(p2.worker_schedules, p2.worker_hours)):
            for s in schedule:
                roster_rows.append({
                    "Worker": f"EMP-{w_idx + 1:03d}",
                    "Day": DAY_NAMES[s.day],
                    "Shift": s.shift_code,
                    "Start": s.start_time_str,
                    "End": s.end_time_str,
                    "Hours": s.duration_hours,
                    "WeeklyHrs": hrs,
                })
        roster_df = pd.DataFrame(roster_rows)
        st.dataframe(roster_df, use_container_width=True, hide_index=True)

    # ---- Per-worker summary ----
    with st.expander("👤 Per-Worker Summary"):
        summary_rows = []
        for w_idx, (sched, hrs) in enumerate(
                zip(p2.worker_schedules, p2.worker_hours)):
            row = {
                "Worker": f"EMP-{w_idx + 1:03d}",
                "Shifts": len(sched),
                "WeeklyHours": f"{hrs:.1f}",
                "FTE(÷45)": f"{hrs / 45.0:.2f}",
            }
            for d in range(7):
                day_shifts = [s for s in sched if s.day == d]
                if day_shifts:
                    row[DAY_NAMES[d][:3]] = ", ".join(
                        s.shift_code for s in day_shifts)
                else:
                    row[DAY_NAMES[d][:3]] = "OFF"
            summary_rows.append(row)
        st.dataframe(pd.DataFrame(summary_rows),
                     use_container_width=True, hide_index=True)

    # ---- XLSX Download ----
    st.subheader("⬇️ Download Weekly Report")
    xlsx_bytes = build_weekly_report_xlsx(result)
    st.download_button(
        "📥 Download XLSX Report",
        data=xlsx_bytes,
        file_name="weekly_shift_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "ShiftCover © 2026 — Powered by Google OR-Tools CP-SAT & Streamlit"
)
