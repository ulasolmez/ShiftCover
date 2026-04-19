"""
ShiftCover – Streamlit front-end
================================
Supports 1-3 occupation workload curves with shared shift structures.
"""

import io
import dataclasses
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from solver import (
    INTERVALS_PER_DAY, INTERVALS_PER_HOUR, TOTAL_INTERVALS,
    DAY_NAMES, OCC_COLORS,
    SolverParams, MultiCurveResult,
    solve_multi, list_possible_shift_codes,
    coverage_dataframe, shifts_to_dataframe,
    shift_type_summary, build_weekly_report_xlsx,
    daily_entry_headcount, max_headcount,
)
from sample_data import generate_sample_demand

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="ShiftCover", layout="wide")
st.title("🕐 ShiftCover – Weekly Shift Optimiser")

st.markdown("""
<style>
/* Sticky Run Optimiser button – floats at top of main area while scrolling */
[data-testid="stMainBlockContainer"] .stButton:has(button[data-testid="baseButton-primary"]) {
    position: sticky;
    top: 3rem;
    z-index: 100;
    background-color: var(--background-color, #0e1117);
    padding: 4px 0 6px 0;
    border-bottom: 1px solid rgba(250,250,250,0.1);
}
</style>
""", unsafe_allow_html=True)

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

    with st.expander("⚙ Shift settings", expanded=True):
        min_shift = st.slider("Min shift (h)", 3.0, 8.0, 3.0, 0.5)
        max_shift = st.slider("Max shift (h)", 6.0, 12.0, 12.0, 0.5)
        max_unique = st.number_input("Max unique shifts (0 = unlimited)", 0, 200, 0)
        no_night = st.checkbox("Exclude night shifts (≥8 h with >50 % in 20:00–06:00)", value=False)
        circular = st.checkbox("Circular week (Sunday → Monday)", value=False)

        _ALL_SLOTS = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
        _HOURLY_SLOTS = [f"{h:02d}:00" for h in range(24)]
        if "slot_filter" not in st.session_state:
            st.session_state["slot_filter"] = _ALL_SLOTS[:]
        st.caption("Allowed start & end times")
        _sc1, _sc2 = st.columns(2)
        if _sc1.button("All times", key="slots_all_btn"):
            st.session_state["slot_filter"] = _ALL_SLOTS[:]
            st.rerun()
        if _sc2.button("Every hour", key="slots_hourly_btn"):
            st.session_state["slot_filter"] = _HOURLY_SLOTS[:]
            st.rerun()
        allowed_slots_sel = st.multiselect(
            "Allowed times (start & end)",
            options=_ALL_SLOTS,
            key="slot_filter",
        )
        allowed_slot_minutes = (
            [int(t[:2]) * 60 + int(t[3:]) for t in allowed_slots_sel]
            if len(allowed_slots_sel) < 48 else None
        )

    # Build list of possible shift codes based on current settings
    _preview_params = SolverParams(
        min_shift_hours=min_shift, max_shift_hours=max_shift,
        shift_start_granularity_min=30, shift_duration_step_min=30,
        exclude_night_shifts=no_night,
        allowed_slot_minutes=allowed_slot_minutes,
    )
    _all_codes = list_possible_shift_codes(_preview_params)

    with st.expander(f"✅ Force include / exclude  ({len(_all_codes)} patterns)", expanded=False):
        st.markdown("""<style>
        div:has(> .green-ms) + div [data-baseweb="tag"] {
            background-color: #2e7d32 !important; color: white !important;
        }
        div:has(> .green-ms) + div [data-baseweb="tag"] span,
        div:has(> .green-ms) + div [data-baseweb="tag"] svg {
            color: white !important; fill: white !important;
        }
        div:has(> .red-ms) + div [data-baseweb="tag"] {
            background-color: #c62828 !important; color: white !important;
        }
        div:has(> .red-ms) + div [data-baseweb="tag"] span,
        div:has(> .red-ms) + div [data-baseweb="tag"] svg {
            color: white !important; fill: white !important;
        }
        </style>""", unsafe_allow_html=True)
        st.markdown('<span class="green-ms"></span>', unsafe_allow_html=True)
        force_incl = st.multiselect(
            "Must include (solver will always use these)",
            options=_all_codes, default=[], key="force_incl")
        _excl_options = [c for c in _all_codes if c not in set(force_incl)]
        st.markdown('<span class="red-ms"></span>', unsafe_allow_html=True)
        force_excl = st.multiselect(
            "Must exclude (solver will never use these)",
            options=_excl_options, default=[], key="force_excl")
        _conflict = set(force_incl) & set(force_excl)
        if _conflict:
            st.warning(f"Conflict: {', '.join(sorted(_conflict))} in both lists")

    with st.expander("🕐 Weekly hours", expanded=False):
        min_wh = st.number_input("Min weekly hours", 20.0, 60.0, 40.0, 1.0)
        max_wh = st.number_input("Max weekly hours", 20.0, 60.0, 50.0, 1.0)

    with st.expander("👤 Worker constraints", expanded=False):
        min_rest = st.number_input("Min rest between shifts (h)", 8.0, 24.0, 12.0, 1.0)

    DAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    with st.expander("📅 Entry / Exit limits per day", expanded=False):
        st.caption("Max distinct start times (entries) and end times (exits) "
                   "per day. 0 = unlimited.")
        max_entries = []
        max_exits = []
        for d in range(7):
            col_e, col_x = st.columns(2)
            with col_e:
                e = st.number_input(f"{DAY_SHORT[d]} entries",
                                    0, 48, 0, key=f"ent_{d}")
                max_entries.append(e)
            with col_x:
                x = st.number_input(f"{DAY_SHORT[d]} exits",
                                    0, 48, 0, key=f"ext_{d}")
                max_exits.append(x)
    use_entries = any(v > 0 for v in max_entries)
    use_exits = any(v > 0 for v in max_exits)

    with st.expander("👥 Max headcount per day", expanded=False):
        st.caption("Max simultaneous workers (all occupations) per day. "
                   "0 = unlimited.")
        max_hc = []
        for d in range(7):
            h = st.number_input(f"{DAY_SHORT[d]} max headcount",
                                0, 500, 0, key=f"hc_{d}")
            max_hc.append(h)
    use_hc = any(v > 0 for v in max_hc)

    with st.expander("🔧 Solver", expanded=False):
        time_limit = st.number_input("Time limit (s)", 10, 600, 120, 10)
        t_penalty = st.number_input("Transition penalty", 0, 500, 50, 10)

params = SolverParams(
    min_shift_hours=min_shift,
    max_shift_hours=max_shift,
    shift_start_granularity_min=30,
    shift_duration_step_min=30,
    min_weekly_hours=min_wh,
    max_weekly_hours=max_wh,
    min_rest_hours=min_rest,
    max_unique_shifts=max_unique,
    transition_penalty=t_penalty,
    solver_time_limit_sec=time_limit,
    max_entries_per_day=max_entries if use_entries else None,
    max_exits_per_day=max_exits if use_exits else None,
    max_headcount_per_day=max_hc if use_hc else None,
    exclude_night_shifts=no_night,
    circular_week=circular,
    force_include_shifts=force_incl if force_incl else None,
    force_exclude_shifts=force_excl if force_excl else None,
    allowed_slot_minutes=allowed_slot_minutes,
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
    with st.expander("❓ How to format your CSV"):
        st.markdown("""
**The optimiser needs one file per occupation** — a single numeric column
with **2 016 rows** (7 days × 288 five-minute intervals).

| Property | Value |
|---|---|
| File type | `.csv` (comma-delimited) or `.xlsx` |
| Rows | Exactly **2 016** |
| Column name | `Required` or `Demand` (or first numeric column) |
| Row order | Monday 00:00 → Sunday 23:55 |
| Interval | 5 minutes |

**Step-by-step**
1. Ensure every 5-min slot exists (288 per day × 7 days = 2 016 rows).
   If your data uses 15- or 30-min steps, repeat each value to fill the 5-min slots.
2. Sort chronologically: Monday 00:00 at the top, Sunday 23:55 at the bottom.
3. Include a numeric column named `Required` or `Demand`. Extra columns (Day, Time, …) are ignored.
4. Save as `.csv` with comma delimiters.

**Example (first rows):**
```
Day,Time,Required
Monday,00:00,2
Monday,00:05,2
Monday,00:10,3
```

**Or minimal single-column:**
```
Required
2
2
3
```

**Tips:** Zeros are fine for off-hours. Decimals are rounded automatically.
Use the *Generate sample* tab to see the exact format.

| Common error | Fix |
|---|---|
| *Need 2016 rows, got N* | Ensure all 7 × 288 slots are present |
| *No numeric column found* | Add a numeric `Required` column |
| *Missing file for …* | Upload a file for each occupation |
""")
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
    # Demand summary
    total_demand_h = sum(d.sum() for d in demands) / INTERVALS_PER_HOUR
    peak_total = int(sum(demands).max())
    _sc1, _sc2 = st.columns(2)
    _sc1.metric("Total demand (worker-hours)", f"{total_demand_h:,.0f}")
    _sc2.metric("Peak simultaneous demand", peak_total)

    # Input validation
    _warnings: list[str] = []
    if min_shift > max_shift:
        _warnings.append("Min shift hours > Max shift hours")
    if min_wh > max_wh:
        _warnings.append("Min weekly hours > Max weekly hours")
    if min_rest + min_shift > 24:
        _warnings.append("Min rest + Min shift > 24 h — no two shifts can fit in one day")
    for w in _warnings:
        st.warning(w)

    if st.button("▶ Run Optimiser", type="primary"):
        log_box = st.empty()
        log_lines: list[str] = []

        def _cb(msg):
            log_lines.append(msg)
            log_box.code("\n".join(log_lines), language="text")

        with st.spinner("Solving …"):
            result = solve_multi(demands, stored_names, params, callback=_cb)

        # stash previous before overwriting
        if "result" in st.session_state:
            st.session_state["prev_result"] = st.session_state["result"]
            st.session_state["prev_result_params"] = st.session_state.get("result_params")
        st.session_state["result"] = result
        st.session_state["result_params"] = params


# ── Results ──────────────────────────────────────────────────────────────────
result: MultiCurveResult | None = st.session_state.get("result")

if result is not None:
    st.divider()

    cp1 = result.combined_phase1
    n_occ = len(result.occupations)

    # ── Infeasibility guard ──────────────────────────────────────────────
    _feasible = cp1.status in ("OPTIMAL", "FEASIBLE")
    if not _feasible:
        st.header("📊 Results")
        st.error(
            f"**Solver status: {cp1.status}**  (solved in {cp1.elapsed_sec:.1f}s)\n\n"
            "The model has no feasible solution with the current settings. "
            "Try relaxing one or more constraints:\n"
            "- Widen the shift-duration range (lower min / raise max)\n"
            "- Reduce minimum weekly hours or raise maximum weekly hours\n"
            "- Lower minimum rest hours between shifts\n"
            "- Remove or loosen entry / exit / headcount limits\n"
            "- Uncheck *Exclude night shifts* if demand peaks at night\n"
            "- Increase the solver time limit (the solver may need more time)"
        )
        st.stop()

    st.header("📊 Results")

    # ── Phase 1 summary metrics ──────────────────────────────────────────
    st.subheader("Phase 1 – Set Covering")
    total_headcount = max_headcount(cp1)
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

    # ── Coverage quality ─────────────────────────────────────────────────
    _combined_dem = result.combined_demand
    _deficit = np.maximum(_combined_dem - cp1.coverage, 0)
    _deficit_intervals = int(np.count_nonzero(_deficit))
    _total_intervals_with_demand = int(np.count_nonzero(_combined_dem))
    _pct_covered = (
        100.0 * (1 - _deficit_intervals / _total_intervals_with_demand)
        if _total_intervals_with_demand > 0 else 100.0
    )
    _gap_wh = float(_deficit.sum()) / INTERVALS_PER_HOUR
    _max_deficit = int(_deficit.max()) if _deficit_intervals > 0 else 0
    _surplus = np.maximum(cp1.coverage - _combined_dem, 0)
    _surplus_wh = float(_surplus.sum()) / INTERVALS_PER_HOUR

    cq1, cq2, cq3, cq4 = st.columns(4)
    cq1.metric("Coverage", f"{_pct_covered:.1f} %")
    cq2.metric("Under-coverage", f"{_gap_wh:,.1f} wh")
    cq3.metric("Max deficit", _max_deficit)
    cq4.metric("Over-coverage", f"{_surplus_wh:,.1f} wh")

    if _deficit_intervals > 0:
        st.warning(
            f"Demand is not fully covered: **{_deficit_intervals}** intervals "
            f"({_gap_wh:.1f} worker-hours) remain below demand. "
            f"Peak deficit: **{_max_deficit}** workers."
        )

    # ── What changed vs previous solve ──────────────────────────────────
    _prev_result: MultiCurveResult | None = st.session_state.get("prev_result")
    _prev_params: SolverParams | None = st.session_state.get("prev_result_params")
    _cur_params: SolverParams | None = st.session_state.get("result_params")
    if (_prev_result is not None
            and _prev_result.combined_phase1.status in ("OPTIMAL", "FEASIBLE")):
        with st.expander("🔄 What changed vs previous solve", expanded=True):
            _pcp1 = _prev_result.combined_phase1

            # --- param diff ---
            _param_rows = []
            if _prev_params is not None and _cur_params is not None:
                for _f in dataclasses.fields(SolverParams):
                    _ov = getattr(_prev_params, _f.name)
                    _nv = getattr(_cur_params, _f.name)
                    if _ov != _nv:
                        _param_rows.append({"Parameter": _f.name,
                                            "Previous": _ov, "Now": _nv})
            if _param_rows:
                st.caption("**Settings changed:**")
                st.dataframe(pd.DataFrame(_param_rows),
                             use_container_width=True, hide_index=True)
            else:
                st.caption("Settings unchanged.")

            # --- metric deltas ---
            _prev_wh = _pcp1.total_worker_intervals / INTERVALS_PER_HOUR
            _prev_active = sum(1 for c in _pcp1.counts if c > 0)
            _prev_hc = max_headcount(_pcp1)
            _prev_peak_sim = int(_pcp1.coverage.max())
            _prev_deficit_arr = np.maximum(
                _prev_result.combined_demand - _pcp1.coverage, 0)
            _prev_gap_wh = float(_prev_deficit_arr.sum()) / INTERVALS_PER_HOUR
            _prev_di = int(np.count_nonzero(_prev_deficit_arr))
            _prev_tid = int(np.count_nonzero(_prev_result.combined_demand))
            _prev_pct = (100.0 * (1 - _prev_di / _prev_tid)
                         if _prev_tid > 0 else 100.0)

            st.caption("**Metrics:**")
            _dc1, _dc2, _dc3, _dc4, _dc5 = st.columns(5)
            _dc1.metric("Active shifts", n_active,
                        delta=n_active - _prev_active)
            _dc2.metric("Worker-hours", f"{total_wh:,.0f}",
                        delta=f"{total_wh - _prev_wh:+,.0f}")
            _dc3.metric("Headcount", total_headcount,
                        delta=total_headcount - _prev_hc,
                        delta_color="inverse")
            _dc4.metric("Coverage", f"{_pct_covered:.1f} %",
                        delta=f"{_pct_covered - _prev_pct:+.1f} %")
            _dc5.metric("Under-cov (wh)", f"{_gap_wh:.1f}",
                        delta=f"{_gap_wh - _prev_gap_wh:+.1f}",
                        delta_color="inverse")

            # --- shifts added / removed ---
            _prev_codes: set[str] = set()
            for _occ in _prev_result.occupations:
                for _s, _cnt in zip(_occ.phase1.shifts, _occ.phase1.counts):
                    if _cnt > 0:
                        _prev_codes.add(_s.shift_code)
            _cur_codes: set[str] = set()
            for _occ in result.occupations:
                for _s, _cnt in zip(_occ.phase1.shifts, _occ.phase1.counts):
                    if _cnt > 0:
                        _cur_codes.add(_s.shift_code)
            _added = sorted(_cur_codes - _prev_codes)
            _removed = sorted(_prev_codes - _cur_codes)
            if _added or _removed:
                _ch1, _ch2 = st.columns(2)
                if _added:
                    _ch1.caption(f"**Shifts added ({len(_added)}):**")
                    _ch1.write(", ".join(_added))
                if _removed:
                    _ch2.caption(f"**Shifts removed ({len(_removed)}):**")
                    _ch2.write(", ".join(_removed))
            else:
                st.caption("No shift types added or removed.")

    # per-occupation headcount
    if n_occ > 1:
        occ_cols = st.columns(n_occ)
        for i, occ in enumerate(result.occupations):
            with occ_cols[i]:
                occ_wh = occ.phase1.total_worker_intervals / INTERVALS_PER_HOUR
                occ_hc = max_headcount(occ.phase1)
                st.metric(f"{occ.name} headcount", occ_hc)
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
    # shade deficit (demand > coverage)
    _day_deficit = np.maximum(td - tc, 0)
    if _day_deficit.any():
        fig_day.add_trace(go.Scatter(
            x=day_x, y=td, mode="lines", name="Deficit",
            line=dict(width=0), showlegend=True,
        ))
        fig_day.add_trace(go.Scatter(
            x=day_x, y=tc, mode="lines", name="_deficit_fill",
            line=dict(width=0), showlegend=False,
            fill="tonexty", fillcolor="rgba(255,0,0,0.25)",
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

    # ── Shift Types ──────────────────────────────────────────────────────
    st.subheader("Shift types")
    all_st = []
    for occ in result.occupations:
        st_df = shift_type_summary(occ.phase1, occ.name)
        all_st.append(st_df)
    if all_st:
        combined_st = pd.concat(all_st, ignore_index=True)
        st.dataframe(combined_st, use_container_width=True, hide_index=True)

    # ── XLSX download ────────────────────────────────────────────────────
    st.subheader("📥 Download report")
    xlsx_bytes = build_weekly_report_xlsx(result)
    st.download_button(
        "⬇ Download XLSX report",
        data=xlsx_bytes,
        file_name="shiftcover_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
