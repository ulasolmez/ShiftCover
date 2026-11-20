"""
ShiftCover – Streamlit front-end
================================
Supports 1-5 occupation workload curves with shared shift structures.
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
    buffer_violations,
)
from sample_data import generate_sample_demand
from groundstar_import import read_workload_file, curve_week, combine_by_occupation
from roster import (RosterParams, slots_from_phase1, solve_roster, precheck,
                    verify_roster, OFF_MODES)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Simplex", layout="wide")
st.title("🕐 Simplex – Weekly Shift Optimiser")

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

MAX_CURVES = 10

# ── Apply occupations from an import (before the sidebar widgets exist) ─────
if st.session_state.get("_pending_occ"):
    _names = st.session_state.pop("_pending_occ")
    st.session_state["n_curves"] = len(_names)
    for _i, _n in enumerate(_names):
        st.session_state[f"occ_name_{_i}"] = _n

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")

    st.session_state.setdefault("n_curves", 1)
    n_curves = st.number_input("Number of occupation curves", 1, MAX_CURVES,
                               key="n_curves",
                               help="Set automatically when you load a workload import.")
    occ_names = []
    _default_names = (["Technician", "Labourer", "Helper", "Supervisor", "Assistant"]
                      + [f"Occupation {k}" for k in range(6, MAX_CURVES + 1)])
    for i in range(n_curves):
        st.session_state.setdefault(f"occ_name_{i}", _default_names[i])
        name = st.text_input(f"Occupation {i+1} name", key=f"occ_name_{i}")
        occ_names.append(name)

    with st.expander("⚙ Shift settings", expanded=True):
        min_shift = st.slider("Min shift (h)", 3.0, 8.0, 3.0, 0.5)
        max_shift = st.slider("Max shift (h)", 6.0, 12.0, 12.0, 0.5)
        max_unique = st.number_input("Max unique shift types (0 = unlimited)", 0, 200, 0,
                                     help="Distinct shift codes (e.g. 0800-1600) across the week; "
                                          "a code used on several days counts once.")
        no_night = st.checkbox("Exclude night shifts (>8.5 h with ≥50 % in 20:00–06:00)", value=False)
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

    with st.expander("👥 Roster (people needed)", expanded=True):
        build_roster = st.checkbox("Build roster after planning shifts", value=True,
                                   help="Assigns every planned shift to a person to "
                                        "get the real headcount.")
        _off_labels = {
            "1": "1 off — ≤ 6 shifts, one rest ≥ 24 h",
            "1.5": "1.5 off — half 1-off / half 2-off (alternating weeks)",
            "2": "2 off — ≤ 5 shifts",
            "auto": "Auto — solver picks 1 or 2 off per person",
        }
        off_mode = st.selectbox("Days off per week", OFF_MODES, index=2,
                                format_func=_off_labels.get)
        target_wh = st.number_input("Target paid hours / week", 20.0, 60.0, 45.0, 0.5,
                                    help="Hours are balanced towards this. In 1.5-off "
                                         "mode the two-week average hits it.")
        min_wh, max_wh = st.slider("Paid hours window / week", 20.0, 70.0, (40.0, 50.0), 0.5,
                                   help="Max is a hard limit; falling below min is "
                                        "allowed but minimised and reported.")
        min_rest = st.number_input("Min rest between shifts (h)", 8.0, 24.0, 12.0, 0.5,
                                   help="Exactly this much is allowed (≥).")
        off_rest = st.number_input("1-off weekly rest (h)", 12.0, 48.0, 24.0, 0.5,
                                   help="A 1-off person needs one rest at least this "
                                        "long somewhere in the week.")
        allow_pt = st.checkbox("Allow part-time staff", value=False)
        if allow_pt:
            pt_min, pt_max = st.slider("Part-time paid hours / week", 5.0, 40.0,
                                       (20.0, 30.0), 0.5)
            pt_shifts = st.number_input("Part-time max shifts / week", 1, 6, 4)
        else:
            pt_min, pt_max, pt_shifts = 20.0, 30.0, 4
        roster_time = st.number_input("Roster time limit per occupation (s)",
                                      10, 1800, 60, 10)

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

    with st.expander("👥 Headcount limits per day (combined)", expanded=False):
        st.caption("Max and min workers entering shifts (all occupations combined) "
                   "per day — counts workers whose shift STARTS on the day. "
                   "0 = unlimited / no minimum.")
        max_hc = []
        min_hc = []
        for d in range(7):
            col_max, col_min = st.columns(2)
            with col_max:
                mx = st.number_input(f"{DAY_SHORT[d]} max",
                                     0, 500, 0, key=f"hc_{d}")
                max_hc.append(mx)
            with col_min:
                mn = st.number_input(f"{DAY_SHORT[d]} min",
                                     0, 500, 0, key=f"min_hc_{d}")
                min_hc.append(mn)
    use_hc = any(v > 0 for v in max_hc)
    use_min_hc = any(v > 0 for v in min_hc)

    # Per-occupation daily headcount limits
    occ_max_hc: list = []
    with st.expander("👤 Max headcount per occupation per day", expanded=False):
        st.caption("Set how many workers of each occupation can enter a shift "
                   "per day. 0 = unlimited (use combined limit or no restriction).")
        for i in range(n_curves):
            st.markdown(f"**{occ_names[i]}**")
            occ_daily = []
            cols = st.columns(7)
            for d in range(7):
                with cols[d]:
                    st.caption(DAY_SHORT[d])
                    v = st.number_input(
                        f"##{occ_names[i]}_{DAY_SHORT[d]}",
                        0, 500, 0,
                        key=f"occ_hc_{i}_{d}",
                        label_visibility="collapsed",
                    )
                    occ_daily.append(v)
            # 0 = unlimited (no constraint applied in solver)
            occ_max_hc.append(occ_daily if any(v > 0 for v in occ_daily) else None)
    use_occ_hc = any(v is not None for v in occ_max_hc)

    # Per-occupation daily minimum headcount
    occ_min_hc: list = []
    with st.expander("👤 Min headcount per occupation per day", expanded=False):
        st.caption("Set a minimum number of workers of each occupation that must "
                   "enter a shift per day. 0 = no minimum.")
        for i in range(n_curves):
            st.markdown(f"**{occ_names[i]}**")
            occ_daily_min = []
            cols = st.columns(7)
            for d in range(7):
                with cols[d]:
                    st.caption(DAY_SHORT[d])
                    v = st.number_input(
                        f"##min_{occ_names[i]}_{DAY_SHORT[d]}",
                        0, 500, 0,
                        key=f"occ_min_hc_{i}_{d}",
                        label_visibility="collapsed",
                    )
                    occ_daily_min.append(v)
            occ_min_hc.append(occ_daily_min if any(v > 0 for v in occ_daily_min) else None)
    use_occ_min_hc = any(v is not None for v in occ_min_hc)

    # Per-occupation per-shift-code headcount limits
    occ_sc_hc: list = []
    with st.expander("🕐 Shift-type headcount limits", expanded=False):
        st.caption("Set daily max/min workers per shift code per occupation. "
                   "Select a shift code, then configure per-day limits. "
                   "0 = unlimited / no minimum. "
                   "Leave unselected codes unconstrained.")
        for i in range(n_curves):
            st.markdown(f"**{occ_names[i]}**")
            # Let user select which shift codes to constrain for this occupation
            selected_codes = st.multiselect(
                f"Shift codes to constrain for {occ_names[i]}",
                options=_all_codes,
                default=[],
                key=f"sc_hc_codes_{i}",
            )
            if selected_codes:
                sc_dict = {}
                for code in selected_codes:
                    st.caption(f"  {code}")
                    max_daily = []
                    min_daily = []
                    # Day header row (first column empty for label)
                    hdr_cols = st.columns([0.4, 1, 1, 1, 1, 1, 1, 1])
                    with hdr_cols[0]:
                        st.caption("")
                    for d in range(7):
                        with hdr_cols[d + 1]:
                            st.caption(DAY_SHORT[d])
                    # Max row
                    max_cols = st.columns([0.4, 1, 1, 1, 1, 1, 1, 1])
                    with max_cols[0]:
                        st.caption("Max")
                    for d in range(7):
                        with max_cols[d + 1]:
                            mx = st.number_input(
                                f"max_{i}_{code}_{d}", 0, 500, 0,
                                key=f"sc_hc_max_{i}_{code.replace('-','_')}_{d}",
                                label_visibility="collapsed",
                            )
                            max_daily.append(mx)
                    # Min row
                    min_cols = st.columns([0.4, 1, 1, 1, 1, 1, 1, 1])
                    with min_cols[0]:
                        st.caption("Min")
                    for d in range(7):
                        with min_cols[d + 1]:
                            mn = st.number_input(
                                f"min_{i}_{code}_{d}", 0, 500, 0,
                                key=f"sc_hc_min_{i}_{code.replace('-','_')}_{d}",
                                label_visibility="collapsed",
                            )
                            min_daily.append(mn)
                    has_max = any(v > 0 for v in max_daily)
                    has_min = any(v > 0 for v in min_daily)
                    if has_max or has_min:
                        sc_dict[code] = {}
                        if has_max:
                            sc_dict[code]["max"] = max_daily
                        if has_min:
                            sc_dict[code]["min"] = min_daily
                occ_sc_hc.append(sc_dict if sc_dict else None)
            else:
                occ_sc_hc.append(None)
    use_occ_sc_hc = any(v is not None for v in occ_sc_hc)

    with st.expander("⏱ Operational buffers", expanded=True):
        st.caption("Real-world lead/lag around the workload curve. "
                   "Set to 0 to get the raw mathematical minimum.")
        prep_lead = st.number_input(
            "Prep before operation start (min)", 0, 120, 30, 5,
            help="Workers must be on shift this long before workload rises. "
                 "Operation at 08:30 → shift starts 08:00.")
        exit_lag = st.number_input(
            "Stay after operation end (min)", 0, 120, 30, 5,
            help="Workers stay this long after workload drops. "
                 "Operation ends 17:00 → shift ends 17:30.")
        handover = st.number_input(
            "Handover overlap (min)", 0, 120, 30, 5,
            help="When a worker leaves while the operation continues, "
                 "the replacement arrives this much earlier so both overlap.")

    with st.expander("🔧 Solver", expanded=False):
        time_limit = st.number_input("Time limit (s)", 10, 600, 120, 10)
        t_penalty = st.number_input("Transition penalty", 0, 500, 50, 10)

roster_params = RosterParams(
    off_mode=off_mode, target_hours=target_wh, min_hours=min_wh, max_hours=max_wh,
    min_rest_hours=min_rest, off_rest_hours=off_rest, allow_part_time=allow_pt,
    pt_min_hours=pt_min, pt_max_hours=pt_max, pt_max_shifts=int(pt_shifts),
    circular_week=circular, time_limit_sec=float(roster_time),
)

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
    min_headcount_per_day=min_hc if use_min_hc else None,
    occ_max_headcount_per_day=occ_max_hc if use_occ_hc else None,
    occ_min_headcount_per_day=occ_min_hc if use_occ_min_hc else None,
    occ_headcount_per_shift_code=occ_sc_hc if use_occ_sc_hc else None,
    exclude_night_shifts=no_night,
    circular_week=circular,
    force_include_shifts=force_incl if force_incl else None,
    force_exclude_shifts=force_excl if force_excl else None,
    allowed_slot_minutes=allowed_slot_minutes,
    prep_lead_min=int(prep_lead),
    exit_lag_min=int(exit_lag),
    handover_overlap_min=int(handover),
)


# ── Demand input ─────────────────────────────────────────────────────────────
st.header("📂 Demand input")

tab_upload, tab_sample = st.tabs(["⬆ Upload files", "🎲 Generate sample"])

with tab_upload:
    st.info("Drop your **GroundStar workload exports** (.xlsx with a *Curve Data* "
            "sheet) — one or several departments at once. Plain CSV/XLSX curves "
            f"with {TOTAL_INTERVALS} rows still work too.")
    with st.expander("❓ Supported files"):
        st.markdown(f"""
**GroundStar workload export (recommended)** — export the workload curve from
GroundStar Planning to Excel and upload the file as is. The *Curve Data* sheet is
read directly: metadata, the time step and every curve column (one per resource
type / qualification, depending on *Curve Building Criteria*).

* Each curve becomes a row in the table below. Set its **Occupation** — this is
  the classification the optimiser plans for (e.g. `DCL`, `CLN`, `INF`).
* Rows with the **same occupation** are summed (e.g. two files for one department).
* Untick **Use** to leave a curve out.
* If the export spans more than one Monday–Sunday week, pick the week per file.
* 15/30/60-min exports are converted to 5-min steps; fractional workload is
  rounded **up**.

**Plain curve** — `.csv`/`.xlsx`, one numeric column (`Required`, `Demand`,
`Workers`, …) with exactly **{TOTAL_INTERVALS}** rows, Monday 00:00 → Sunday 23:55
in 5-min steps.
""")

    up_files = st.file_uploader(
        "Workload files", type=["xlsx", "xlsm", "csv"],
        accept_multiple_files=True, key="wl_files")

    @st.cache_data(show_spinner=False)
    def _read_cached(name: str, data: bytes):
        return read_workload_file(_NamedBytes(data, name))

    class _NamedBytes(io.BytesIO):
        def __init__(self, data, name):
            super().__init__(data)
            self.name = name

    parsed = []
    for f in up_files or []:
        try:
            parsed.append(_read_cached(f.name, f.getvalue()))
        except Exception as exc:
            st.error(f"{f.name}: {exc}")

    if parsed:
        rows, curve_refs = [], []
        for fi, wf in enumerate(parsed):
            for w in wf.warnings:
                st.warning(w)
            week = None
            if wf.kind == "groundstar":
                starts = wf.week_starts()
                meta_bits = [f"**{wf.file_name}**"]
                if wf.resource_type:
                    meta_bits.append(f"resource type `{wf.resource_type}`")
                if wf.meta.get("Curve Building Criteria"):
                    meta_bits.append(f"curves by *{wf.meta['Curve Building Criteria']}*")
                if starts:
                    labels = [f"{d:%a %d %b %Y} → {(d + pd.Timedelta(days=6)):%a %d %b}"
                              for d in starts]
                    if len(starts) > 1:
                        pick = st.selectbox(f"Week for {wf.file_name}", labels,
                                            key=f"wl_week_{fi}_{wf.file_name}")
                        week = starts[labels.index(pick)]
                    else:
                        week = starts[0]
                        meta_bits.append(f"week {labels[0]}")
                else:
                    meta_bits.append("no full Mon–Sun week — weekdays mapped from "
                                     "their first occurrence")
                st.caption(" · ".join(meta_bits))
            default = {id(c) for c in wf.default_curves()}
            for c in wf.curves:
                d = curve_week(c, week)
                occ = c.name if c.role in ("Standard", "Plain") else (
                    wf.resource_type or c.name)
                rows.append({
                    "Use": id(c) in default,
                    "Occupation": occ,
                    "File": wf.file_name,
                    "Curve": c.name,
                    "Type": c.role,
                    "Peak": int(d.max()),
                    "Workload h": round(d.sum() / 12, 1),
                })
                curve_refs.append(d)

        _sig = "|".join(f"{wf.file_name}:{len(wf.curves)}" for wf in parsed)
        edited = st.data_editor(
            pd.DataFrame(rows), key=f"wl_table_{hash(_sig)}",
            hide_index=True, width="stretch",
            disabled=["File", "Curve", "Type", "Peak", "Workload h"],
            column_config={
                "Use": st.column_config.CheckboxColumn(width="small"),
                "Occupation": st.column_config.TextColumn(
                    help="Classification to plan for. Same name = curves summed."),
            })

        sel = [{"occupation": r["Occupation"], "demand": curve_refs[i]}
               for i, r in edited.iterrows() if r["Use"] and str(r["Occupation"]).strip()]
        combined = combine_by_occupation(sel)
        if combined:
            st.caption("Will load: " + ", ".join(
                f"**{n}** (peak {int(d.max())}, {d.sum() / 12:,.0f} h)"
                for n, d in combined.items()))
        if st.button("Load workload", key="btn_load", type="primary",
                     disabled=not combined):
            if len(combined) > MAX_CURVES:
                st.error(f"{len(combined)} occupations selected — max is {MAX_CURVES}.")
            else:
                st.session_state["demands"] = list(combined.values())
                st.session_state["occ_names"] = list(combined.keys())
                st.session_state["_pending_occ"] = list(combined.keys())
                st.session_state["demand_source"] = "import"
                st.rerun()
    if (st.session_state.get("demand_source") == "import"
            and st.session_state.get("demands") is not None):
        st.success("Loaded: " + ", ".join(st.session_state["occ_names"]))

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
        st.session_state["demand_source"] = "sample"
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
    st.plotly_chart(fig_d, width="stretch")


# ── Solve ────────────────────────────────────────────────────────────────────
if demands is not None:
    # ── Daily peak demand reference ──────────────────────────────────────
    with st.expander("📊 Daily peak demand by occupation", expanded=True):
        peak_rows = []
        for i, (d, name) in enumerate(zip(demands, stored_names)):
            for day in range(7):
                s = day * INTERVALS_PER_DAY
                e = s + INTERVALS_PER_DAY
                peak_val = int(d[s:e].max())
                peak_rows.append({
                    "Occupation": name,
                    "Day": DAY_NAMES[day],
                    "Peak Demand": peak_val,
                })
        if peak_rows:
            st.dataframe(pd.DataFrame(peak_rows),
                         width="stretch", hide_index=True)

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
    if target_wh > max_wh:
        _warnings.append(f"Target hours {target_wh:g} > max {max_wh:g} — target can't be reached")
    for w in _warnings:
        st.warning(w)

    if st.button("▶ Run Optimiser", type="primary"):
        log_box = st.empty()
        log_lines: list[str] = []

        def _cb(msg):
            log_lines.append(msg)
            log_box.code("\n".join(log_lines), language="text")

        with st.spinner("Solving …"):
            try:
                result = solve_multi(demands, stored_names, params, callback=_cb)
            except MemoryError:
                st.error("Out of memory — try reducing the number of occupations or shortening the time limit.")
                st.stop()
            except Exception as exc:
                st.error(f"Solver crashed: {exc}")
                st.stop()

        rosters = {}
        if build_roster and result.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
            with st.spinner("Building roster …"):
                for occ in result.occupations:
                    sl = slots_from_phase1(occ.phase1)
                    try:
                        rosters[occ.name] = (solve_roster(sl, roster_params, occ.name,
                                                          callback=_cb), sl)
                    except Exception as exc:
                        st.error(f"Roster for {occ.name} crashed: {exc}")
        st.session_state["rosters"] = rosters

        # stash previous before overwriting
        if "result" in st.session_state:
            st.session_state["prev_result"] = st.session_state["result"]
            st.session_state["prev_result_params"] = st.session_state.get("result_params")
        st.session_state["result"] = result
        st.session_state["result_params"] = params


# ── Roster rendering ─────────────────────────────────────────────────────────
def _roster_table(r):
    rows = []
    for q in r.people:
        row = {"Person": q.pid, "Type": q.kind}
        for d, t in enumerate(q.shifts):
            row[DAY_NAMES[d][:3]] = t.code if t is not None else "off"
        row["Shifts"] = q.n_shifts
        row["Paid h"] = round(q.paid_hours, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def _render_roster(rosters, result):
    if not rosters:
        return
    st.subheader("👥 Roster — people needed")
    rp = next(iter(rosters.values()))[0].params
    tot = sum(r.headcount for r, _ in rosters.values())
    lb = sum(r.lower_bound for r, _ in rosters.values())
    allp = [q for r, _ in rosters.values() for q in r.people]
    n1 = sum(1 for q in allp if q.kind == "FT 1-off")
    n2 = sum(1 for q in allp if q.kind == "FT 2-off")
    npt = sum(1 for q in allp if q.kind == "PT")
    hrs = [q.paid_hours for q in allp] or [0]
    c = st.columns(6)
    c[0].metric("People", tot, help="Distinct people to work every planned shift.")
    c[1].metric("Lower bound", lb, help="No roster can use fewer people: max of "
                "hours ÷ max hours, shifts ÷ shifts per person, busiest day, and "
                "consecutive days under the rest rule (late finishers can't take "
                "next morning's shifts). People = bound means the roster is optimal.")
    c[2].metric("FT 1-off / 2-off", f"{n1} / {n2}")
    c[3].metric("Part-time", npt)
    c[4].metric("Avg paid h", f"{np.mean(hrs):.1f}")
    c[5].metric("Paid h range", f"{min(hrs):.1f}–{max(hrs):.1f}")

    for name, (r, sl) in rosters.items():
        label = (f"{name}: {r.headcount} people (lower bound {r.lower_bound}) · "
                 f"{r.status} · {r.elapsed_sec:.0f}s")
        with st.expander(label, expanded=len(rosters) == 1):
            if not r.people:
                st.error(f"No roster: {r.status}. {r.message}")
                continue
            errs = verify_roster(r, sl)
            if errs:
                st.error("Roster check failed: " + "; ".join(errs[:5]))
            else:
                st.success(f"✅ Rules respected — rest ≥ {rp.min_rest_hours:g} h, "
                           f"1-off rest ≥ {rp.off_rest_hours:g} h, shift and hours limits.")
            lbs = ", ".join(f"{k} {v}" for k, v in r.lower_bounds.items())
            _gap = r.headcount - r.lower_bound
            st.caption(f"Lower bounds: {lbs}. " + (
                "Roster is at the bound — proven minimum." if _gap == 0 else
                f"Gap to bound: {_gap} — a longer roster time limit may close it; "
                f"the 1-off 24 h rule can also keep it above the bound."))
            for msg in precheck(sl, rp):
                st.warning(msg)
            under = r.under_min()
            if under:
                avg_paid = sum(t.paid * t.count for t in sl) / max(1, sum(t.count for t in sl)) / 60
                st.info(
                    f"{len(under)} of {r.headcount} people are below their minimum hours. "
                    f"The plan's shifts average {avg_paid:.1f} paid h, so 5 shifts ≈ "
                    f"{5 * avg_paid:.1f} h and 6 shifts ≈ {6 * avg_paid:.1f} h. "
                    "Raise *Min shift (h)* so the plan uses longer shifts, or use "
                    "fewer days off.")
            hrs_o = [q.paid_hours for q in r.people]
            fig = go.Figure(go.Histogram(x=hrs_o, xbins=dict(size=1),
                                         marker_color="#1f77b4"))
            lo, hi = rp.min_hours, rp.max_hours
            fig.add_vrect(x0=lo, x1=hi, fillcolor="green", opacity=0.08, line_width=0)
            fig.add_vline(x=rp.target_hours, line_dash="dash", line_color="green")
            fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10),
                              title="Paid hours per person", xaxis_title="paid h / week",
                              yaxis_title="people", bargap=0.05)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(_roster_table(r), hide_index=True, width="stretch",
                         height=min(600, 38 + 35 * len(r.people)))


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

    _paid_h = sum(t.paid * t.count for o in result.occupations
                  for t in slots_from_phase1(o.phase1)) / 60
    _tgt = roster_params.target_hours

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Status", cp1.status)
    c2.metric("Active shifts", n_active)
    c3.metric("Total worker-h", f"{total_wh:,.0f}",
              help=f"On-shift hours incl. breaks. Paid (after breaks): {_paid_h:,.0f} h")
    c4.metric("Peak daily starts", total_headcount,
              help="Most shifts starting on one day — a floor for headcount, not the "
                   "number of people. See the roster below for people.")
    c5.metric("Peak Simultaneous", peak_sim)
    c6.metric(f"FTE (paid ÷ {_tgt:g})", f"{_paid_h / _tgt:.1f}",
              help="Paid hours (shift length minus unpaid breaks) ÷ target weekly hours.")
    c7.metric("Solve time", f"{cp1.elapsed_sec:.1f}s")

    _rb1, _rb2 = st.columns([1, 3])
    if _rb1.button("🔁 Rebuild roster", help="Re-run only the roster step with the "
                   "current roster settings (keeps this shift plan)."):
        _new = {}
        _rlog = _rb2.empty()
        with st.spinner("Building roster …"):
            for occ in result.occupations:
                _sl = slots_from_phase1(occ.phase1)
                _new[occ.name] = (solve_roster(_sl, roster_params, occ.name,
                                               callback=lambda m: _rlog.caption(m)), _sl)
        st.session_state["rosters"] = _new
        _rlog.empty()
    _render_roster(st.session_state.get("rosters") or {}, result)

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

    # ── Operational buffer check (prep / exit / handover) ───────────────
    _rp = result.params
    if result.combined_required is not None:
        _bv = buffer_violations(result)
        if _bv["intervals"] == 0:
            st.success(
                f"✅ Operational buffers respected — prep {_rp.prep_lead_min} min "
                f"before ramp-ups, stay {_rp.exit_lag_min} min after ramp-downs, "
                f"{_rp.handover_overlap_min} min handover overlap.")
        else:
            st.warning(
                f"Operational buffers not fully met: {_bv['intervals']} intervals "
                f"({_bv['short_wh']:.1f} worker-hours short of prep/exit/handover "
                f"requirement).")

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
                             width="stretch", hide_index=True)
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
            _dc3.metric("Peak daily starts", total_headcount,
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

    # per-occupation headcount metrics
    if n_occ > 1:
        occ_cols = st.columns(n_occ)
        for i, occ in enumerate(result.occupations):
            with occ_cols[i]:
                occ_wh = occ.phase1.total_worker_intervals / INTERVALS_PER_HOUR
                occ_hc = max_headcount(occ.phase1)
                st.metric(f"{occ.name} peak daily starts", occ_hc)
                st.metric(f"{occ.name} worker-h", f"{occ_wh:,.0f}")

    # combined daily headcount vs minimum table (entry-based)
    if _rp.min_headcount_per_day:
        with st.expander("👥 Combined daily headcount vs minimum", expanded=False):
            rows = []
            daily_entries = daily_entry_headcount(cp1)
            for day in range(7):
                min_req = _rp.min_headcount_per_day[day]
                if min_req <= 0:
                    continue
                actual = daily_entries[day]
                rows.append({
                    "Day": DAY_NAMES[day],
                    "Workers entering shifts": actual,
                    "Required Min": min_req,
                    "Status": "✅" if actual >= min_req else "❌ VIOLATION",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows),
                             width="stretch", hide_index=True)
            else:
                st.caption("No combined minimum headcount requirements were set.")

    # per-occupation daily headcount vs limit table (entry-based)
    if _rp.occ_max_headcount_per_day:
        with st.expander("👤 Per-occupation daily headcount vs limit", expanded=False):
            rows = []
            for i, occ in enumerate(result.occupations):
                occ_limits = _rp.occ_max_headcount_per_day[i]
                if occ_limits is None:
                    continue
                daily = daily_entry_headcount(occ.phase1)
                for day in range(7):
                    limit = occ_limits[day]
                    if limit <= 0:
                        continue
                    actual = daily[day]
                    rows.append({
                        "Occupation": occ.name,
                        "Day": DAY_NAMES[day],
                        "Workers entering shifts": actual,
                        "Limit": limit,
                        "Status": "✅" if actual <= limit else "❌ VIOLATION",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows),
                             width="stretch", hide_index=True)
            else:
                st.caption("No per-occupation headcount limits were set.")

    # per-occupation shift-code headcount table
    if _rp.occ_headcount_per_shift_code:
        with st.expander("🕐 Shift-type headcount vs limits", expanded=False):
            rows = []
            for i, occ in enumerate(result.occupations):
                sc_limits = _rp.occ_headcount_per_shift_code[i]
                if sc_limits is None:
                    continue
                for code, spec in sc_limits.items():
                    if isinstance(spec, list):
                        max_limits = spec
                        min_limits = None
                    else:
                        max_limits = spec.get("max")
                        min_limits = spec.get("min")
                    # Compute actual workers per shift-code per day
                    daily_workers = [0] * 7
                    for s, cnt in zip(occ.phase1.shifts, occ.phase1.counts):
                        if s.shift_code == code and cnt > 0:
                            daily_workers[s.day] += cnt
                    for day in range(7):
                        actual = daily_workers[day]
                        max_l = max_limits[day] if max_limits else 0
                        min_l = min_limits[day] if min_limits else 0
                        if max_l > 0 or min_l > 0:
                            status = "✅"
                            if max_l > 0 and actual > max_l:
                                status = "❌ VIOLATION (max)"
                            if min_l > 0 and actual < min_l:
                                status = "❌ VIOLATION (min)"
                            rows.append({
                                "Occupation": occ.name,
                                "Shift Code": code,
                                "Day": DAY_NAMES[day],
                                "Workers": actual,
                                "Max Limit": max_l if max_l > 0 else "—",
                                "Min Limit": min_l if min_l > 0 else "—",
                                "Status": status,
                            })
            if rows:
                st.dataframe(pd.DataFrame(rows),
                             width="stretch", hide_index=True)
            else:
                st.caption("No shift-type headcount constraints were active.")

    # per-occupation daily headcount vs minimum table (entry-based)
    if _rp.occ_min_headcount_per_day:
        with st.expander("👤 Per-occupation daily headcount vs minimum", expanded=False):
            rows = []
            for i, occ in enumerate(result.occupations):
                occ_mins = _rp.occ_min_headcount_per_day[i]
                if occ_mins is None:
                    continue
                daily = daily_entry_headcount(occ.phase1)
                for day in range(7):
                    min_req = occ_mins[day]
                    if min_req <= 0:
                        continue
                    actual = daily[day]
                    rows.append({
                        "Occupation": occ.name,
                        "Day": DAY_NAMES[day],
                        "Workers entering shifts": actual,
                        "Required Min": min_req,
                        "Status": "✅" if actual >= min_req else "❌ VIOLATION",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows),
                             width="stretch", hide_index=True)
            else:
                st.caption("No per-occupation minimum headcount requirements were set.")

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
    if result.combined_required is not None:
        fig_w.add_trace(go.Scatter(
            x=x_vals, y=result.combined_required, mode="lines",
            name="Required incl. prep/exit",
            line=dict(color="orange", width=1.5, dash="dot"),
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
    st.plotly_chart(fig_w, width="stretch")

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
    if result.combined_required is not None:
        fig_day.add_trace(go.Scatter(
            x=day_x, y=result.combined_required[day_start:day_end],
            mode="lines", name="Required incl. prep/exit",
            line=dict(color="orange", width=1.5, dash="dot"),
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
    st.plotly_chart(fig_day, width="stretch")

    # ── Shift Types ──────────────────────────────────────────────────────
    st.subheader("Shift types")
    all_st = []
    for occ in result.occupations:
        st_df = shift_type_summary(occ.phase1, occ.name)
        all_st.append(st_df)
    if all_st:
        combined_st = pd.concat(all_st, ignore_index=True)
        st.dataframe(combined_st, width="stretch", hide_index=True)

    # ── XLSX download ────────────────────────────────────────────────────
    st.subheader("📥 Download report")
    xlsx_bytes = build_weekly_report_xlsx(result)
    _rost = st.session_state.get("rosters") or {}
    if _rost:
        import openpyxl as _oxl
        _wb = _oxl.load_workbook(io.BytesIO(xlsx_bytes))
        for _name, (_r, _sl) in _rost.items():
            if not _r.people:
                continue
            _ws = _wb.create_sheet(f"Roster {_name}"[:31])
            _df = _roster_table(_r)
            _ws.append(list(_df.columns))
            for _row in _df.itertuples(index=False):
                _ws.append(list(_row))
            _ws.freeze_panes = "C2"
        _buf = io.BytesIO(); _wb.save(_buf); xlsx_bytes = _buf.getvalue()
    st.download_button(
        "⬇ Download XLSX report",
        data=xlsx_bytes,
        file_name="simplex_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
