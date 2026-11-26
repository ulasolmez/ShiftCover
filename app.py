"""
Simplex (task-based) – Streamlit front-end
==========================================
Same layout and shift rules as the curve-based Simplex, but the input is a
list of tasks (from a flight schedule + task rules, a task list, or a
sample).  The optimiser picks each task's start time and the shifts
together, then checks that every task can be given to one worker.
"""

import dataclasses

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from solver import (
    INTERVALS_PER_DAY, INTERVALS_PER_HOUR, TOTAL_INTERVALS,
    DAY_NAMES, OCC_COLORS,
    list_possible_shift_codes, daily_entry_headcount, max_headcount,
)
from tasks import (
    default_rules, read_flights, read_task_list, tasks_from_flights,
    generate_sample_flights, flights_to_dataframe, tasks_to_dataframe,
    task_list_csv, flights_csv, occupations_of, earliest_load, RULE_COLUMNS,
)
from groundstar_rules import (
    decode_text, is_inform_flights, read_inform_flights, leg_week_starts,
    build_events, events_dataframe, events_to_flights, flights_to_events,
    read_engage_rules, rules_summary, referenced_groups, default_actype_groups,
    parse_flight_number_groups, default_fn_text, engage_tasks,
)
from task_solver import (
    TaskParams, TaskResult, solve_tasks, utilisation, task_hours,
    assignment_dataframe, workers_dataframe, build_task_report_xlsx,
    shift_types,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Simplex Tasks", layout="wide")
st.title("🕐 Simplex – Task-based Shift Optimiser")

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

DAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
x5 = list(range(TOTAL_INTERVALS))
tasks = st.session_state.get("tasks")
occ_names: list = st.session_state.get("occ_names") or []
n_occ_in = len(occ_names)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")
    if occ_names:
        st.caption("Occupations (from the tasks): " + ", ".join(f"**{n}**" for n in occ_names))
    else:
        st.caption("Occupations come from the tasks — load or generate tasks first.")

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
            "Allowed times (start & end)", options=_ALL_SLOTS, key="slot_filter")
        allowed_slot_minutes = (
            [int(t[:2]) * 60 + int(t[3:]) for t in allowed_slots_sel]
            if len(allowed_slots_sel) < 48 else None
        )

    _preview_params = TaskParams(
        min_shift_hours=min_shift, max_shift_hours=max_shift,
        shift_start_granularity_min=30, shift_duration_step_min=30,
        exclude_night_shifts=no_night, allowed_slot_minutes=allowed_slot_minutes,
    )
    _all_codes = list_possible_shift_codes(_preview_params)

    with st.expander(f"✅ Force include / exclude  ({len(_all_codes)} patterns)", expanded=False):
        force_incl = st.multiselect("Must include (solver will always use these)",
                                    options=_all_codes, default=[], key="force_incl")
        force_excl = st.multiselect("Must exclude (solver will never use these)",
                                    options=[c for c in _all_codes if c not in set(force_incl)],
                                    default=[], key="force_excl")

    with st.expander("📅 Entry / Exit limits per day", expanded=False):
        st.caption("Max distinct start times (entries) and end times (exits) "
                   "per day. 0 = unlimited.")
        max_entries, max_exits = [], []
        for d in range(7):
            ce, cx = st.columns(2)
            max_entries.append(ce.number_input(f"{DAY_SHORT[d]} entries", 0, 48, 0, key=f"ent_{d}"))
            max_exits.append(cx.number_input(f"{DAY_SHORT[d]} exits", 0, 48, 0, key=f"ext_{d}"))
    use_entries = any(v > 0 for v in max_entries)
    use_exits = any(v > 0 for v in max_exits)

    with st.expander("👥 Headcount limits per day (combined)", expanded=False):
        st.caption("Max and min workers entering shifts (all occupations combined) "
                   "per day. 0 = unlimited / no minimum.")
        max_hc, min_hc = [], []
        for d in range(7):
            cm, cn = st.columns(2)
            max_hc.append(cm.number_input(f"{DAY_SHORT[d]} max", 0, 500, 0, key=f"hc_{d}"))
            min_hc.append(cn.number_input(f"{DAY_SHORT[d]} min", 0, 500, 0, key=f"min_hc_{d}"))
    use_hc = any(v > 0 for v in max_hc)
    use_min_hc = any(v > 0 for v in min_hc)

    def _per_occ_days(title, key, caption):
        out = []
        with st.expander(title, expanded=False):
            st.caption(caption)
            if not occ_names:
                st.caption("Load tasks first.")
            for i, name in enumerate(occ_names):
                st.markdown(f"**{name}**")
                cols = st.columns(7)
                vals = []
                for d in range(7):
                    with cols[d]:
                        st.caption(DAY_SHORT[d])
                        vals.append(st.number_input(f"##{key}_{name}_{d}", 0, 500, 0,
                                                    key=f"{key}_{name}_{d}",
                                                    label_visibility="collapsed"))
                out.append(vals if any(v > 0 for v in vals) else None)
        return out

    occ_max_hc = _per_occ_days("👤 Max headcount per occupation per day", "occ_hc",
                               "Workers of each occupation that may enter a shift per day. "
                               "0 = unlimited.")
    occ_min_hc = _per_occ_days("👤 Min headcount per occupation per day", "occ_min_hc",
                               "Minimum workers of each occupation entering a shift per day. "
                               "0 = no minimum.")

    occ_sc_hc: list = []
    with st.expander("🕐 Shift-type headcount limits", expanded=False):
        st.caption("Daily max/min workers per shift code per occupation. 0 = unlimited / no minimum.")
        if not occ_names:
            st.caption("Load tasks first.")
        for i, name in enumerate(occ_names):
            st.markdown(f"**{name}**")
            codes = st.multiselect(f"Shift codes to constrain for {name}", options=_all_codes,
                                   default=[], key=f"sc_hc_codes_{name}")
            sc = {}
            for code in codes:
                st.caption(f"  {code}")
                hdr = st.columns([0.4] + [1] * 7)
                for d in range(7):
                    hdr[d + 1].caption(DAY_SHORT[d])
                mxr, mnr = st.columns([0.4] + [1] * 7), st.columns([0.4] + [1] * 7)
                mxr[0].caption("Max")
                mnr[0].caption("Min")
                mx = [mxr[d + 1].number_input("max", 0, 500, 0, label_visibility="collapsed",
                                              key=f"sc_max_{name}_{code}_{d}") for d in range(7)]
                mn = [mnr[d + 1].number_input("min", 0, 500, 0, label_visibility="collapsed",
                                              key=f"sc_min_{name}_{code}_{d}") for d in range(7)]
                spec = {}
                if any(mx):
                    spec["max"] = mx
                if any(mn):
                    spec["min"] = mn
                if spec:
                    sc[code] = spec
            occ_sc_hc.append(sc or None)

    with st.expander("🧭 Task work rules", expanded=True):
        st.caption("How a worker's time on shift can be used for tasks.")
        briefing = st.number_input("Briefing at shift start (min)", 0, 60, 15, 5,
                                   help="No tasks in the first minutes of a shift "
                                        "(briefing, equipment pick-up).")
        closeout = st.number_input("Close-out at shift end (min)", 0, 60, 10, 5,
                                   help="No tasks in the last minutes of a shift "
                                        "(equipment return, handover).")
        travel = st.number_input("Travel between stands (min)", 0, 30, 5, 5,
                                 help="Walking/driving time before each task; "
                                      "none between two tasks at the same stand.")
        breaks = st.checkbox("Plan breaks (30 min from 4 h, 60 min from 8 h)", value=True,
                             help="The break may sit anywhere in the middle half of the shift; "
                                  "no task during the break.")
        max_opts = st.slider("Start times tried per task", 1, 25, 13,
                             help="Tasks with a time window get this many evenly spread start "
                                  "options. 1 = every task starts as early as it can.")

    with st.expander("🔧 Solver", expanded=False):
        time_limit = st.number_input("Time limit (s)", 20, 600, 120, 10)
        t_penalty = st.number_input("Transition penalty", 0, 500, 50, 10)
        rounds = st.number_input("Assignment-check rounds", 0, 5, 3, 1,
                                 help="If a task cannot be given to one worker, the shift "
                                      "plan is solved again with an extra rule for it.")

params = TaskParams(
    min_shift_hours=min_shift, max_shift_hours=max_shift,
    shift_start_granularity_min=30, shift_duration_step_min=30,
    max_unique_shifts=max_unique, transition_penalty=t_penalty,
    solver_time_limit_sec=time_limit,
    max_entries_per_day=max_entries if use_entries else None,
    max_exits_per_day=max_exits if use_exits else None,
    max_headcount_per_day=max_hc if use_hc else None,
    min_headcount_per_day=min_hc if use_min_hc else None,
    occ_max_headcount_per_day=occ_max_hc if any(v is not None for v in occ_max_hc) else None,
    occ_min_headcount_per_day=occ_min_hc if any(v is not None for v in occ_min_hc) else None,
    occ_headcount_per_shift_code=occ_sc_hc if any(v is not None for v in occ_sc_hc) else None,
    exclude_night_shifts=no_night, circular_week=circular,
    force_include_shifts=force_incl or None, force_exclude_shifts=force_excl or None,
    allowed_slot_minutes=allowed_slot_minutes,
    briefing_min=int(briefing), closeout_min=int(closeout), travel_min=int(travel),
    breaks=breaks, assign_rounds=int(rounds), max_task_options=int(max_opts),
)


def _set_tasks(new_tasks, source, flights=None):
    if not source.startswith("flights"):
        st.session_state.pop("build_info", None)
    st.session_state["tasks"] = new_tasks
    st.session_state["flights"] = flights
    st.session_state["occ_names"] = occupations_of(new_tasks)
    st.session_state["task_source"] = source
    st.rerun()


# ── Task input ───────────────────────────────────────────────────────────────
st.header("📂 Task input")

if "rules" not in st.session_state:
    st.session_state["rules"] = default_rules()
with st.expander("📋 Task rules (relative to STA / STD)", expanded=False):
    st.caption("Used for **Flights + task rules** and **Generate sample**. "
               "A task may start at Ref + From at the earliest and must be finished "
               "by Ref + To. When To − From equals the duration, the task is fixed. "
               "Staff depends on the aircraft category (NB = narrow-body, WB = wide-body). "
               "Add rows for more tasks; untick **Use** to leave one out.")
    rules = st.data_editor(
        st.session_state["rules"], key="rules_editor", num_rows="dynamic",
        hide_index=True, width="stretch",
        column_config={
            "Use": st.column_config.CheckboxColumn(width="small"),
            "Ref": st.column_config.SelectboxColumn(options=["STA", "STD"], width="small"),
            "From (min)": st.column_config.NumberColumn(step=5),
            "To (min)": st.column_config.NumberColumn(step=5),
            "Duration (min)": st.column_config.NumberColumn(min_value=5, step=5),
            "Staff NB": st.column_config.NumberColumn(min_value=0, step=1),
            "Staff WB": st.column_config.NumberColumn(min_value=0, step=1),
        })
    rules = rules.dropna(subset=["Occupation", "Task", "Ref"])[RULE_COLUMNS]

tab_fl, tab_tl, tab_sample = st.tabs(["✈ Flights + task rules", "⬆ Task list", "🎲 Generate sample"])


@st.cache_data(show_spinner=False)
def _read_flights_cached(name, data, week):
    return read_flights(data, name, week)


@st.cache_data(show_spinner=False)
def _read_tasks_cached(name, data, week, circ):
    return read_task_list(data, name, week, circ)


@st.cache_data(show_spinner=False)
def _read_inform_cached(data):
    return read_inform_flights(data)


@st.cache_data(show_spinner=False)
def _read_rules_cached(data):
    return read_engage_rules(data)


with tab_fl:
    st.info("Upload a **flight schedule** – the INFORM flight export (.txt) or a table "
            "(.xlsx/.csv) – and, optionally, a **GroundStar engage-rule export**. "
            "Without a rule export the task rules table above is used.")
    with st.expander("❓ Supported files"):
        st.markdown("""
**INFORM flight export (.txt)** – semicolon-separated, one row per leg, no header
(as exported from INFORM). Arrival and departure of a rotation are linked through
the linked-flight columns. Used: date, airline, flight number, A/D, linked flight,
domestic/international, scheduled/charter, passenger/cargo, registration, aircraft
type, stand, STA/STD, pax, seats, cargo + mail kg. If the file spans more than one
Monday–Sunday week you pick the week.

**Flight table (.xlsx/.csv)**

| Column | Examples | |
|---|---|---|
| Flight | `Flight`, `Flight no`, `Arrival flight` | optional |
| Day | `Mon`, `Monday`, `1`–`7` or a date | needed if STA/STD are clock times |
| STA / STD | `07:35`, `0735` or a date-time | at least one |
| Category | `NB`/`WB`, or an aircraft type (`A321`, `B77W`) | optional (NB) |
| Stand | `A12` | optional, used for travel time |

**GroundStar engage-rule export** (`PBEngageRule`, .csv/.txt) – one file with
a table per resource type (DIY_RAMP, DIY_TRF, …). Each table may have its own
columns in any order and the tables may come in any order; UTF-16, UTF-8 and
Windows-1254 files are read. Rule phase 1 = arrival (relative to STA), 0 =
departure (relative to STD), 2 = ground time (STA → STD). Within one engage
group only the matching rule(s) with the highest priority are used, so
airline-specific rules override the defaults.
""")
    f_up = st.file_uploader("Flight schedule", type=["txt", "xlsx", "xlsm", "csv"],
                            key="fl_file")
    r_up = st.file_uploader("Engage-rule export (optional)", type=["csv", "txt", "tsv"],
                            key="rule_file")

    events, fl = None, None
    if f_up is not None:
        try:
            raw = f_up.getvalue()
            if f_up.name.lower().endswith((".txt", ".csv")) and is_inform_flights(decode_text(raw)):
                legs, warns = _read_inform_cached(raw)
                weeks = leg_week_starts(legs)
                labels = [f"{w:%a %d %b %Y} → {(w + pd.Timedelta(days=6)):%a %d %b}" for w in weeks]
                week = weeks[0]
                if len(weeks) > 1:
                    pick = st.selectbox("Week", labels, key="fl_week_inf")
                    week = weeks[labels.index(pick)]
                events = build_events(legs, week)
                for w in warns:
                    st.warning(w)
                n_rot = sum(1 for e in events if e.chained)
                st.caption(f"INFORM export: **{len(legs)}** legs in the file · week "
                           f"{labels[weeks.index(week)]}: **{n_rot}** rotations, "
                           f"**{len(events) - n_rot}** single legs")
                st.dataframe(events_dataframe(events), hide_index=True, width="stretch",
                             height=220)
                fl = events_to_flights(events)
            else:
                fl, warns, weeks = _read_flights_cached(f_up.name, raw, None)
                if len(weeks) > 1:
                    labels = [f"{w:%a %d %b %Y}" for w in weeks]
                    pick = st.selectbox("Week", labels, key="fl_week")
                    fl, warns, _ = _read_flights_cached(f_up.name, raw,
                                                        weeks[labels.index(pick)])
                for w in warns:
                    st.warning(w)
                st.caption(f"**{len(fl)}** flights read")
                st.dataframe(flights_to_dataframe(fl).head(200), hide_index=True,
                             width="stretch", height=220)
        except Exception as exc:
            st.error(f"{f_up.name}: {exc}")
            fl = None

    rules_df = None
    if r_up is not None:
        try:
            rules_df, rwarns = _read_rules_cached(r_up.getvalue())
            for w in rwarns:
                st.warning(w)
        except Exception as exc:
            st.error(f"{r_up.name}: {exc}")

    if rules_df is not None:
        st.markdown(f"**Engage rules** – {len(rules_df)} rules in "
                    f"{rules_df['Tab'].nunique()} tables")
        summ = rules_summary(rules_df)
        st.dataframe(summ, hide_index=True, width="stretch")
        deps = list(summ["Department"])
        default_dep = [d for d in deps if "RAMP" in d.upper()][:1] or deps[:1]
        sel_deps = st.multiselect("Departments to plan", deps, default=default_dep,
                                  key="rule_deps")
        plan_by = st.radio("Plan per", ["Qualification", "Department"], horizontal=True,
                           key="rule_plan_by",
                           help="Qualification: one plan group per mandQual1 (e.g. "
                                "DIY_W, DIY_DA). Department: all qualifications of a "
                                "resource type pooled.")
        grp_names = referenced_groups(rules_df, ["acTypeGroup", "notAcTypeGroup"])
        ac_groups = {}
        if events is not None and grp_names:
            with st.expander("✈ Aircraft type groups", expanded=False):
                st.caption("Which aircraft types belong to the groups used in the rules "
                           "(acTypeGroup / notAcTypeGroup). Defaults are a guess — "
                           "check them against GroundStar base data.")
                legs_all = [l for e in events for l in (e.arr, e.dep) if l is not None]
                ac_edit = st.data_editor(default_actype_groups(legs_all, grp_names),
                                         key="ac_groups_editor", hide_index=True,
                                         width="stretch",
                                         disabled=["AC type", "Legs", "Seats"])
                ac_groups = {r["AC type"]: {g for g in grp_names if bool(r[g])}
                             for _, r in ac_edit.iterrows()}
        fn_names = referenced_groups(rules_df[rules_df["ResourceType"].isin(sel_deps)],
                                     ["flightNumberGroup"])
        fn_text = default_fn_text(fn_names)
        if fn_names:
            with st.expander("🔢 Flight number groups", expanded=False):
                st.caption("Rules use these flight-number groups: "
                           + ", ".join(f"`{n}`" for n in fn_names)
                           + ". One line per group: an airline code for all its flights "
                             "(`THYA = TKJ` – AJet's old THY group, today TKJ) or flight "
                             "numbers (`THY2600-2699, THY2805`). A rule with a group covers "
                             "the group's flights whatever its airline column says. "
                             "Undefined groups match no flight.")
                fn_text = st.text_area("Groups", default_fn_text(fn_names),
                                       key="fn_groups", label_visibility="collapsed")
        if events is None and fl:
            st.caption("With a flight table (not the INFORM export) pax, domestic/"
                       "international and aircraft-group filters cannot be checked; "
                       "rules that need them will not match.")

    can_build = bool(fl) and (rules_df is None or bool(st.session_state.get("rule_deps")))
    if f_up is not None and st.button("Build tasks", key="btn_build", type="primary",
                                      disabled=not can_build):
        if rules_df is not None:
            evs = events if events is not None else flights_to_events(fl)
            er = engage_tasks(evs, rules_df, sel_deps,
                              "qualification" if plan_by == "Qualification" else "department",
                              ac_groups, parse_flight_number_groups(fn_text), circular)
            if er.tasks:
                st.session_state["build_info"] = {"warnings": er.warnings,
                                                  "summary": er.summary,
                                                  "rule_use": er.rule_use}
                _set_tasks(er.tasks, "flights + engage rules", fl)
            else:
                for w in er.warnings:
                    st.warning(w)
                st.error("No tasks — no rule of the chosen departments matches these flights.")
        else:
            new, w2 = tasks_from_flights(fl, rules, circular)
            if new:
                st.session_state["build_info"] = {"warnings": w2}
                _set_tasks(new, "flights + task rules", fl)
            else:
                for w in w2:
                    st.warning(w)
                st.error("No tasks — check the task rules (Use ticked, staff > 0).")

with tab_tl:
    st.info("Upload a **task list** (.xlsx/.csv) with one row per task, e.g. a GroundStar "
            "task export.")
    with st.expander("❓ Supported columns"):
        st.markdown("""
| Column | Examples | |
|---|---|---|
| Task | `Task`, `Name`, `Activity` | optional |
| Occupation | `Qualification`, `Skill`, `Resource`, `Department` | optional (one group) |
| Day | `Mon` or a date | needed if times are clock times |
| Earliest start | `Earliest start`, `Start`, `Begin` | required |
| Latest end | `Latest end`, `End`, `Finish` | required |
| Duration (min) | `Duration` | optional (= end − start, a fixed task) |
| Staff | `Staff`, `Quantity`, `Persons` | optional (1) |
| Flight, Location | `Flight`, `Stand` | optional |
""")
    t_up = st.file_uploader("Task list", type=["xlsx", "xlsm", "csv"], key="tl_file")
    if t_up is not None:
        try:
            tl, warns, weeks = _read_tasks_cached(t_up.name, t_up.getvalue(), None, circular)
            if len(weeks) > 1:
                labels = [f"{w:%a %d %b %Y}" for w in weeks]
                pick = st.selectbox("Week", labels, key="tl_week")
                tl, warns, _ = _read_tasks_cached(t_up.name, t_up.getvalue(),
                                                  weeks[labels.index(pick)], circular)
            for w in warns:
                st.warning(w)
            st.caption(f"**{len(tl)}** tasks read")
            st.dataframe(tasks_to_dataframe(tl).head(200), hide_index=True,
                         width="stretch", height=220)
            if st.button("Load tasks", key="btn_load_tasks", type="primary", disabled=not tl):
                _set_tasks(tl, "task list")
        except Exception as exc:
            st.error(f"{t_up.name}: {exc}")

with tab_sample:
    c1, c2, c3 = st.columns(3)
    fpd = c1.slider("Flights per weekday", 5, 60, 20,
                    help="Weekends get about 25 % fewer flights.")
    wb = c2.slider("Wide-body share (%)", 0, 50, 15)
    seed = c3.number_input("Random seed", 0, 9999, 42)
    st.caption("A hub-style week with arrival/departure banks. Tasks come from the "
               "task rules above (by default Ramp and Cleaning are ticked).")
    if st.button("🎲 Generate sample", key="btn_sample"):
        fl = generate_sample_flights(fpd, wb / 100, seed)
        new, w2 = tasks_from_flights(fl, rules, circular)
        for w in w2:
            st.warning(w)
        if new:
            _set_tasks(new, "sample", fl)
        else:
            st.error("No tasks — tick at least one task rule.")

# ── Task preview ─────────────────────────────────────────────────────────────
if tasks:
    src = st.session_state.get("task_source", "")
    flights = st.session_state.get("flights")
    st.success(f"Loaded {len(tasks):,} tasks ({src})")
    info = st.session_state.get("build_info") or {}
    for w in info.get("warnings", []):
        st.warning(w)
    if info.get("rule_use") is not None and not info["rule_use"].empty:
        with st.expander("📐 Rules used", expanded=False):
            st.dataframe(info["summary"], hide_index=True, width="stretch")
            st.dataframe(info["rule_use"], hide_index=True, width="stretch", height=260)
    st.subheader("Task preview")
    tot_h = sum(t.duration * t.staff for t in tasks) / 60
    flex = sum(1 for t in tasks if t.slack > 0)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Tasks", f"{len(tasks):,}")
    m2.metric("Flights", f"{len(flights):,}" if flights else "—")
    m3.metric("Task hours", f"{tot_h:,.0f}")
    m4.metric("Occupations", len(occ_names))
    m5.metric("With a time window", f"{100 * flex / len(tasks):.0f} %")

    fig_p = go.Figure()
    for i, name in enumerate(occ_names):
        fig_p.add_trace(go.Scatter(
            x=x5, y=earliest_load(tasks, name), mode="lines", name=name,
            stackgroup="one", line=dict(width=0.5, color=OCC_COLORS[i % len(OCC_COLORS)])))
    for d in range(1, 7):
        fig_p.add_vline(x=d * INTERVALS_PER_DAY, line_dash="dash", line_color="grey", opacity=0.4)
    fig_p.update_xaxes(tickvals=[d * INTERVALS_PER_DAY for d in range(7)], ticktext=DAY_NAMES)
    fig_p.update_layout(height=300, margin=dict(l=40, r=20, t=30, b=30),
                        legend=dict(orientation="h", y=1.12))
    st.caption("Staff needed if every task starts as early as it can (the optimiser "
               "moves tasks inside their windows to smooth this).")
    st.plotly_chart(fig_p, width="stretch")

    with st.expander("📊 Tasks by occupation", expanded=True):
        rows = []
        for name in occ_names:
            ts = [t for t in tasks if t.occupation == name]
            rows.append({
                "Occupation": name, "Tasks": len(ts),
                "Task hours": round(sum(t.duration * t.staff for t in ts) / 60, 1),
                "Avg duration (min)": round(np.mean([t.duration for t in ts]), 1),
                "Peak staff (earliest starts)": int(earliest_load(tasks, name).max()),
                "With a time window": sum(1 for t in ts if t.slack > 0),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("🔎 Task list", expanded=False):
        st.dataframe(tasks_to_dataframe(tasks), hide_index=True, width="stretch", height=300)
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Task list (CSV)", task_list_csv(tasks), "tasks.csv", "text/csv")
        if flights:
            d2.download_button("⬇ Flights (CSV)", flights_csv(flights), "flights.csv", "text/csv")

    if min_shift > max_shift:
        st.warning("Min shift hours > Max shift hours")

    if st.button("▶ Run Optimiser", type="primary"):
        log_box = st.empty()
        log_lines: list = []

        def _cb(msg):
            log_lines.append(msg)
            log_box.code("\n".join(log_lines), language="text")

        with st.spinner("Solving …"):
            try:
                res = solve_tasks(tasks, params, callback=_cb)
            except MemoryError:
                st.error("Out of memory — try fewer start times per task or a shorter week.")
                st.stop()
            except Exception as exc:
                st.error(f"Solver crashed: {exc}")
                st.stop()
        if "result" in st.session_state:
            st.session_state["prev_result"] = st.session_state["result"]
            st.session_state["prev_result_params"] = st.session_state.get("result_params")
        st.session_state["result"] = res
        st.session_state["result_params"] = params


# ── Results ──────────────────────────────────────────────────────────────────
res: TaskResult = st.session_state.get("result")

if res is not None and tasks:
    st.divider()
    st.header("📊 Results")
    result = res.multi
    cp1 = result.combined_phase1
    n_occ = len(result.occupations)

    if cp1.status not in ("OPTIMAL", "FEASIBLE"):
        st.error(
            f"**Solver status: {cp1.status}**  ({cp1.elapsed_sec:.1f}s)\n\n"
            "No shift plan was found with the current settings. Try:\n"
            "- a wider shift-duration range (lower min / raise max)\n"
            "- fewer entry / exit / headcount limits\n"
            "- unticking *Exclude night shifts* if tasks run at night\n"
            "- a longer time limit")
        st.stop()

    st.subheader("Shift plan")
    total_wh = cp1.total_worker_intervals / INTERVALS_PER_HOUR
    n_active = sum(1 for c in cp1.counts if c > 0)
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Status", cp1.status)
    c2.metric("Active shifts", n_active)
    c3.metric("Paid hours", f"{total_wh:,.0f}")
    c4.metric("Headcount", max_headcount(cp1), help="Most workers starting a shift on one day.")
    c5.metric("Peak on duty", int(cp1.coverage.max()))
    c6.metric("FTE (÷45)", f"{total_wh / 45:.1f}")
    c7.metric("Solve time", f"{cp1.elapsed_sec:.1f}s")

    staffed = len(res.tasks) - len(res.unassigned)
    util = utilisation(res)
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Tasks staffed", f"{staffed:,} / {len(res.tasks):,}")
    q2.metric("Task hours", f"{task_hours(res):,.0f}")
    q3.metric("Utilisation", f"{util:.1f} %", help="Task hours ÷ paid hours.")
    q4.metric("Workers", len(res.workers))
    q5.metric("Repairs", f"{res.extended_shifts} + {res.extra_workers}",
              help="Shifts lengthened + workers added by the assignment check.")

    _rp = result.params
    if not res.unassigned:
        st.success("✅ Every task is staffed: each one is done whole by one worker, with "
                   f"{_rp.travel_min} min travel between stands, {_rp.briefing_min} min "
                   f"briefing, {_rp.closeout_min} min close-out"
                   + (" and breaks planned." if _rp.breaks else "."))
    else:
        tmap = {t.tid: t for t in res.tasks}
        st.warning(f"{len(res.unassigned)} task(s) could not be staffed — no allowed shift "
                   "can hold them (e.g. at the very start or end of a non-circular week – turn on *Circular week*): "
                   + ", ".join(f"{tmap[i].name} {tmap[i].flight}" for i in res.unassigned[:10]))
    if res.extended_shifts or res.extra_workers:
        st.info(f"The assignment check lengthened **{res.extended_shifts}** shift(s) and added "
                f"**{res.extra_workers}** worker(s) so that every task fits whole in one shift. "
                "These changes are not re-checked against entry/exit or headcount limits.")

    # ── What changed vs previous solve ──────────────────────────────────
    prev: TaskResult = st.session_state.get("prev_result")
    if prev is not None and prev.status in ("OPTIMAL", "FEASIBLE"):
        with st.expander("🔄 What changed vs previous solve", expanded=False):
            pp, cp = st.session_state.get("prev_result_params"), st.session_state.get("result_params")
            rows = []
            if pp is not None and cp is not None:
                for f in dataclasses.fields(TaskParams):
                    if getattr(pp, f.name) != getattr(cp, f.name):
                        rows.append({"Parameter": f.name, "Previous": str(getattr(pp, f.name)),
                                     "Now": str(getattr(cp, f.name))})
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            else:
                st.caption("Settings unchanged.")
            pc = prev.multi.combined_phase1
            pwh = pc.total_worker_intervals / INTERVALS_PER_HOUR
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Paid hours", f"{total_wh:,.0f}", f"{total_wh - pwh:+,.0f}", delta_color="inverse")
            d2.metric("Active shifts", n_active, n_active - sum(1 for c in pc.counts if c > 0))
            d3.metric("Utilisation", f"{util:.1f} %", f"{util - utilisation(prev):+.1f} %")
            d4.metric("Workers", len(res.workers), len(res.workers) - len(prev.workers),
                      delta_color="inverse")

    # ── Limits check (entry-based headcount) ────────────────────────────
    limit_rows = []
    if _rp.max_headcount_per_day or _rp.min_headcount_per_day:
        daily = daily_entry_headcount(cp1)
        for d in range(7):
            mx = (_rp.max_headcount_per_day or [0] * 7)[d]
            mn = (_rp.min_headcount_per_day or [0] * 7)[d]
            if mx or mn:
                ok = (not mx or daily[d] <= mx) and (not mn or daily[d] >= mn)
                limit_rows.append({"Scope": "All", "Day": DAY_NAMES[d], "Workers": daily[d],
                                   "Max": mx or "—", "Min": mn or "—",
                                   "Status": "✅" if ok else "❌"})
    for i, occ in enumerate(result.occupations):
        mxl = (_rp.occ_max_headcount_per_day or [None] * n_occ)[i] if _rp.occ_max_headcount_per_day else None
        mnl = (_rp.occ_min_headcount_per_day or [None] * n_occ)[i] if _rp.occ_min_headcount_per_day else None
        if not (mxl or mnl):
            continue
        daily = daily_entry_headcount(occ.phase1)
        for d in range(7):
            mx = mxl[d] if mxl else 0
            mn = mnl[d] if mnl else 0
            if mx or mn:
                ok = (not mx or daily[d] <= mx) and (not mn or daily[d] >= mn)
                limit_rows.append({"Scope": occ.name, "Day": DAY_NAMES[d], "Workers": daily[d],
                                   "Max": mx or "—", "Min": mn or "—",
                                   "Status": "✅" if ok else "❌"})
    if limit_rows:
        with st.expander("👥 Headcount vs limits", expanded=False):
            st.dataframe(pd.DataFrame(limit_rows), hide_index=True, width="stretch")

    # ── Weekly coverage ─────────────────────────────────────────────────
    st.subheader("Weekly coverage")
    prod_total = sum(res.productive.values())
    fig_w = go.Figure()
    fig_w.add_trace(go.Scatter(x=x5, y=result.combined_demand, mode="lines", name="Task load",
                               line=dict(color="red", width=2),
                               fill="tozeroy", fillcolor="rgba(255,0,0,0.08)"))
    fig_w.add_trace(go.Scatter(x=x5, y=prod_total, mode="lines", name="Available for tasks",
                               line=dict(color="green", width=2),
                               fill="tozeroy", fillcolor="rgba(0,128,0,0.08)"))
    fig_w.add_trace(go.Scatter(x=x5, y=cp1.coverage, mode="lines", name="On shift",
                               line=dict(color="grey", width=1, dash="dot")))
    if n_occ > 1:
        for i, occ in enumerate(result.occupations):
            clr = OCC_COLORS[i % len(OCC_COLORS)]
            fig_w.add_trace(go.Scatter(x=x5, y=occ.demand, mode="lines", name=f"{occ.name} load",
                                       line=dict(color=clr, width=1, dash="dash"),
                                       visible="legendonly"))
            fig_w.add_trace(go.Scatter(x=x5, y=res.productive[occ.name], mode="lines",
                                       name=f"{occ.name} available", line=dict(color=clr, width=1),
                                       visible="legendonly"))
    for d in range(1, 7):
        fig_w.add_vline(x=d * INTERVALS_PER_DAY, line_dash="dash", line_color="grey", opacity=0.4)
    fig_w.update_xaxes(tickvals=[d * INTERVALS_PER_DAY for d in range(7)], ticktext=DAY_NAMES)
    fig_w.update_layout(height=400, margin=dict(l=40, r=20, t=30, b=30),
                        legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig_w, width="stretch")
    st.caption("Task load includes the walk to the stand. *Available for tasks* is staff on "
               "shift minus briefing, close-out and breaks.")

    # ── Daily view: coverage + worker plan ──────────────────────────────
    st.subheader("Daily plan")
    dc1, dc2 = st.columns([1, 1])
    sel_day = dc1.selectbox("Day", DAY_NAMES, key="sel_day")
    sel_occ = dc2.selectbox("Occupation", [o.name for o in result.occupations], key="sel_occ")
    di = DAY_NAMES.index(sel_day)
    ds, de = di * INTERVALS_PER_DAY, (di + 1) * INTERVALS_PER_DAY
    oi = [o.name for o in result.occupations].index(sel_occ)
    occ = result.occupations[oi]
    hx = [k / INTERVALS_PER_HOUR for k in range(INTERVALS_PER_DAY)]

    dm1, dm2, dm3, dm4 = st.columns(4)
    d_load = occ.demand[ds:de]
    d_prod = res.productive[sel_occ][ds:de]
    day_workers = [w for w in res.workers if w.occupation == sel_occ and w.shift.day == di]
    day_tasks = [t for t in res.tasks if t.occupation == sel_occ and t.tid in res.start
                 and res.start[t.tid] // 1440 == di]
    dm1.metric("Peak task load", int(d_load.max()))
    dm2.metric("Peak available", int(d_prod.max()))
    dm3.metric("Workers starting", len(day_workers))
    dm4.metric("Tasks", len(day_tasks))

    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(x=hx, y=d_load, mode="lines", name="Task load",
                               line=dict(color="red", width=2),
                               fill="tozeroy", fillcolor="rgba(255,0,0,0.08)"))
    fig_d.add_trace(go.Scatter(x=hx, y=d_prod, mode="lines", name="Available for tasks",
                               line=dict(color="green", width=2, shape="hv"),
                               fill="tozeroy", fillcolor="rgba(0,128,0,0.08)"))
    fig_d.add_trace(go.Scatter(x=hx, y=occ.phase1.coverage[ds:de], mode="lines",
                               name="On shift", line=dict(color="grey", width=1, dash="dot",
                                                          shape="hv")))
    fig_d.update_xaxes(tickvals=list(range(0, 25, 1)), ticktext=[f"{h:02d}:00" for h in range(25)],
                       tickangle=45, range=[0, 24])
    fig_d.update_layout(height=320, margin=dict(l=40, r=20, t=30, b=50),
                        legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig_d, width="stretch")

    # Gantt: one row per worker starting that day
    if day_workers:
        tmap = {t.tid: t for t in res.tasks}
        names = sorted({tmap[i].name for w in day_workers for i in w.tasks})
        colour = {n: OCC_COLORS[(k + 1) % len(OCC_COLORS)] for k, n in enumerate(names)}
        day_workers.sort(key=lambda w: (w.shift.global_start, w.shift.global_end, w.wid))
        rows_y = [f"{w.wid} · {w.code}" for w in day_workers]
        fig_g = go.Figure()
        to_h = (lambda k: (k - ds) / INTERVALS_PER_HOUR)
        fig_g.add_trace(go.Bar(
            y=rows_y, x=[w.shift.duration_intervals / INTERVALS_PER_HOUR for w in day_workers],
            base=[to_h(w.shift.global_start) for w in day_workers], orientation="h",
            marker=dict(color="rgba(150,160,175,0.25)",
                        line=dict(color="rgba(120,130,145,0.6)", width=1)),
            name="Shift", hovertemplate="%{y}<extra>shift</extra>"))
        bw = [w for w in day_workers if w.break_end > w.break_start]
        if bw:
            fig_g.add_trace(go.Bar(
                y=[rows_y[day_workers.index(w)] for w in bw],
                x=[(w.break_end - w.break_start) / INTERVALS_PER_HOUR for w in bw],
                base=[to_h(w.break_start) for w in bw], orientation="h",
                marker=dict(color="rgba(90,90,90,0.55)"), name="Break",
                hovertemplate="break<extra></extra>"))
        for n in names:
            ys, xs, bs, tx = [], [], [], []
            for w, ylab in zip(day_workers, rows_y):
                for i in w.tasks:
                    t = tmap[i]
                    if t.name != n:
                        continue
                    s0 = res.start[i] / 5
                    ys.append(ylab)
                    xs.append(t.duration / 60)
                    bs.append(to_h(s0))
                    tx.append(f"{t.name} {t.flight} {t.location}".strip())
            fig_g.add_trace(go.Bar(y=ys, x=xs, base=bs, orientation="h", name=n,
                                   marker=dict(color=colour[n]), text=tx,
                                   hovertemplate="%{text}<extra></extra>",
                                   textposition="none"))
        fig_g.update_layout(barmode="overlay", height=max(260, 22 * len(day_workers) + 90),
                            margin=dict(l=10, r=20, t=10, b=40),
                            legend=dict(orientation="h", yanchor="bottom", y=1.0,
                                        xanchor="left", x=0))
        fig_g.update_yaxes(autorange="reversed", tickfont=dict(size=10))
        fig_g.update_xaxes(tickvals=list(range(0, 37, 1)),
                           ticktext=[f"{h % 24:02d}:00" for h in range(37)],
                           range=[max(0, min(b for b in [to_h(w.shift.global_start) for w in day_workers]) - 0.5),
                                  max(to_h(w.shift.global_end) for w in day_workers) + 0.5])
        st.markdown(f"**Worker plan – {sel_occ}, {sel_day}** (shifts starting that day)")
        st.plotly_chart(fig_g, width="stretch")
    else:
        st.caption("No shifts start on this day for this occupation.")

    # ── Tables ───────────────────────────────────────────────────────────
    st.subheader("Shift types")
    st.dataframe(shift_types(res), width="stretch", hide_index=True)

    with st.expander("🧾 Task assignment", expanded=False):
        adf = assignment_dataframe(res)
        st.dataframe(adf, width="stretch", hide_index=True, height=350)
    with st.expander("👷 Workers", expanded=False):
        st.dataframe(workers_dataframe(res), width="stretch", hide_index=True, height=350)
    with st.expander("🪵 Solver log", expanded=False):
        st.code("\n".join(res.log), language="text")

    st.subheader("📥 Download report")
    st.download_button(
        "⬇ Download XLSX report", data=build_task_report_xlsx(res),
        file_name="simplex_tasks_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
