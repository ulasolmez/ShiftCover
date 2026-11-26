"""Tests for the task-based Simplex (tasks.py + task_solver.py)."""

import io

import pandas as pd
import pytest

from solver import INTERVALS_PER_HOUR
from tasks import (
    Task, Flight, default_rules, tasks_from_flights, read_flights,
    read_task_list, generate_sample_flights, task_list_csv, flights_csv,
    occupations_of,
)
from task_solver import (
    TaskParams, solve_tasks, occupancy, build_task_report_xlsx,
    assignment_dataframe, workers_dataframe, shift_windows, _iv,
)


# ── helpers ──────────────────────────────────────────────────────────────────
def _check_plan(res, p):
    """Every staffed task: whole inside one worker's productive time, no
    overlap (incl. travel) with the worker's other tasks or break."""
    tmap = {t.tid: t for t in res.tasks}
    wmap = {w.wid: w for w in res.workers}
    for tid, wids in res.assigned.items():
        t = tmap[tid]
        assert len(set(wids)) == len(wids), "same worker twice on one task"
        st = res.start[tid] // 5
        for wid in wids:
            w = wmap[wid]
            a, b = st - _iv(p.travel_min), st + _iv(t.duration)
            assert w.prod_start <= a and b <= w.prod_end, f"{t.name} outside shift"
            if w.break_end > w.break_start:
                assert not (w.break_start < st + _iv(t.duration) and w.break_end > st), \
                    f"{t.name} during break"
    for w in res.workers:
        spans = sorted((res.start[i] // 5, res.start[i] // 5 + _iv(tmap[i].duration))
                       for i in w.tasks)
        for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
            assert b1 <= a2, f"{w.wid} has overlapping tasks"
        assert w.shift.duration_intervals >= p.min_shift_hours * INTERVALS_PER_HOUR


def _small_tasks(fpd=6, seed=3):
    fl = generate_sample_flights(fpd, 0.2, seed)
    fl = [f for f in fl if f.day <= 1]            # Monday + Tuesday only
    tasks, _ = tasks_from_flights(fl, default_rules())
    return fl, tasks


# ── task building / import ──────────────────────────────────────────────────
def test_rules_apply_offsets_and_wide_body_staff():
    fl = [Flight("TK1", 0, sta=8 * 60, std=9 * 60, category="WB", stand="A1")]
    tasks, warns = tasks_from_flights(fl, default_rules())
    assert not warns
    unl = next(t for t in tasks if t.name == "Unloading")
    assert (unl.earliest, unl.latest_end, unl.duration, unl.staff) == (480, 520, 20, 5)
    load = next(t for t in tasks if t.name == "Loading")
    assert (load.earliest, load.latest_end) == (540 - 55, 540 - 10)
    assert {t.occupation for t in tasks} == {"Ramp", "Cleaning"}   # default ticks
    assert all(t.location == "A1" and t.flight == "TK1" for t in tasks)


def test_tasks_outside_week_are_dropped_unless_circular():
    fl = [Flight("LATE", 6, sta=6 * 1440 + 23 * 60 + 50, std=None)]
    t1, w1 = tasks_from_flights(fl, default_rules(), circular=False)
    assert not any(t.name == "Unloading" for t in t1) and w1
    t2, _ = tasks_from_flights(fl, default_rules(), circular=True)
    assert any(t.name == "Unloading" for t in t2)


def test_read_flights_clock_times_and_overnight():
    csv = ("Flight,Day,STA,STD,AC type,Stand\n"
           "XQ1,Mon,07:35,08:30,A321,B4\n"
           "XQ2,Tue,23:10,00:20,B77W,C1\n"
           "XQ3,Wed,,06:00,A320,A2\n").encode()
    fl, warns, weeks = read_flights(csv, "f.csv")
    assert not weeks
    assert [f.category for f in fl] == ["NB", "WB", "NB"]
    assert fl[0].sta == 7 * 60 + 35 and fl[0].std == 8 * 60 + 30
    assert fl[1].std == 1440 + 1440 + 20           # departs after midnight
    assert fl[2].sta is None and fl[2].std == 2 * 1440 + 360


def test_read_flights_dated_rows_pick_week():
    df = pd.DataFrame({
        "Flight": ["A", "B"],
        "STA": [pd.Timestamp("2025-08-04 10:00"), pd.Timestamp("2025-08-12 11:00")],
        "STD": [pd.Timestamp("2025-08-04 11:00"), pd.Timestamp("2025-08-12 12:00")],
    })
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    fl, _, weeks = read_flights(buf.getvalue(), "f.xlsx")
    assert len(weeks) == 2
    assert [f.flight for f in fl] == ["A"]          # second week is outside
    fl2, _, _ = read_flights(buf.getvalue(), "f.xlsx", weeks[1])
    assert [f.flight for f in fl2] == ["B"] and fl2[0].sta == 1440 + 11 * 60


def test_task_list_roundtrip():
    _, tasks = _small_tasks()
    back, warns, _ = read_task_list(task_list_csv(tasks), "tasks.csv")
    assert not warns
    assert len(back) == len(tasks)
    for a, b in zip(tasks, back):
        assert (a.name, a.occupation, a.earliest, a.latest_end, a.duration, a.staff) == \
               (b.name, b.occupation, b.earliest, b.latest_end, b.duration, b.staff)


def test_flights_csv_roundtrip():
    fl, _ = _small_tasks()[0], None
    back, _, _ = read_flights(flights_csv(fl), "flights.csv")
    assert len(back) == len(fl)


# ── solver ───────────────────────────────────────────────────────────────────
def test_shift_windows_briefing_closeout_break():
    from solver import CandidateShift
    p = TaskParams(briefing_min=15, closeout_min=10, breaks=True)
    s = CandidateShift(0, 0, 8 * 12, 8 * 12)       # 08:00-16:00
    ps, pe, bs, be = shift_windows(s, p)
    assert ps == 8 * 12 + 3 and pe == 16 * 12 - 2
    assert be - bs == 12                             # 60 min break from 8 h
    s3 = CandidateShift(1, 0, 8 * 12, 3 * 12)
    ps, pe, bs, be = shift_windows(s3, p)
    assert bs == be                                  # no break under 4 h


@pytest.fixture(scope="module")
def small_result():
    _, tasks = _small_tasks()
    p = TaskParams(solver_time_limit_sec=30)
    return tasks, p, solve_tasks(tasks, p)


def test_every_task_staffed_and_plan_valid(small_result):
    tasks, p, res = small_result
    assert res.status in ("OPTIMAL", "FEASIBLE")
    assert not res.unassigned
    for t in tasks:
        assert len(res.assigned[t.tid]) == t.staff
        st = res.start[t.tid]
        assert t.earliest <= st + 4 and st + t.duration <= t.latest_end + 4
    _check_plan(res, p)


def test_task_load_is_covered_by_available_staff(small_result):
    _, _, res = small_result
    for occ in res.multi.occupations:
        assert (res.productive[occ.name] >= occ.demand).all()


def test_report_and_tables(small_result):
    _, _, res = small_result
    x = build_task_report_xlsx(res)
    sheets = pd.ExcelFile(io.BytesIO(x)).sheet_names
    for name in ("Shift Types", "Coverage", "Shifttype", "Task Assignment", "Workers"):
        assert name in sheets
    adf = assignment_dataframe(res)
    assert (adf["Status"] == "✅").all()
    assert len(workers_dataframe(res)) == len(res.workers)


def test_rules_still_apply_max_headcount_and_force_include():
    _, tasks = _small_tasks()
    tasks = [t for t in tasks if t.occupation == "Ramp"]
    p = TaskParams(solver_time_limit_sec=30, transition_penalty=0,
                   force_include_shifts=["0800-1600"], assign_rounds=1)
    res = solve_tasks(tasks, p)
    codes = {s.shift_code for s, c in zip(res.multi.combined_phase1.shifts,
                                          res.multi.combined_phase1.counts) if c}
    assert "0800-1600" in codes
    _check_plan(res, p)


def test_fixed_task_cannot_move():
    t = Task(0, "Check-in", "Pax", 600, 705, 105, 2, "X1", "C1")
    p = TaskParams(solver_time_limit_sec=15)
    res = solve_tasks([t], p)
    assert res.start[0] == 600
    assert len(res.assigned[0]) == 2
    _check_plan(res, p)


def test_unreachable_task_is_reported():
    # Sunday 23:55 — no non-circular shift can be productive then
    t = Task(0, "Late", "Ramp", 7 * 1440 - 10, 7 * 1440, 10, 1)
    res = solve_tasks([t], TaskParams(solver_time_limit_sec=10))
    assert res.unassigned == [0]


def test_occupations_keep_order():
    _, tasks = _small_tasks()
    assert occupations_of(tasks)[0] == tasks[0].occupation
