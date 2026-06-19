"""Per-occupation headcount tests (entry-based).
All max/min constraints now count workers ENTERING shifts per day."""

import numpy as np
from solver import (
    SolverParams, solve_multi,
    daily_entry_headcount, max_headcount,
    DAY_NAMES, INTERVALS_PER_DAY, INTERVALS_PER_HOUR, TOTAL_INTERVALS,
)
from sample_data import generate_sample_demand

_pass = 0
_fail = 0

def ok(label, value):
    global _pass; _pass += 1
    print(f"  [PASS] {label}: {value}")

def fail(label, got, detail=""):
    global _fail; _fail += 1
    print(f"  [FAIL] {label}: {got}  {detail}")

def check(label, cond, detail=""):
    if cond: ok(label, "True")
    else: fail(label, "False", detail)


def flat_demand(val=3):
    arr = np.zeros(TOTAL_INTERVALS, dtype=int)
    for day in range(7):
        s = day * INTERVALS_PER_DAY + 48  # 04:00
        e = s + 120   # 10 hours later → 14:00
        arr[s:e] = val
    return arr


# == TEST 1 – Single occ, demand=3, per-occ limit=8 easily feasible ============
print("=" * 65)
print("TEST 1 – Single occ, demand=3, per-occ limit=8 (easily feasible)")
print("=" * 65)
d = flat_demand(3)
p = SolverParams(solver_time_limit_sec=30,
                 occ_max_headcount_per_day=[[8]*7])
r = solve_multi([d], ["Staff"], p)
check("TEST1a: feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    entries = daily_entry_headcount(r.occupations[0].phase1)
    for day in range(7):
        check(f"TEST1b-{DAY_NAMES[day]}: entries({entries[day]}) <= 8",
              entries[day] <= 8)

# == TEST 2 – Single occ, demand=8, per-occ limit=3 must be INFEASIBLE ========
print("\n" + "=" * 65)
print("TEST 2 – Single occ, demand=8, per-occ limit=3 (must be INFEASIBLE)")
print("=" * 65)
d = flat_demand(8)
p = SolverParams(solver_time_limit_sec=30,
                 occ_max_headcount_per_day=[[3]*7])
r = solve_multi([d], ["Staff"], p)
check("TEST2a: INFEASIBLE (demand 8 > limit 3)",
      r.combined_phase1.status not in ("OPTIMAL", "FEASIBLE"))

# == TEST 3 – Multi-occ with distinct per-occ limits (feasible) =================
print("\n" + "=" * 65)
print("TEST 3 – Multi-occ with distinct per-occ limits (feasible)")
print("=" * 65)
d1 = flat_demand(2); d2 = flat_demand(3); d3 = flat_demand(3)
p = SolverParams(solver_time_limit_sec=45,
                 occ_max_headcount_per_day=[
                     [5]*7, [6]*7, [6]*7])
r = solve_multi([d1, d2, d3], ["A", "B", "C"], p)
check("TEST3a: feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    limits = [5, 6, 6]
    for i, occ in enumerate(r.occupations):
        entries = daily_entry_headcount(occ.phase1)
        for day in range(7):
            check(f"TEST3b-{occ.name}-{DAY_NAMES[day]}: {entries[day]} <= {limits[i]}",
                  entries[day] <= limits[i])

# == TEST 4 – limit=0 means unlimited ==========================================
print("\n" + "=" * 65)
print("TEST 4 – limit=0 means unlimited (covers demand=5)")
print("=" * 65)
d = flat_demand(5)
p = SolverParams(solver_time_limit_sec=30,
                 occ_max_headcount_per_day=[[0]*7])
r = solve_multi([d], ["Staff"], p)
check("TEST4a: feasible when limit=0 (unlimited)",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

# == TEST 5 – Zero demand with per-occ limits ==================================
print("\n" + "=" * 65)
print("TEST 5 – Zero demand with per-occ limits")
print("=" * 65)
d = np.zeros(TOTAL_INTERVALS, dtype=int)
p = SolverParams(solver_time_limit_sec=30,
                 occ_max_headcount_per_day=[[3]*7])
r = solve_multi([d], ["Staff"], p)
check("TEST5a: feasible with zero demand", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    total = sum(r.occupations[0].phase1.counts)
    check("TEST5b: zero coverage for zero demand", total == 0)

# == TEST 6 – Different per-day limits per occupation ==========================
print("\n" + "=" * 65)
print("TEST 6 – Different per-day limits per occupation (feasible)")
print("=" * 65)
d = flat_demand(1)
p = SolverParams(solver_time_limit_sec=30,
                 occ_max_headcount_per_day=[
                     [3, 5, 2, 4, 3, 1, 1],
                     [2, 6, 3, 5, 4, 2, 1],
                 ])
r = solve_multi([d, d], ["A", "B"], p)
check("TEST6a: feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    for i, occ in enumerate(r.occupations):
        entries = daily_entry_headcount(occ.phase1)
        limits = p.occ_max_headcount_per_day[i]
        for day in range(7):
            check(f"TEST6b-{occ.name}-{DAY_NAMES[day]}: {entries[day]} <= {limits[day]}",
                  entries[day] <= limits[day])

# == TEST 7 – Per-occ limits + global max_headcount_per_day combined ============
print("\n" + "=" * 65)
print("TEST 7 – Per-occ limits + global max_headcount_per_day combined")
print("=" * 65)
d = flat_demand(2)
p = SolverParams(solver_time_limit_sec=45,
                 max_headcount_per_day=[8]*7,
                 occ_max_headcount_per_day=[[8]*7, [8]*7, [8]*7])
r = solve_multi([d, d, d], ["A", "B", "C"], p)
check("TEST7a: feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    for i, occ in enumerate(r.occupations):
        entries = daily_entry_headcount(occ.phase1)
        for day in range(7):
            check(f"TEST7b-{occ.name}-{DAY_NAMES[day]}: {entries[day]} <= 8",
                  entries[day] <= 8)
    total_e = daily_entry_headcount(r.combined_phase1)
    for day in range(7):
        check(f"TEST7c-{DAY_NAMES[day]}: total({total_e[day]}) <= 8",
              total_e[day] <= 8)

# == TEST 8 – Tight per-occ limit forces infeasible =============================
print("\n" + "=" * 65)
print("TEST 8 – Tight per-occ limit forces infeasible")
print("=" * 65)
d = flat_demand(5)
p = SolverParams(solver_time_limit_sec=30,
                 occ_max_headcount_per_day=[[1]*7])
r = solve_multi([d], ["Staff"], p)
check("TEST8a: infeasible when per-occ limit < demand",
      r.combined_phase1.status not in ("OPTIMAL", "FEASIBLE"))

# == TEST 9 – None per-occ limits (backward compat) =============================
print("\n" + "=" * 65)
print("TEST 9 – None per-occ limits (backward compat: no constraint)")
print("=" * 65)
d = flat_demand(5)
p = SolverParams(solver_time_limit_sec=30,
                 occ_max_headcount_per_day=[None])
r = solve_multi([d], ["Staff"], p)
check("TEST9a: works without per-occ limits",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

# == TEST 10 – Per-occ zero on weekends (= unlimited, coverage ok) =============
print("\n" + "=" * 65)
print("TEST 10 – Per-occ zero on weekends (= unlimited, coverage ok)")
print("=" * 65)
d = flat_demand(3)
p = SolverParams(solver_time_limit_sec=30,
                 occ_max_headcount_per_day=[[5, 5, 5, 5, 5, 0, 0]])
r = solve_multi([d], ["Staff"], p)
check("TEST10a: feasible with weekend limit=0 (unlimited)",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

# == TEST 11 – Mixed: some occs with limits, some without ======================
print("\n" + "=" * 65)
print("TEST 11 – Mixed: some occs with limits, some without")
print("=" * 65)
d = flat_demand(2)
p = SolverParams(solver_time_limit_sec=45,
                 occ_max_headcount_per_day=[[3]*7, None, [5]*7])
r = solve_multi([d, d, d], ["A", "B", "C"], p)
check("TEST11a: mixed None/default limits handled",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    limits = [3, None, 5]
    for i, occ in enumerate(r.occupations):
        entries = daily_entry_headcount(occ.phase1)
        lim = limits[i]
        if lim is not None:
            for day in range(7):
                check(f"TEST11b-{occ.name}-{DAY_NAMES[day]}: {entries[day]} <= {lim}",
                      entries[day] <= lim)

# == TEST 12 – 5 occupations with per-occ headcount limits =====================
print("\n" + "=" * 65)
print("TEST 12 – 5 occupations with per-occ headcount limits")
print("=" * 65)
d = flat_demand(1)
p = SolverParams(solver_time_limit_sec=45,
                 occ_max_headcount_per_day=[
                     [3, 3, 3, 3, 3, 2, 2],
                     [4, 4, 4, 4, 4, 3, 3],
                     [5, 5, 5, 5, 5, 4, 4],
                     [2, 2, 2, 2, 2, 2, 2],
                     [3, 3, 3, 3, 3, 3, 3],
                 ])
r = solve_multi([d]*5, ["Tech", "Lab", "Help", "Sup", "Asst"], p)
check("TEST12a: 5-occ feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    for i, occ in enumerate(r.occupations):
        entries = daily_entry_headcount(occ.phase1)
        limits = p.occ_max_headcount_per_day[i]
        for day in range(7):
            check(f"TEST12b-{occ.name}-{DAY_NAMES[day]}: {entries[day]} <= {limits[day]}",
                  entries[day] <= limits[day])

# == TEST 13 – Mixed limits across days (some 0=unlimited) =====================
print("\n" + "=" * 65)
print("TEST 13 – Mixed limits across days (some 0=unlimited)")
print("=" * 65)
d = flat_demand(2)
p = SolverParams(solver_time_limit_sec=30,
                 occ_max_headcount_per_day=[[3, 3, 3, 0, 0, 0, 0]])
r = solve_multi([d], ["Staff"], p)
check("TEST13a: feasible with partial limits",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    entries = daily_entry_headcount(r.occupations[0].phase1)
    for day in range(3):
        check(f"TEST13b-{DAY_NAMES[day]}: {entries[day]} <= 3",
              entries[day] <= 3)

# == TEST 14 – Empty occ_max_headcount_per_day list ============================
print("\n" + "=" * 65)
print("TEST 14 – Empty occ_max_headcount_per_day list")
print("=" * 65)
d = flat_demand(3)
p = SolverParams(solver_time_limit_sec=30)
r = solve_multi([d], ["Staff"], p)
check("TEST14a: empty list treated as no constraint",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

# == TEST 15 – Per-occ limit exactly equals demand =============================
print("\n" + "=" * 65)
print("TEST 15 – Per-occ limit exactly equals demand")
print("=" * 65)
d = flat_demand(4)
p = SolverParams(solver_time_limit_sec=30,
                 occ_max_headcount_per_day=[[4]*7])
r = solve_multi([d], ["Staff"], p)
check("TEST15a: feasible when limit == demand",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    entries = daily_entry_headcount(r.occupations[0].phase1)
    for day in range(7):
        check(f"TEST15b-{DAY_NAMES[day]}: {entries[day]} <= 4",
              entries[day] <= 4)

# == TEST 16 – Single occ, demand=3, per-occ min=2 (easily feasible) ===========
print("\n" + "=" * 65)
print("TEST 16 – Single occ, demand=3, per-occ min=2 (feasible)")
print("=" * 65)
d = flat_demand(3)
p = SolverParams(solver_time_limit_sec=30,
                 occ_min_headcount_per_day=[[2]*7])
r = solve_multi([d], ["Staff"], p)
check("TEST16a: feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    entries = daily_entry_headcount(r.occupations[0].phase1)
    for day in range(7):
        check(f"TEST16b-{DAY_NAMES[day]}: entries({entries[day]}) >= 2",
              entries[day] >= 2)

# == TEST 17 – Min=5 but max=4 makes infeasible (min > max) ====================
print("\n" + "=" * 65)
print("TEST 17 – Min=5 but max=4 makes infeasible (min > max)")
print("=" * 65)
d = flat_demand(3)
p = SolverParams(solver_time_limit_sec=30,
                 occ_min_headcount_per_day=[[5]*7],
                 occ_max_headcount_per_day=[[4]*7])
r = solve_multi([d], ["Staff"], p)
check("TEST17a: INFEASIBLE (min 5 > max 4)",
      r.combined_phase1.status not in ("OPTIMAL", "FEASIBLE"))

# == TEST 18 – Multi-occ with mixed min + max limits (feasible) =================
print("\n" + "=" * 65)
print("TEST 18 – Multi-occ with mixed min + max limits (feasible)")
print("=" * 65)
d1 = flat_demand(3); d2 = flat_demand(4)
p = SolverParams(solver_time_limit_sec=30,
                 occ_min_headcount_per_day=[[2]*7, [3]*7],
                 occ_max_headcount_per_day=[[5]*7, [6]*7])
r = solve_multi([d1, d2], ["A", "B"], p)
check("TEST18a: feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    mins = [2, 3]; maxs = [5, 6]
    for i, occ in enumerate(r.occupations):
        entries = daily_entry_headcount(occ.phase1)
        for day in range(7):
            check(f"TEST18b-{occ.name}-{DAY_NAMES[day]}-min",
                  entries[day] >= mins[i])
            check(f"TEST18c-{occ.name}-{DAY_NAMES[day]}-max",
                  entries[day] <= maxs[i])

# == TEST 19 – Min=0 (no constraint, backward compat) ==========================
print("\n" + "=" * 65)
print("TEST 19 – Min=0 (no constraint, backward compat)")
print("=" * 65)
d = flat_demand(3)
p = SolverParams(solver_time_limit_sec=30,
                 occ_min_headcount_per_day=[[0]*7])
r = solve_multi([d], ["Staff"], p)
check("TEST19a: feasible with min=0",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

# == TEST 20 – Min = Demand exactly (tight, should be feasible) ================
print("\n" + "=" * 65)
print("TEST 20 – Min = Demand exactly (tight, should be feasible)")
print("=" * 65)
d = flat_demand(3)
p = SolverParams(solver_time_limit_sec=30,
                 occ_min_headcount_per_day=[[3]*7])
r = solve_multi([d], ["Staff"], p)
check("TEST20a: feasible when min == demand",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    entries = daily_entry_headcount(r.occupations[0].phase1)
    for day in range(7):
        check(f"TEST20b-{DAY_NAMES[day]}: entries({entries[day]}) >= 3",
              entries[day] >= 3)

# == TEST 21 – Empty occ_min_headcount_per_day list (backward compat) ==========
print("\n" + "=" * 65)
print("TEST 21 – Empty occ_min_headcount_per_day list (backward compat)")
print("=" * 65)
d = flat_demand(3)
p = SolverParams(solver_time_limit_sec=30)
r = solve_multi([d], ["Staff"], p)
check("TEST21a: empty list treated as no min constraint",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

# == TEST 22 – Mixed: some occs with min, some without =========================
print("\n" + "=" * 65)
print("TEST 22 – Mixed: some occs with min, some without")
print("=" * 65)
d = flat_demand(3)
p = SolverParams(solver_time_limit_sec=45,
                 occ_min_headcount_per_day=[[3]*7, None, [2]*7])
r = solve_multi([d, d, d], ["A", "B", "C"], p)
check("TEST22a: mixed min limits handled",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    mins = [3, None, 2]
    for i, occ in enumerate(r.occupations):
        entries = daily_entry_headcount(occ.phase1)
        lim = mins[i]
        if lim is not None:
            for day in range(7):
                check(f"TEST22b-{occ.name}-{DAY_NAMES[day]}: {entries[day]} >= {lim}",
                      entries[day] >= lim)

# == TEST 23 – Min headcount + global max_headcount_per_day combined ============
print("\n" + "=" * 65)
print("TEST 23 – Min headcount + global max_headcount_per_day combined")
print("=" * 65)
d = flat_demand(3)
p = SolverParams(solver_time_limit_sec=45,
                 occ_min_headcount_per_day=[[2]*7, [1]*7],
                 max_headcount_per_day=[6]*7)
r = solve_multi([d, d], ["A", "B"], p)
check("TEST23a: combined min + global max feasible",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    for i, occ in enumerate(r.occupations):
        entries = daily_entry_headcount(occ.phase1)
        m = 2 if i == 0 else 1
        for day in range(7):
            check(f"TEST23b-{occ.name}-{DAY_NAMES[day]}-min",
                  entries[day] >= m)
    total_e = daily_entry_headcount(r.combined_phase1)
    for day in range(7):
        check(f"TEST23c-{DAY_NAMES[day]}-total: {total_e[day]} <= 6",
              total_e[day] <= 6)

# == TEST 24 – 5 occupations with per-occ min headcount ========================
print("\n" + "=" * 65)
print("TEST 24 – 5 occupations with per-occ min headcount")
print("=" * 65)
d = flat_demand(3)
p = SolverParams(solver_time_limit_sec=45,
                 occ_min_headcount_per_day=[
                     [2]*7, [1]*7, [1]*7, [1]*7, [1]*7,
                 ])
r = solve_multi([d]*5, ["Tech", "Lab", "Help", "Sup", "Asst"], p)
check("TEST24a: 5-occ with min headcount feasible",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    mins = [2, 1, 1, 1, 1]
    for i, occ in enumerate(r.occupations):
        entries = daily_entry_headcount(occ.phase1)
        for day in range(7):
            check(f"TEST24b-{occ.name}-{DAY_NAMES[day]}: {entries[day]} >= {mins[i]}",
                  entries[day] >= mins[i])

# == SUMMARY ===================================================================
print("\n" + "=" * 65)
print(f"SUMMARY  PASS={_pass}  FAIL={_fail}")
print("=" * 65)
if _fail == 0:
    print("All per-occupation headcount tests passed.")
else:
    print(f"{_fail} checks FAILED – see above.")