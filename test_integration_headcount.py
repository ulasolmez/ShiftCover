"""Integration test – ALL headcount parameter types simultaneously (entry-based).
Every constraint now counts workers ENTERING shifts on a day."""

import numpy as np
from solver import (
    SolverParams, solve_multi,
    daily_entry_headcount, max_headcount,
    DAY_NAMES, INTERVALS_PER_DAY, INTERVALS_PER_HOUR, TOTAL_INTERVALS,
)

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


def flat_demand(val, start_h=4, dur_h=10):
    arr = np.zeros(TOTAL_INTERVALS, dtype=int)
    for day in range(7):
        s = day * INTERVALS_PER_DAY + start_h * INTERVALS_PER_HOUR
        e = s + dur_h * INTERVALS_PER_HOUR
        arr[s:e] = val
    return arr


# == SCENARIO 1 – 5 occupations, ALL headcount constraints active ==============
print("=" * 65)
print("INTEGRATION — 5 occupations, ALL headcount constraints active")
print("=" * 65)

occ_names = ["Technician", "Labourer", "Helper", "Supervisor", "Assistant"]
d1 = flat_demand(12, 6, 14)
d2 = flat_demand(8,  8, 8)
d3 = flat_demand(6,  6, 10)
d4 = flat_demand(4,  7, 9)
d5 = flat_demand(4,  8, 10)
demands_all = [d1, d2, d3, d4, d5]

combined_max = [80]*7
occ_max = [
    [20]*7, [16]*7, [14]*7, [10]*7, [10]*7,
]
occ_shift_code = [
    {"0800-1600": {"max": [10]*7}, "0600-1400": {"min": [2]*7}},
    {"0800-1600": {"max": [8]*7}},
    {"0600-1400": {"max": [6]*7}},
    None, None,
]

params = SolverParams(
    min_shift_hours=3.0, max_shift_hours=12.0, solver_time_limit_sec=300,
    max_headcount_per_day=combined_max,
    occ_max_headcount_per_day=occ_max,
    occ_headcount_per_shift_code=occ_shift_code,
)
print("\nSolving with all constraints active...")
result = solve_multi(demands_all, occ_names, params)
cp1 = result.combined_phase1
print(f"Solver status: {cp1.status}  time={cp1.elapsed_sec:.1f}s")
check("INT-0: solver found solution", cp1.status in ("OPTIMAL", "FEASIBLE"),
      f"status={cp1.status}")

if cp1.status in ("OPTIMAL", "FEASIBLE"):
    e_comb = daily_entry_headcount(cp1)
    for day in range(7):
        check(f"INT1-{DAY_NAMES[day]}: combined_entries({e_comb[day]}) <= {combined_max[day]}",
              e_comb[day] <= combined_max[day])

    for i, occ in enumerate(result.occupations):
        e = daily_entry_headcount(occ.phase1)
        occ_limits = occ_max[i]
        for day in range(7):
            check(f"INT2-{occ.name}-{DAY_NAMES[day]}: {e[day]} <= {occ_limits[day]}",
                  e[day] <= occ_limits[day])

    for i, (occ, sc) in enumerate(zip(result.occupations, occ_shift_code)):
        if sc is None: continue
        for code, spec in sc.items():
            if isinstance(spec, list): mx_l, mn_l = spec, None
            else: mx_l = spec.get("max"); mn_l = spec.get("min")
            daily = [0]*7
            for s, cnt in zip(occ.phase1.shifts, occ.phase1.counts):
                if s.shift_code == code and cnt > 0:
                    daily[s.day] += cnt
            for day in range(7):
                if mx_l and mx_l[day] > 0:
                    check(f"INT3-{occ.name}-{code}-{DAY_NAMES[day]}: {daily[day]} <= {mx_l[day]}",
                          daily[day] <= mx_l[day])
                if mn_l and mn_l[day] > 0:
                    check(f"INT4-{occ.name}-{code}-{DAY_NAMES[day]}: {daily[day]} >= {mn_l[day]}",
                          daily[day] >= mn_l[day])

    for occ in result.occupations:
        deficit = np.maximum(occ.demand - occ.phase1.coverage, 0)
        check(f"INT5-{occ.name}: demand covered", int(np.count_nonzero(deficit)) == 0)

    ps = np.zeros(TOTAL_INTERVALS, dtype=int)
    for occ in result.occupations:
        ps += occ.phase1.coverage
    check("INT6: combined==sum", np.all(ps == cp1.coverage))


# == SCENARIO 2 – Tight scenario (3 occs with min/max + shift-code) ============
print("\n" + "=" * 65)
print("INTEGRATION — 3 occupations, tight limits")
print("=" * 65)

d = flat_demand(6, 6, 12)
p = SolverParams(
    min_shift_hours=4.0, max_shift_hours=10.0, solver_time_limit_sec=60,
    max_headcount_per_day=[40]*7, min_headcount_per_day=[8,8,8,8,8,3,3],
    occ_max_headcount_per_day=[[18]*7]*3,
    occ_min_headcount_per_day=[[2,2,2,2,2,1,1]]*3,
    occ_headcount_per_shift_code=[
        {"0800-1600": {"max": [10]*7}},
        {"0600-1400": {"max": [10]*7, "min": [2]*7}},
        {"1000-1800": {"max": [10]*7}},
    ],
    max_entries_per_day=[14]*7, max_exits_per_day=[14]*7,
)
r2 = solve_multi([d, d, d], ["A", "B", "C"], p)
cp2 = r2.combined_phase1
print(f"Solver status: {cp2.status}  time={cp2.elapsed_sec:.1f}s")
check("INT-T1: tight scenario feasible", cp2.status in ("OPTIMAL", "FEASIBLE"),
      f"status={cp2.status}")

if cp2.status in ("OPTIMAL", "FEASIBLE"):
    e_comb = daily_entry_headcount(cp2)
    mins = [8,8,8,8,8,3,3]
    for day in range(7):
        check(f"INT-T2-{DAY_NAMES[day]}: entries({e_comb[day]}) <= 40", e_comb[day] <= 40)
        check(f"INT-T3-{DAY_NAMES[day]}: entries({e_comb[day]}) >= {mins[day]}",
              e_comb[day] >= mins[day])

    for i, (occ, sc) in enumerate(zip(r2.occupations, p.occ_headcount_per_shift_code)):
        e = daily_entry_headcount(occ.phase1)
        for day in range(7):
            check(f"INT-T4-{occ.name}-{DAY_NAMES[day]}: {e[day]} <= 18", e[day] <= 18)
            mn_p = 2 if day < 5 else 1
            check(f"INT-T5-{occ.name}-{DAY_NAMES[day]}: {e[day]} >= {mn_p}", e[day] >= mn_p)
        if sc:
            for code, spec in sc.items():
                mx_l = spec.get("max")
                mn_l = spec.get("min") if isinstance(spec, dict) else None
                daily = [0]*7
                for s, cnt in zip(occ.phase1.shifts, occ.phase1.counts):
                    if s.shift_code == code and cnt > 0:
                        daily[s.day] += cnt
                for day in range(7):
                    if mx_l and mx_l[day] > 0:
                        check(f"INT-T6-{occ.name}-{code}-{DAY_NAMES[day]}: {daily[day]} <= {mx_l[day]}",
                              daily[day] <= mx_l[day])
                    if mn_l and mn_l[day] > 0:
                        check(f"INT-T7-{occ.name}-{code}-{DAY_NAMES[day]}: {daily[day]} >= {mn_l[day]}",
                              daily[day] >= mn_l[day])

    from collections import defaultdict
    de = defaultdict(set); dx = defaultdict(set)
    for s, cnt in zip(cp2.shifts, cp2.counts):
        if cnt > 0:
            de[s.day].add(s.start_interval)
            dx[s.day].add(s.start_interval + s.duration_intervals)
    for day in range(7):
        check(f"INT-T8a-{DAY_NAMES[day]}: entries({len(de[day])}) <= 14",
              len(de[day]) <= 14)
        check(f"INT-T8b-{DAY_NAMES[day]}: exits({len(dx[day])}) <= 14",
              len(dx[day]) <= 14)


# == SCENARIO 3 – Zero demand ==================================================
print("\n" + "=" * 65)
print("INTEGRATION — Zero demand with constraints")
print("=" * 65)
rz = solve_multi([np.zeros(TOTAL_INTERVALS, dtype=int)], ["Staff"],
    SolverParams(solver_time_limit_sec=30, max_headcount_per_day=[5]*7,
                 min_headcount_per_day=[0]*7,
                 occ_max_headcount_per_day=[[3]*7],
                 occ_headcount_per_shift_code=[{"0800-1600": {"max": [2]*7}}]))
cz = rz.combined_phase1
check("INT-Z1: zero-demand feasible", cz.status in ("OPTIMAL", "FEASIBLE"))
if cz.status in ("OPTIMAL", "FEASIBLE"):
    check("INT-Z2: zero workers assigned", sum(rz.occupations[0].phase1.counts) == 0)

# == SCENARIO 4 – Impossible (min > max) =======================================
print("\n" + "=" * 65)
print("INTEGRATION – Impossible (min > max)")
print("=" * 65)
ri = solve_multi([flat_demand(3, 6, 8)], ["Staff"],
    SolverParams(solver_time_limit_sec=15,
                 min_headcount_per_day=[50]*7, max_headcount_per_day=[10]*7))
check("INT-I1: min=50 > max=10 -> INFEASIBLE",
      ri.combined_phase1.status not in ("OPTIMAL", "FEASIBLE"))

# == SUMMARY ===================================================================
print("\n" + "=" * 65)
print(f"SUMMARY  PASS={_pass}  FAIL={_fail}")
print("=" * 65)
if _fail == 0:
    print("All integration tests passed.")
else:
    print(f"{_fail} checks FAILED.")