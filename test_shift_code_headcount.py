"""
Shift-type (per-shift-code) headcount limit tests.

Tests for:
  - Max workers per shift code per day (single occ)
  - Min workers per shift code per day (single occ)
  - Both min & max together
  - Multi-occ with distinct shift-code limits
  - Min > max on same code → INFEASIBLE
  - Max only (list format, backward compat)
  - None entries (no constraint on some occs)
"""

import numpy as np
from solver import (
    SolverParams, solve_multi,
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


def flat_demand(val=3):
    arr = np.zeros(TOTAL_INTERVALS, dtype=int)
    for day in range(7):
        s = day * INTERVALS_PER_DAY + 48  # 04:00
        e = s + 120   # 10 hours later → 14:00
        arr[s:e] = val
    return arr


def quick_params(**kw):
    p = SolverParams(solver_time_limit_sec=30, **kw)
    return p


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 65)
print("TEST 1 – Max per shift code (max-only, list format)")
print("=" * 65)

# Constrain 0600-1400 to max 5, demand=3 → feasible
d = flat_demand(3)
sc = {"0600-1400": {"max": [5, 5, 5, 5, 5, 5, 5]}}
p = quick_params(occ_headcount_per_shift_code=[sc])
r = solve_multi([d], ["Staff"], p)
check("TEST1a: max=5 feasible with demand=3", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

# Count actual 0600-1400 workers per day
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    occ = r.occupations[0]
    daily = [0] * 7
    for s, cnt in zip(occ.phase1.shifts, occ.phase1.counts):
        if s.shift_code == "0600-1400":
            daily[s.day] += cnt
    for day in range(7):
        check(f"TEST1b-{DAY_NAMES[day]}: daily_workers({daily[day]}) <= 5",
              daily[day] <= 5, f"actual={daily[day]} > limit=5")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("TEST 2 – Min per shift code (min only)")
print("=" * 65)

# Demand=8, min=6 on 0600-1400 → must use at least 6 workers of that code
d = flat_demand(8)
sc = {"0600-1400": {"min": [6, 6, 6, 6, 6, 6, 6]}}
p = quick_params(occ_headcount_per_shift_code=[sc])
r = solve_multi([d], ["Staff"], p)
check("TEST2a: min=6 feasible with demand=8", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    occ = r.occupations[0]
    daily = [0] * 7
    for s, cnt in zip(occ.phase1.shifts, occ.phase1.counts):
        if s.shift_code == "0600-1400":
            daily[s.day] += cnt
    for day in range(7):
        check(f"TEST2b-{DAY_NAMES[day]}: daily_workers({daily[day]}) >= 6",
              daily[day] >= 6, f"actual={daily[day]} < min=6")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("TEST 3 – Both min & max on same shift code")
print("=" * 65)

# min=2, max=5, demand=4 → feasible, must stay within [2, 5]
d = flat_demand(4)
sc = {"0600-1400": {"max": [5, 5, 5, 5, 5, 5, 5],
                     "min": [2, 2, 2, 2, 2, 2, 2]}}
p = quick_params(occ_headcount_per_shift_code=[sc])
r = solve_multi([d], ["Staff"], p)
check("TEST3a: min=2, max=5 feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    occ = r.occupations[0]
    daily = [0] * 7
    for s, cnt in zip(occ.phase1.shifts, occ.phase1.counts):
        if s.shift_code == "0600-1400":
            daily[s.day] += cnt
    for day in range(7):
        check(f"TEST3b-{DAY_NAMES[day]}-max", daily[day] <= 5,
              f"actual={daily[day]} > max=5")
        check(f"TEST3c-{DAY_NAMES[day]}-min", daily[day] >= 2,
              f"actual={daily[day]} < min=2")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("TEST 4 – Min > max on same shift code → INFEASIBLE")
print("=" * 65)

d = flat_demand(5)
sc = {"0600-1400": {"max": [3, 3, 3, 3, 3, 3, 3],
                     "min": [8, 8, 8, 8, 8, 8, 8]}}
p = quick_params(occ_headcount_per_shift_code=[sc])
r = solve_multi([d], ["Staff"], p)
check("TEST4a: INFEASIBLE when min=8 > max=3",
      r.combined_phase1.status not in ("OPTIMAL", "FEASIBLE"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("TEST 5 – Multi-occ with distinct shift-code limits")
print("=" * 65)

d1 = flat_demand(5)
d2 = flat_demand(5)
sc1 = {"0600-1400": {"max": [4, 4, 4, 4, 4, 4, 4]}}
sc2 = {"0800-1600": {"max": [3, 3, 3, 3, 3, 3, 3],
                      "min": [1, 1, 1, 1, 1, 1, 1]}}
p = quick_params(occ_headcount_per_shift_code=[sc1, sc2])
r = solve_multi([d1, d2], ["Tech", "Lab"], p)
check("TEST5a: multi-occ feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    for occ_i, (occ, sc) in enumerate(zip(r.occupations, [sc1, sc2])):
        for code, spec in sc.items():
            max_limits = spec.get("max", [0]*7)
            min_limits = spec.get("min")
            daily = [0] * 7
            for s, cnt in zip(occ.phase1.shifts, occ.phase1.counts):
                if s.shift_code == code:
                    daily[s.day] += cnt
            for day in range(7):
                if max_limits[day] > 0:
                    check(f"TEST5b-{occ.name}-{code}-{DAY_NAMES[day]}-max",
                          daily[day] <= max_limits[day],
                          f"actual={daily[day]} > max={max_limits[day]}")
                if min_limits and min_limits[day] > 0:
                    check(f"TEST5c-{occ.name}-{code}-{DAY_NAMES[day]}-min",
                          daily[day] >= min_limits[day],
                          f"actual={daily[day]} < min={min_limits[day]}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("TEST 6 – Different per-day limits for same code")
print("=" * 65)

d = flat_demand(5)
sc = {"0600-1400": {"max": [2, 4, 6, 8, 10, 12, 0],  # Sun=0 = unlimited
                     "min": [0, 0, 0, 0, 0, 0, 0]}}
p = quick_params(occ_headcount_per_shift_code=[sc])
r = solve_multi([d], ["Staff"], p)
check("TEST6a: varying daily max feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    occ = r.occupations[0]
    daily = [0] * 7
    for s, cnt in zip(occ.phase1.shifts, occ.phase1.counts):
        if s.shift_code == "0600-1400":
            daily[s.day] += cnt
    limits = [2, 4, 6, 8, 10, 12, 0]
    for day in range(7):
        lim = limits[day]
        if lim > 0:
            check(f"TEST6b-{DAY_NAMES[day]}: workers({daily[day]}) <= {lim}",
                  daily[day] <= lim)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("TEST 7 – None entry → no constraint (backward compat)")
print("=" * 65)

d = flat_demand(5)
sc = None  # no constraint for this occupation
p = quick_params(occ_headcount_per_shift_code=[sc])
r = solve_multi([d], ["Staff"], p)
check("TEST7a: None treated as no constraint (feasible)",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("TEST 8 – Mixed: some occs with shift-code limits, some without")
print("=" * 65)

d1 = flat_demand(4)
d2 = flat_demand(4)
d3 = flat_demand(4)
sc1 = {"0600-1400": {"max": [3, 3, 3, 3, 3, 3, 3]}}
sc2 = None
sc3 = {"0800-1600": {"min": [2, 2, 2, 2, 2, 2, 2]}}
p = quick_params(occ_headcount_per_shift_code=[sc1, sc2, sc3])
r = solve_multi([d1, d2, d3], ["A", "B", "C"], p)
check("TEST8a: mixed constraints feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    # Occ A: max=3 on 0600-1400
    daily = [0] * 7
    for s, cnt in zip(r.occupations[0].phase1.shifts, r.occupations[0].phase1.counts):
        if s.shift_code == "0600-1400":
            daily[s.day] += cnt
    for day in range(7):
        check(f"TEST8b-A-{DAY_NAMES[day]}: {daily[day]} <= 3", daily[day] <= 3)

    # Occ C: min=2 on 0800-1600
    daily = [0] * 7
    for s, cnt in zip(r.occupations[2].phase1.shifts, r.occupations[2].phase1.counts):
        if s.shift_code == "0800-1600":
            daily[s.day] += cnt
    for day in range(7):
        check(f"TEST8c-C-{DAY_NAMES[day]}: {daily[day]} >= 2", daily[day] >= 2)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("TEST 9 – Max only (list format, backward compat)")
print("=" * 65)

d = flat_demand(3)
sc = {"0600-1400": [5, 5, 5, 5, 5, 5, 5]}  # list = max only
p = quick_params(occ_headcount_per_shift_code=[sc])
r = solve_multi([d], ["Staff"], p)
check("TEST9a: list format max=5 feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    occ = r.occupations[0]
    daily = [0] * 7
    for s, cnt in zip(occ.phase1.shifts, occ.phase1.counts):
        if s.shift_code == "0600-1400":
            daily[s.day] += cnt
    for day in range(7):
        check(f"TEST9b-{DAY_NAMES[day]}: {daily[day]} <= 5", daily[day] <= 5)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("TEST 10 – Constraint on code NOT used by solver")
print("=" * 65)

d = flat_demand(1)
# Set a max of 0 on 0600-1400, forcing solver to use a different code
sc = {"0600-1400": {"max": [0, 0, 0, 0, 0, 0, 0]}}
p = quick_params(occ_headcount_per_shift_code=[sc])
r = solve_multi([d], ["Staff"], p)
check("TEST10a: feasible with 0600-1400 max=0", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    occ = r.occupations[0]
    daily = [0] * 7
    for s, cnt in zip(occ.phase1.shifts, occ.phase1.counts):
        if s.shift_code == "0600-1400" and cnt > 0:
            daily[s.day] += cnt
    for day in range(7):
        check(f"TEST10b-{DAY_NAMES[day]}: no 0600-1400 workers",
              daily[day] == 0, f"found {daily[day]} workers on constrained code")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print(f"SUMMARY  PASS={_pass}  FAIL={_fail}")
print("=" * 65)
if _fail == 0:
    print("All shift-type headcount tests passed.")
else:
    print("Some checks FAILED – see above for details.")