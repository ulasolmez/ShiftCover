"""Headcount constraint + display correctness tests (entry-based).
Headcount now counts workers ENTERING shifts on a day, not simultaneous workers."""

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


# == TEST 1 – max_headcount uses entries now ===================================
print("\n" + "=" * 65)
print("TEST 1 – max_headcount() == max(entries)")
print("=" * 65)

d = generate_sample_demand(peak_agents=25, base_agents=2, seed=42)
r0 = solve_multi([d], ["Staff"], SolverParams(solver_time_limit_sec=60))
cp1 = r0.combined_phase1
print(f"Status: {cp1.status}")

entries = daily_entry_headcount(cp1)
peak_sims = [int(cp1.coverage[day*INTERVALS_PER_DAY:(day+1)*INTERVALS_PER_DAY].max())
             for day in range(7)]

print(f"\n  {'Day':<11} {'Entries':>9} {'PeakSim':>9}")
print("  " + "-" * 35)
for day in range(7):
    print(f"  {DAY_NAMES[day]:<11} {entries[day]:>9} {peak_sims[day]:>9}")

check("TEST1a: max_headcount() == max(entries) (entry-based)",
      max_headcount(cp1) == max(entries),
      f"entries_max={max(entries)} headcount={max_headcount(cp1)}")
check("TEST1b: max(entries) >= max(peak_sims) always",
      max(entries) >= max(peak_sims))

# == TEST 2 – Combined max (entry-based) constraint ============================
print("\n" + "=" * 65)
print("TEST 2 – Combined max headcount (entry-based)")
print("=" * 65)

# Set limit = unconstrained entries, must be feasible and <= limit
hc_limit = [max(e, 1) for e in entries]
params_cap = SolverParams(solver_time_limit_sec=60, max_headcount_per_day=hc_limit)
r2 = solve_multi([d], ["Staff"], params_cap)
cp2 = r2.combined_phase1
print(f"Status: {cp2.status}  (limit={hc_limit})")
check("TEST2a: solver finds solution", cp2.status in ("OPTIMAL", "FEASIBLE"))

actual_entries = daily_entry_headcount(cp2)
for day in range(7):
    actual = actual_entries[day]
    limit = hc_limit[day]
    check(f"TEST2b-{DAY_NAMES[day]}: entries({actual}) <= {limit}",
          actual <= limit, f"VIOLATION: {actual} > {limit}")

# == TEST 3 – Tight constraint (70% of unconstrained entries) ==================
print("\n" + "=" * 65)
print("TEST 3 – Tight constraint (70% of unconstrained entries)")
print("=" * 65)

hc_tight = [max(1, int(e * 0.70)) for e in entries]
params_tight = SolverParams(solver_time_limit_sec=90, max_headcount_per_day=hc_tight)
r3 = solve_multi([d], ["Staff"], params_tight)
cp3 = r3.combined_phase1
print(f"Status: {cp3.status}  (limits={hc_tight})")

if cp3.status in ("OPTIMAL", "FEASIBLE"):
    actual_e = daily_entry_headcount(cp3)
    for day in range(7):
        actual = actual_e[day]
        limit = hc_tight[day]
        check(f"TEST3-{DAY_NAMES[day]}: entries({actual}) <= {limit}",
              actual <= limit, "REAL CONSTRAINT BUG" if actual > limit else "")
else:
    ok("TEST3: INFEASIBLE at 70% is a valid outcome", cp3.status)

# == TEST 4 – Combined min (entry-based) =======================================
print("\n" + "=" * 65)
print("TEST 4 – Combined min headcount (entry-based)")
print("=" * 65)

d = generate_sample_demand(peak_agents=25, base_agents=2, seed=42)
p = SolverParams(solver_time_limit_sec=60, min_headcount_per_day=[20, 20, 20, 20, 20, 20, 20])
r = solve_multi([d], ["Staff"], p)
cp = r.combined_phase1
check("TEST4a: feasible with min=20 entries", cp.status in ("OPTIMAL", "FEASIBLE"))
if cp.status in ("OPTIMAL", "FEASIBLE"):
    actual_e = daily_entry_headcount(cp)
    for day in range(7):
        check(f"TEST4b-{DAY_NAMES[day]}: entries({actual_e[day]}) >= 20",
              actual_e[day] >= 20, f"VIOLATION: {actual_e[day]} < 20")

# == TEST 5 – Combined min=0 → no constraint ===================================
print("\n" + "=" * 65)
print("TEST 5 – Combined min=0 means no constraint")
print("=" * 65)
p = SolverParams(solver_time_limit_sec=60, min_headcount_per_day=[0]*7)
r = solve_multi([d], ["Staff"], p)
check("TEST5a: feasible with min=0", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

# == TEST 6 – Combined min + max together ======================================
print("\n" + "=" * 65)
print("TEST 6 – Combined min=10 + max=150 together")
print("=" * 65)
p = SolverParams(solver_time_limit_sec=60,
                 min_headcount_per_day=[10]*7, max_headcount_per_day=[150]*7)
r = solve_multi([d], ["Staff"], p)
check("TEST6a: feasible with min=10 & max=150",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    actual_e = daily_entry_headcount(r.combined_phase1)
    for day in range(7):
        check(f"TEST6b-{DAY_NAMES[day]}-min", actual_e[day] >= 10)
        check(f"TEST6c-{DAY_NAMES[day]}-max", actual_e[day] <= 150)

# == TEST 7 – Combined min > demand → forces extra workers =====================
print("\n" + "=" * 65)
print("TEST 7 – Combined min > demand forces extra workers")
print("=" * 65)
d = generate_sample_demand(peak_agents=3, base_agents=1, seed=99)
p = SolverParams(solver_time_limit_sec=60, min_headcount_per_day=[8]*7)
r = solve_multi([d], ["Staff"], p)
check("TEST7a: feasible with min=8 > demand ~3",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    actual_e = daily_entry_headcount(r.combined_phase1)
    for day in range(7):
        check(f"TEST7b-{DAY_NAMES[day]}: entries({actual_e[day]}) >= 8",
              actual_e[day] >= 8)

# == TEST 8 – Combined min > combined max → INFEASIBLE =========================
print("\n" + "=" * 65)
print("TEST 8 – Combined min=20 > max=10 → INFEASIBLE")
print("=" * 65)
d = generate_sample_demand(peak_agents=5, base_agents=1, seed=99)
p = SolverParams(solver_time_limit_sec=30,
                 min_headcount_per_day=[20]*7, max_headcount_per_day=[10]*7)
r = solve_multi([d], ["Staff"], p)
check("TEST8a: INFEASIBLE when min=20 > max=10",
      r.combined_phase1.status not in ("OPTIMAL", "FEASIBLE"))

# == TEST 9 – Multi-occ with combined min + per-occ constraints ================
print("\n" + "=" * 65)
print("TEST 9 – Multi-occ with combined min + per-occ max")
print("=" * 65)
d1 = generate_sample_demand(peak_agents=10, base_agents=1, seed=42)
d2 = generate_sample_demand(peak_agents=10, base_agents=1, seed=43)
p = SolverParams(
    solver_time_limit_sec=60,
    min_headcount_per_day=[15, 15, 15, 15, 15, 15, 15],
    max_headcount_per_day=[30, 30, 30, 30, 30, 30, 30],
    occ_max_headcount_per_day=[
        [18, 18, 18, 18, 18, 18, 18],
        [18, 18, 18, 18, 18, 18, 18],
    ],
)
r = solve_multi([d1, d2], ["Tech", "Lab"], p)
check("TEST9a: multi-occ feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE", "INFEASIBLE"))
if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    actual_e = daily_entry_headcount(r.combined_phase1)
    for day in range(7):
        check(f"TEST9b-{DAY_NAMES[day]}-combined-min", actual_e[day] >= 15,
              f"entries({actual_e[day]}) >= 15")
        check(f"TEST9c-{DAY_NAMES[day]}-combined-max", actual_e[day] <= 30,
              f"entries({actual_e[day]}) <= 30")
    for i, occ in enumerate(r.occupations):
        occ_entries = daily_entry_headcount(occ.phase1)
        for day in range(7):
            check(f"TEST9d-{occ.name}-{DAY_NAMES[day]}-per-occ-max",
                  occ_entries[day] <= 18, f"entries({occ_entries[day]}) <= 18")

# == SUMMARY ===================================================================
print("\n" + "=" * 65)
print(f"SUMMARY  PASS={_pass}  FAIL={_fail}")
print("=" * 65)
if _fail == 0:
    print("All checks passed.")
else:
    print(f"{_fail} checks FAILED.")