"""
Headcount constraint + display correctness tests.

Three distinct values for a given day:
  entries     = sum(shift_count[s] for s where s.start_day == d)
  peak_sim    = max(coverage[day_start : day_end])   <- simultaneous workers
  constraint  = max_headcount_per_day[d]

Bugs being investigated:
  A) display bug:  entries > constraint  (even though peak_sim <= constraint)
  B) real bug:     peak_sim  > constraint (the CP-SAT constraint is violated)

Expected after fix:
  - peak_sim  <= constraint  on every day  (constraint holds)
  - displayed headcount == peak_sim  (not entries)
"""

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
    global _pass
    _pass += 1
    print(f"  [PASS] {label}: {value}")

def fail(label, got, detail=""):
    global _fail
    _fail += 1
    print(f"  [FAIL] {label}: {got}  {detail}")

def check(label, cond, detail=""):
    if cond:
        ok(label, "True")
    else:
        fail(label, "False", detail)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 1 – Unconstrained baseline
#          Expose the entries vs peak_sim discrepancy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("TEST 1 – entries vs peak_simultaneous (unconstrained solve)")
print("=" * 65)

d = generate_sample_demand(peak_agents=25, base_agents=2, seed=42)
r0 = solve_multi([d], ["Staff"], SolverParams(solver_time_limit_sec=60))
cp1 = r0.combined_phase1
print(f"Status: {cp1.status}")

entries   = daily_entry_headcount(cp1)
peak_sims = [int(cp1.coverage[day*INTERVALS_PER_DAY:(day+1)*INTERVALS_PER_DAY].max())
             for day in range(7)]

print(f"\n  {'Day':<11} {'Entries':>9} {'PeakSim':>9} {'Entries>PeakSim':>16}")
print("  " + "-" * 50)
entries_overstate_any = False
for day in range(7):
    over = entries[day] > peak_sims[day]
    if over:
        entries_overstate_any = True
    print(f"  {DAY_NAMES[day]:<11} {entries[day]:>9} {peak_sims[day]:>9} {'YES <-- overstated' if over else 'no':>16}")

print()
check("TEST1a: entries overstate peak_sim on at least one day (expected)",
      entries_overstate_any,
      "(this is why displayed headcount > constraint limit)")
check("TEST1b: max_headcount() == max(entries)",
      max_headcount(cp1) == max(entries))
check("TEST1c: peak_sim == coverage.max()",
      max(peak_sims) == int(cp1.coverage.max()))
# These should differ whenever multiple overlapping shifts cover the same day
check("TEST1d: max(entries) >= max(peak_sims) always",
      max(entries) >= max(peak_sims))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 2 – Constraint correctness
#          Set limit to unconstrained peak_sim → should stay feasible
#          and never violate the limit in actual coverage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("TEST 2 – Constraint holds in solution (limit = unconstrained peak)")
print("=" * 65)

# Use the unconstrained peak_sim per day as the limit – must stay feasible
# and satisfy the constraint exactly
hc_limit = [max(ps, 1) for ps in peak_sims]
params_cap = SolverParams(
    solver_time_limit_sec=60,
    max_headcount_per_day=hc_limit,
)
r2 = solve_multi([d], ["Staff"], params_cap)
cp2 = r2.combined_phase1
print(f"Status: {cp2.status}  (limit={hc_limit})")

check("TEST2a: solver finds a solution", cp2.status in ("OPTIMAL", "FEASIBLE"))

for day in range(7):
    s, e = day * INTERVALS_PER_DAY, (day + 1) * INTERVALS_PER_DAY
    actual_sim = int(cp2.coverage[s:e].max())
    limit = hc_limit[day]
    check(
        f"TEST2b-{DAY_NAMES[day]}: actual_sim({actual_sim}) <= limit({limit})",
        actual_sim <= limit,
        f"VIOLATION: {actual_sim} workers simultaneous but limit={limit}",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 3 – Tight constraint: set limit to 70% of peak_sim
#          Verify the constraint is actually binding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("TEST 3 – Tight constraint (70% of unconstrained peak per day)")
print("=" * 65)

hc_tight = [max(1, int(ps * 0.70)) for ps in peak_sims]
params_tight = SolverParams(
    solver_time_limit_sec=90,
    max_headcount_per_day=hc_tight,
)
r3 = solve_multi([d], ["Staff"], params_tight)
cp3 = r3.combined_phase1
print(f"Status: {cp3.status}  (limits={hc_tight})")

if cp3.status in ("OPTIMAL", "FEASIBLE"):
    real_violations = []
    for day in range(7):
        s, e = day * INTERVALS_PER_DAY, (day + 1) * INTERVALS_PER_DAY
        actual_sim = int(cp3.coverage[s:e].max())
        limit = hc_tight[day]
        if actual_sim > limit:
            real_violations.append((DAY_NAMES[day], actual_sim, limit))
        check(
            f"TEST3-{DAY_NAMES[day]}: sim({actual_sim}) <= tight_limit({limit})",
            actual_sim <= limit,
            "REAL CONSTRAINT BUG" if actual_sim > limit else "",
        )
    if real_violations:
        print(f"\n  *** REAL CONSTRAINT VIOLATIONS FOUND: ***")
        for day_name, sim, lim in real_violations:
            print(f"      {day_name}: {sim} simultaneous workers but limit={lim}")
    else:
        print("  All days within tight limit – constraint is working correctly.")
else:
    print(f"  Solver returned {cp3.status} – constraint may be infeasible at 70%.")
    print("  This is acceptable (tight limit makes covering infeasible).")
    ok("TEST3: INFEASIBLE at 70% is a valid outcome", cp3.status)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 4 – Replicate user scenario: reported HC >> limit
#          Shows displayed entries vs actual simultaneous
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("TEST 4 – Reproduce reported issue: HC display >> limit")
print("=" * 65)

LIMIT = 41
params_41 = SolverParams(
    solver_time_limit_sec=90,
    max_headcount_per_day=[LIMIT] * 7,
)
r4 = solve_multi([d], ["Staff"], params_41)
cp4 = r4.combined_phase1
print(f"Status: {cp4.status}  (limit={LIMIT} on all days)")

if cp4.status in ("OPTIMAL", "FEASIBLE"):
    entries4   = daily_entry_headcount(cp4)
    peak_sims4 = [int(cp4.coverage[day*INTERVALS_PER_DAY:(day+1)*INTERVALS_PER_DAY].max())
                  for day in range(7)]
    print(f"\n  {'Day':<11} {'Entries':>9} {'PeakSim':>9} {'Limit':>6} {'Sim OK?':>8} {'Disp BUG?':>10}")
    print("  " + "-" * 60)
    display_bugs = []
    constraint_bugs = []
    for day in range(7):
        sim_ok   = peak_sims4[day] <= LIMIT
        disp_bug = entries4[day]   > LIMIT
        if not sim_ok:
            constraint_bugs.append((DAY_NAMES[day], peak_sims4[day]))
        if disp_bug:
            display_bugs.append((DAY_NAMES[day], entries4[day]))
        print(f"  {DAY_NAMES[day]:<11} {entries4[day]:>9} {peak_sims4[day]:>9} {LIMIT:>6}"
              f"  {'OK' if sim_ok else 'VIOLATION':>8}  {'BUG' if disp_bug else 'ok':>10}")

    print()
    if constraint_bugs:
        print("  *** REAL CONSTRAINT BUGS (peak_sim > limit): ***")
        for dn, v in constraint_bugs:
            fail(f"TEST4a constraint {dn}", f"sim={v} > limit={LIMIT}", "REAL BUG")
    else:
        ok("TEST4a: constraint correct – peak_sim <= limit on every day", True)

    if display_bugs:
        print("\n  *** DISPLAY BUGS (entries > limit, but sim is OK): ***")
        for dn, v in display_bugs:
            fail(f"TEST4b display {dn}", f"entries={v} > limit={LIMIT}",
                 "reported by max_headcount() – fix: use peak_sim not entries")
    else:
        ok("TEST4b: no display overstatement on any day", True)
else:
    print(f"  Status: {cp4.status} – the demand cannot be covered within HC={LIMIT}.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print(f"SUMMARY  PASS={_pass}  FAIL={_fail}")
print("=" * 65)
if _fail == 0:
    print("All checks passed.")
else:
    print("Some checks FAILED – see above for details.")
