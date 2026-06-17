"""
Integration test – ALL headcount parameter types simultaneously
with 5 different occupations, verifying every constraint holds.
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


def flat_demand(val, start_h=4, dur_h=10):
    """Demand from start_h to start_h+dur_h each day."""
    arr = np.zeros(TOTAL_INTERVALS, dtype=int)
    for day in range(7):
        s = day * INTERVALS_PER_DAY + start_h * INTERVALS_PER_HOUR
        e = s + dur_h * INTERVALS_PER_HOUR
        arr[s:e] = val
    return arr


# ══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("INTEGRATION — 5 occupations, ALL headcount constraints active")
print("=" * 65)

# ── Demand curves (5 distinct occupations) ──────────────────────────────
occ_names = ["Technician", "Labourer", "Helper", "Supervisor", "Assistant"]
d1 = flat_demand(12, 6, 14)   # Technician: 12 peak
d2 = flat_demand(8,  8, 8)    # Labourer
d3 = flat_demand(6,  6, 10)   # Helper
d4 = flat_demand(4,  7, 9)    # Supervisor
d5 = flat_demand(4,  8, 10)   # Assistant
demands_all = [d1, d2, d3, d4, d5]

# ── Constraint matrix (all types active) ────────────────────────────────
# Combined max per day: generous upper bound
combined_max = [50, 50, 50, 50, 50, 40, 40]
# Combined min per day: low but non-zero
combined_min = [5, 5, 5, 5, 5, 2, 2]

# Per-occ max per day (some tight, some loose)
occ_max = [
    [18, 18, 18, 18, 18, 14, 14],  # Technician: max 18
    [14, 14, 14, 14, 14, 10, 10],  # Labourer
    [12, 12, 12, 12, 12, 8,  8],   # Helper
    [10, 10, 10, 10, 10, 6,  6],   # Supervisor
    [10, 10, 10, 10, 10, 6,  6],   # Assistant
]

# Per-occ min per day (some tight, some zero)
occ_min = [
    [3, 3, 3, 3, 3, 0, 0],  # Technician: min 3 weekdays
    [2, 2, 2, 2, 2, 0, 0],  # Labourer
    None,                     # Helper: no min
    [1, 1, 1, 1, 1, 0, 0],  # Supervisor
    None,                     # Assistant: no min
]

# Per-occ shift-code constraints
occ_shift_code = [
    # Technician: max 8 on 0800-1600, min 2 on 0600-1400
    {"0800-1600": {"max": [8, 8, 8, 8, 8, 6, 6]},
     "0600-1400": {"min": [2, 2, 2, 2, 2, 0, 0]}},
    # Labourer: max 6 on 0800-1600
    {"0800-1600": {"max": [6, 6, 6, 6, 6, 5, 5]}},
    # Helper: max 5 on 0600-1400
    {"0600-1400": {"max": [5, 5, 5, 5, 5, 4, 4]}},
    None,  # Supervisor: no shift-code constraints
    None,  # Assistant: no shift-code constraints
]

# Entry/Exit limits (generous)
entry_limits = [12, 12, 12, 12, 12, 10, 10]
exit_limits  = [12, 12, 12, 12, 12, 10, 10]

params = SolverParams(
    min_shift_hours=3.0,
    max_shift_hours=12.0,
    solver_time_limit_sec=120,
    max_headcount_per_day=combined_max,
    min_headcount_per_day=combined_min,
    occ_max_headcount_per_day=occ_max,
    occ_min_headcount_per_day=occ_min,
    occ_headcount_per_shift_code=occ_shift_code,
    max_entries_per_day=entry_limits,
    max_exits_per_day=exit_limits,
)

print("\nSolving with all constraints active...")
result = solve_multi(demands_all, occ_names, params)

# ── Basic feasibility check ────────────────────────────────────────────
cp1 = result.combined_phase1
print(f"Solver status: {cp1.status}  time={cp1.elapsed_sec:.1f}s")
check("INT-0: solver found solution",
      cp1.status in ("OPTIMAL", "FEASIBLE"),
      f"status={cp1.status}")

if cp1.status not in ("OPTIMAL", "FEASIBLE"):
    print("\nFATAL: No feasible solution – other checks skipped.")
else:
    # ── 1. Combined max headcount per day ──────────────────────────────
    print("\n--- 1. Combined max headcount per day ---")
    cov = cp1.coverage
    for day in range(7):
        s, e = day * INTERVALS_PER_DAY, (day + 1) * INTERVALS_PER_DAY
        mx = int(cov[s:e].max())
        limit = combined_max[day]
        check(f"INT1-{DAY_NAMES[day]}: combined_max({mx}) <= {limit}", mx <= limit,
              f"VIOLATION: {mx} > {limit}")

    # ── 2. Combined min headcount per day ──────────────────────────────
    print("\n--- 2. Combined min headcount per day ---")
    for day in range(7):
        s, e = day * INTERVALS_PER_DAY, (day + 1) * INTERVALS_PER_DAY
        mn = int(cov[s:e].min())
        limit = combined_min[day]
        check(f"INT2-{DAY_NAMES[day]}: combined_min({mn}) >= {limit}", mn >= limit,
              f"VIOLATION: {mn} < {limit}")

    # ── 3. Per-occupation max headcount ────────────────────────────────
    print("\n--- 3. Per-occupation max headcount ---")
    for i, (occ, occ_limits) in enumerate(zip(result.occupations, occ_max)):
        ocov = occ.phase1.coverage
        for day in range(7):
            s, e = day * INTERVALS_PER_DAY, (day + 1) * INTERVALS_PER_DAY
            mx = int(ocov[s:e].max())
            limit = occ_limits[day]
            check(f"INT3-{occ.name}-{DAY_NAMES[day]}: max({mx}) <= {limit}",
                  mx <= limit, f"VIOLATION: {mx} > {limit}")

    # ── 4. Per-occupation min headcount ────────────────────────────────
    print("\n--- 4. Per-occupation min headcount ---")
    for i, (occ, occ_mins) in enumerate(zip(result.occupations, occ_min)):
        if occ_mins is None:
            continue
        ocov = occ.phase1.coverage
        for day in range(7):
            mn_req = occ_mins[day]
            if mn_req <= 0:
                continue
            s, e = day * INTERVALS_PER_DAY, (day + 1) * INTERVALS_PER_DAY
            mn = int(ocov[s:e].min())
            check(f"INT4-{occ.name}-{DAY_NAMES[day]}: min({mn}) >= {mn_req}",
                  mn >= mn_req, f"VIOLATION: {mn} < {mn_req}")

    # ── 5. Per-occupation shift-code headcount ─────────────────────────
    print("\n--- 5. Per-occupation shift-code headcount ---")
    for i, (occ, sc) in enumerate(zip(result.occupations, occ_shift_code)):
        if sc is None:
            continue
        for code, spec in sc.items():
            if isinstance(spec, list):
                max_limits = spec
                min_limits = None
            else:
                max_limits = spec.get("max")
                min_limits = spec.get("min")
            # Count actual workers per day for this code
            daily = [0] * 7
            for s, cnt in zip(occ.phase1.shifts, occ.phase1.counts):
                if s.shift_code == code and cnt > 0:
                    daily[s.day] += cnt
            for day in range(7):
                if max_limits and max_limits[day] > 0:
                    check(f"INT5a-{occ.name}-{code}-{DAY_NAMES[day]}: "
                          f"workers({daily[day]}) <= max({max_limits[day]})",
                          daily[day] <= max_limits[day],
                          f"VIOLATION: {daily[day]} > {max_limits[day]}")
                if min_limits and min_limits[day] > 0:
                    check(f"INT5b-{occ.name}-{code}-{DAY_NAMES[day]}: "
                          f"workers({daily[day]}) >= min({min_limits[day]})",
                          daily[day] >= min_limits[day],
                          f"VIOLATION: {daily[day]} < {min_limits[day]}")

    # ── 6. Entry / Exit limits ─────────────────────────────────────────
    print("\n--- 6. Entry / Exit limits ---")
    from collections import defaultdict
    day_entry_times = defaultdict(set)
    day_exit_times = defaultdict(set)
    for s, cnt in zip(cp1.shifts, cp1.counts):
        if cnt > 0:
            day_entry_times[s.day].add(s.start_interval)
            end_ivl = s.start_interval + s.duration_intervals
            day_exit_times[s.day].add(end_ivl)
    for day in range(7):
        check(f"INT6a-{DAY_NAMES[day]}: entries({len(day_entry_times[day])}) <= {entry_limits[day]}",
              len(day_entry_times[day]) <= entry_limits[day],
              f"VIOLATION: {len(day_entry_times[day])} > {entry_limits[day]}")
        check(f"INT6b-{DAY_NAMES[day]}: exits({len(day_exit_times[day])}) <= {exit_limits[day]}",
              len(day_exit_times[day]) <= exit_limits[day],
              f"VIOLATION: {len(day_exit_times[day])} > {exit_limits[day]}")

    # ── 7. Demand coverage ─────────────────────────────────────────────
    print("\n--- 7. Demand coverage (all occupations) ---")
    for i, occ in enumerate(result.occupations):
        deficit = np.maximum(occ.demand - occ.phase1.coverage, 0)
        deficit_count = int(np.count_nonzero(deficit))
        max_deficit = int(deficit.max()) if deficit_count > 0 else 0
        check(f"INT7-{occ.name}: demand covered (deficit={deficit_count} intervals, max={max_deficit})",
              deficit_count == 0,
              f"UNDER-COVERAGE: {deficit_count} intervals max gap={max_deficit}")

    # ── 8. Per-occupation coverage equals combined ─────────────────────
    print("\n--- 8. Combined coverage == sum of per-occ coverage ---")
    per_occ_sum = np.zeros(TOTAL_INTERVALS, dtype=int)
    for occ in result.occupations:
        per_occ_sum += occ.phase1.coverage
    match = np.all(per_occ_sum == cp1.coverage)
    mismatches = int(np.count_nonzero(per_occ_sum != cp1.coverage))
    check(f"INT8: combined coverage matches sum (mismatches={mismatches})",
          match, f"{mismatches} intervals differ")

    # ── 9. Per-occ headcount + shift-code headcount consistency ───────
    print("\n--- 9. Per-occ headcount <= shift-code max + consistency ---")
    for i, occ in enumerate(result.occupations):
        occ_limit_list = occ_max[i]
        for day in range(7):
            s, e = day * INTERVALS_PER_DAY, (day + 1) * INTERVALS_PER_DAY
            mx = int(occ.phase1.coverage[s:e].max())
            # The per-occ max must always be >= actual peak
            check(f"INT9-{occ.name}-{DAY_NAMES[day]}: per-occ-max({mx}) <= limit({occ_limit_list[day]})",
                  mx <= occ_limit_list[day])

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("INTEGRATION — 3 occupations, adversarial tight limits")
print("=" * 65)

# Tight scenario: narrow min/max bands, shift-code limits force specific codes
d1 = flat_demand(6, 6, 12)
d2 = flat_demand(6, 6, 12)
d3 = flat_demand(6, 6, 12)

params_tight = SolverParams(
    min_shift_hours=4.0,
    max_shift_hours=10.0,
    solver_time_limit_sec=60,
    max_headcount_per_day=[22, 22, 22, 22, 22, 18, 18],     # generous combined max
    min_headcount_per_day=[8, 8, 8, 8, 8, 4, 4],             # moderate combined min
    occ_max_headcount_per_day=[
        [12, 12, 12, 12, 12, 9, 9],
        [12, 12, 12, 12, 12, 9, 9],
        [12, 12, 12, 12, 12, 9, 9],
    ],
    occ_min_headcount_per_day=[
        [2, 2, 2, 2, 2, 1, 1],
        [2, 2, 2, 2, 2, 1, 1],
        [2, 2, 2, 2, 2, 1, 1],
    ],
    occ_headcount_per_shift_code=[
        # Occ 0: 0800-1600 max 7
        {"0800-1600": {"max": [7, 7, 7, 7, 7, 6, 6]}},
        # Occ 1: 0600-1400 max 6, min 2
        {"0600-1400": {"max": [6, 6, 6, 6, 6, 5, 5],
                        "min": [2, 2, 2, 2, 2, 1, 1]}},
        # Occ 2: 1000-1800 max 6
        {"1000-1800": {"max": [6, 6, 6, 6, 6, 5, 5]}},
    ],
    max_entries_per_day=[10, 10, 10, 10, 10, 8, 8],
    max_exits_per_day=[10, 10, 10, 10, 10, 8, 8],
)

print("\nSolving tight scenario...")
r2 = solve_multi([d1, d2, d3], ["A", "B", "C"], params_tight)
cp2 = r2.combined_phase1
print(f"Solver status: {cp2.status}  time={cp2.elapsed_sec:.1f}s")

check("INT-T1: tight scenario feasible", cp2.status in ("OPTIMAL", "FEASIBLE"),
      f"status={cp2.status}")

if cp2.status in ("OPTIMAL", "FEASIBLE"):
    cov2 = cp2.coverage
    # Combined max
    for day in range(7):
        s, e = day * INTERVALS_PER_DAY, (day + 1) * INTERVALS_PER_DAY
        mx = int(cov2[s:e].max())
        check(f"INT-T2-{DAY_NAMES[day]}: combined_max({mx}) <= {params_tight.max_headcount_per_day[day]}",
              mx <= params_tight.max_headcount_per_day[day])
        mn = int(cov2[s:e].min())
        check(f"INT-T3-{DAY_NAMES[day]}: combined_min({mn}) >= {params_tight.min_headcount_per_day[day]}",
              mn >= params_tight.min_headcount_per_day[day])

    # Per-occ + shift-code
    for i, (occ, occ_max_l, occ_min_l, sc) in enumerate(
        zip(r2.occupations,
            params_tight.occ_max_headcount_per_day,
            params_tight.occ_min_headcount_per_day,
            params_tight.occ_headcount_per_shift_code)):
        ocov = occ.phase1.coverage
        for day in range(7):
            s, e = day * INTERVALS_PER_DAY, (day + 1) * INTERVALS_PER_DAY
            mx = int(ocov[s:e].max())
            mn = int(ocov[s:e].min())
            check(f"INT-T4-{occ.name}-{DAY_NAMES[day]}: max({mx}) <= {occ_max_l[day]}",
                  mx <= occ_max_l[day])
            if occ_min_l[day] > 0:
                check(f"INT-T5-{occ.name}-{DAY_NAMES[day]}: min({mn}) >= {occ_min_l[day]}",
                      mn >= occ_min_l[day])
        if sc:
            for code, spec in sc.items():
                if isinstance(spec, list):
                    max_l = spec; min_l = None
                else:
                    max_l = spec.get("max"); min_l = spec.get("min")
                daily = [0] * 7
                for s, cnt in zip(occ.phase1.shifts, occ.phase1.counts):
                    if s.shift_code == code and cnt > 0:
                        daily[s.day] += cnt
                for day in range(7):
                    if max_l and max_l[day] > 0:
                        check(f"INT-T6-{occ.name}-{code}-{DAY_NAMES[day]}: {daily[day]} <= {max_l[day]}",
                              daily[day] <= max_l[day])
                    if min_l and min_l[day] > 0:
                        check(f"INT-T7-{occ.name}-{code}-{DAY_NAMES[day]}: {daily[day]} >= {min_l[day]}",
                              daily[day] >= min_l[day])


# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("INTEGRATION — Edge: ZERO demand with ALL constraints active")
print("=" * 65)

zero_d = np.zeros(TOTAL_INTERVALS, dtype=int)
params_zero = SolverParams(
    solver_time_limit_sec=30,
    max_headcount_per_day=[5, 5, 5, 5, 5, 5, 5],
    min_headcount_per_day=[0, 0, 0, 0, 0, 0, 0],  # min 0 = no min
    occ_max_headcount_per_day=[[3, 3, 3, 3, 3, 3, 3]],
    occ_headcount_per_shift_code=[
        {"0800-1600": {"max": [2, 2, 2, 2, 2, 2, 2]}}
    ],
)

print("\nSolving zero-demand scenario...")
rz = solve_multi([zero_d], ["Staff"], params_zero)
cpz = rz.combined_phase1
print(f"Solver status: {cpz.status}")

check("INT-Z1: zero-demand feasible", cpz.status in ("OPTIMAL", "FEASIBLE"))
if cpz.status in ("OPTIMAL", "FEASIBLE"):
    # All shifts should have count 0 (no demand to cover)
    total_assigned = sum(rz.occupations[0].phase1.counts)
    check("INT-Z2: zero workers assigned for zero demand", total_assigned == 0,
          f"got {total_assigned} workers")
    # Max constraint should hold (0 <= 5)
    mx = int(cpz.coverage.max())
    check(f"INT-Z3: max_headcount({mx}) <= 5", mx <= 5)


# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("INTEGRATION — Edge: IMPOSSIBLE combination (min > max)")
print("=" * 65)

d_edge = flat_demand(3, 6, 8)
params_impossible = SolverParams(
    solver_time_limit_sec=15,
    min_headcount_per_day=[50, 50, 50, 50, 50, 50, 50],   # impossibly high min
    max_headcount_per_day=[10, 10, 10, 10, 10, 10, 10],   # impossibly low max
)

print("\nSolving impossible scenario...")
ri = solve_multi([d_edge], ["Staff"], params_impossible)
cpi = ri.combined_phase1
print(f"Solver status: {cpi.status}")
check("INT-I1: min=50 > max=10 → INFEASIBLE",
      cpi.status not in ("OPTIMAL", "FEASIBLE"),
      f"unexpected status={cpi.status}")

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"SUMMARY  PASS={_pass}  FAIL={_fail}")
print("=" * 65)
if _fail == 0:
    print("All integration tests passed – every constraint type works together.")
else:
    print(f"{_fail} checks FAILED – see above for details.")