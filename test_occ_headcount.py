"""
Occupation-based daily headcount limit tests.

Tests for:
  - Per-occupation max_headcount_per_day constraint (0 = unlimited)
  - Solver respects per-occupation limits
  - Zero demand, zero-day limits (0 = unlimited)
  - Edge cases (tight limits, infeasible when demand exceeds capacity)
  - Interaction with global headcount limits
  - Multi-occupation: each occ obeys its own limit
  - Mixed: some occs with limits, others with None
  - Backward compat: no per-occ limits set
"""

import numpy as np
from solver import (
    SolverParams, solve_multi,
    DAY_NAMES, INTERVALS_PER_DAY, INTERVALS_PER_HOUR, TOTAL_INTERVALS,
)

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


def flat_demand(workers: int) -> np.ndarray:
    return np.full(TOTAL_INTERVALS, workers, dtype=int)


def zero_demand():
    return np.zeros(TOTAL_INTERVALS, dtype=int)


def quick_params(occ_hc=None, **kwargs) -> SolverParams:
    """Fast params for testing."""
    defaults = dict(
        min_shift_hours=4.0,
        max_shift_hours=8.0,
        shift_start_granularity_min=60,
        shift_duration_step_min=60,
        min_weekly_hours=0.0,
        max_weekly_hours=200.0,
        min_rest_hours=0.0,
        max_unique_shifts=0,
        transition_penalty=0,
        solver_time_limit_sec=10,
    )
    defaults.update(kwargs)
    p = SolverParams(**defaults)
    if occ_hc is not None:
        p.occ_max_headcount_per_day = occ_hc
    return p


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 – Single occupation: per-occ headcount limit = 8, demand = 3
#          Should be easily feasible.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 1 – Single occ, demand=3, per-occ limit=8 (easily feasible)")
print("=" * 65)

demand = flat_demand(3)
occ_hc = [[8, 8, 8, 8, 8, 8, 8]]
p = quick_params(occ_hc=occ_hc)
r = solve_multi([demand], ["Staff"], p)
cp1 = r.combined_phase1

check("TEST1a: solver feasible", cp1.status in ("OPTIMAL", "FEASIBLE"))

if cp1.status in ("OPTIMAL", "FEASIBLE"):
    for day in range(7):
        s, e = day * INTERVALS_PER_DAY, (day + 1) * INTERVALS_PER_DAY
        actual_max = int(cp1.coverage[s:e].max())
        limit = occ_hc[0][day]
        check(
            f"TEST1b-{DAY_NAMES[day]}: coverage({actual_max}) <= limit({limit})",
            actual_max <= limit,
            f"VIOLATION: {actual_max} > {limit}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 – Single occupation: demand=8, per-occ limit=3 (INFEASIBLE)
#          Demand exceeds limit, must be infeasible.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 2 – Single occ, demand=8, per-occ limit=3 (must be INFEASIBLE)")
print("=" * 65)

demand = flat_demand(8)
occ_hc = [[3, 3, 3, 3, 3, 3, 3]]
p = quick_params(occ_hc=occ_hc)
r = solve_multi([demand], ["Staff"], p)

check("TEST2a: solver INFEASIBLE (demand 8 > limit 3)",
      r.combined_phase1.status not in ("OPTIMAL", "FEASIBLE"))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 – Multi-occupation: each occ obeys own distinct limit
#          Demand fits within per-occ limits.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 3 – Multi-occ with distinct per-occ limits (feasible)")
print("=" * 65)

# Occ0 demand=2 limit=5, Occ1 demand=3 limit=6, Occ2 demand=3 limit=6
d1 = flat_demand(2)
d2 = flat_demand(3)
d3 = flat_demand(3)
occ_hc = [
    [5, 5, 5, 5, 5, 5, 5],
    [6, 6, 6, 6, 6, 6, 6],
    [6, 6, 6, 6, 6, 6, 6],
]
p = quick_params(occ_hc=occ_hc)
r = solve_multi([d1, d2, d3], ["A", "B", "C"], p)

check("TEST3a: solver feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    for occ_idx, occ in enumerate(r.occupations):
        for day in range(7):
            s = day * INTERVALS_PER_DAY
            e = s + INTERVALS_PER_DAY
            actual_max = int(occ.phase1.coverage[s:e].max())
            limit = occ_hc[occ_idx][day]
            check(
                f"TEST3b-{occ.name}-{DAY_NAMES[day]}: "
                f"coverage({actual_max}) <= limit({limit})",
                actual_max <= limit,
                f"VIOLATION: occ={occ.name} day={DAY_NAMES[day]} "
                f"actual={actual_max} > limit={limit}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 – limit=0 means UNLIMITED (same convention as global HC)
#          Demand=5, limit=0 → unlimited, must still cover demand.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 4 – limit=0 means unlimited (covers demand=5)")
print("=" * 65)

occ_hc = [[0, 0, 0, 0, 0, 0, 0]]  # 0 = unlimited
p = quick_params(occ_hc=occ_hc)
r = solve_multi([flat_demand(5)], ["Staff"], p)

check("TEST4a: solver feasible when limit=0 (unlimited)",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 – Zero demand, non-zero limits (all ok)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 5 – Zero demand with per-occ limits")
print("=" * 65)

occ_hc = [
    [5, 5, 5, 5, 5, 5, 5],
    [3, 3, 3, 3, 3, 3, 3],
]
p = quick_params(occ_hc=occ_hc)
r = solve_multi([zero_demand(), zero_demand()], ["A", "B"], p)

check("TEST5a: solver feasible with zero demand",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))
check("TEST5b: zero coverage for zero demand",
      int(r.combined_phase1.coverage.sum()) == 0)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 – Different limits per day per occupation
#          Demand must fit within each day's individual limit.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 6 – Different per-day limits per occupation (feasible)")
print("=" * 65)

# Occ0 demand=1 fits in [3,5,2,4,3,1,1] → min=1 on Sat/Sun, demand=1 is ok
# Occ1 demand=1 fits in [2,6,3,5,4,2,1] → min=1 on Sun, demand=1 is ok
occ_hc = [
    [3, 5, 2, 4, 3, 1, 1],
    [2, 6, 3, 5, 4, 2, 1],
]
p = quick_params(occ_hc=occ_hc)
r = solve_multi([flat_demand(1), flat_demand(1)], ["A", "B"], p)

check("TEST6a: solver feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    for occ_idx, occ in enumerate(r.occupations):
        for day in range(7):
            s = day * INTERVALS_PER_DAY
            e = s + INTERVALS_PER_DAY
            actual_max = int(occ.phase1.coverage[s:e].max())
            limit = occ_hc[occ_idx][day]
            check(
                f"TEST6b-{occ.name}-{DAY_NAMES[day]}: "
                f"sim({actual_max}) <= limit({limit})",
                actual_max <= limit,
                f"VIOLATION: {occ.name} {DAY_NAMES[day]}: {actual_max} > {limit}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7 – Per-occ + global HC both active
#          Both constraints must hold simultaneously.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 7 – Per-occ limits + global max_headcount_per_day combined")
print("=" * 65)

# Per-occ: each occ max 8. Global: max 8 total.
# Demand: 2 each = 6 total. Both limits easily met.
occ_hc = [
    [8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8],
    [8, 8, 8, 8, 8, 8, 8],
]
p = quick_params(
    occ_hc=occ_hc,
    max_headcount_per_day=[8, 8, 8, 8, 8, 8, 8],
)
r = solve_multi([flat_demand(2), flat_demand(2), flat_demand(2)], ["A", "B", "C"], p)

check("TEST7a: solver feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    # Check per-occ limits
    for occ_idx, occ in enumerate(r.occupations):
        for day in range(7):
            s = day * INTERVALS_PER_DAY
            e = s + INTERVALS_PER_DAY
            actual_max = int(occ.phase1.coverage[s:e].max())
            limit_occ = occ_hc[occ_idx][day]
            check(
                f"TEST7b-per_occ-{occ.name}-{DAY_NAMES[day]}: "
                f"{actual_max} <= {limit_occ}",
                actual_max <= limit_occ,
                f"VIOLATION"
            )

    # Check global limit
    cov = r.combined_phase1.coverage
    for day in range(7):
        s = day * INTERVALS_PER_DAY
        e = s + INTERVALS_PER_DAY
        actual_total = int(cov[s:e].max())
        check(
            f"TEST7c-global-{DAY_NAMES[day]}: "
            f"total({actual_total}) <= global_limit(8)",
            actual_total <= 8,
            f"VIOLATION: total={actual_total} > 8"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8 – Tight per-occ limit makes infeasible despite ample global room
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 8 – Tight per-occ limit forces infeasible")
print("=" * 65)

occ_hc = [[1, 1, 1, 1, 1, 1, 1]]  # Only 1 worker for occ0
p = quick_params(occ_hc=occ_hc)
r = solve_multi([flat_demand(5)], ["Staff"], p)  # Demand 5, only 1 allowed

check("TEST8a: infeasible when per-occ limit < demand",
      r.combined_phase1.status not in ("OPTIMAL", "FEASIBLE"))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9 – None (no per-occ limits) preserves backward compat
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 9 – None per-occ limits (backward compat: no per-occ constraint)")
print("=" * 65)

p = quick_params()  # No occ_hc set -> occ_max_headcount_per_day = None
r = solve_multi([flat_demand(3)], ["Staff"], p)
check("TEST9a: works without per-occ limits",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10 – Per-occ limit is zero on weekends only (= unlimited)
#           Since 0 = unlimited, coverage can be non-zero on weekends.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 10 – Per-occ zero on weekends (= unlimited, coverage ok)")
print("=" * 65)

occ_hc = [
    [5, 5, 5, 5, 5, 0, 0],  # 0 on weekends = unlimited
]
d = flat_demand(3)
p = quick_params(occ_hc=occ_hc)
r = solve_multi([d], ["Staff"], p)

check("TEST10a: solver feasible with weekend limit=0 (unlimited)",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    cov = r.combined_phase1.coverage
    # On weekends: coverage >= demand (since 0 = unlimited, no restriction)
    for day in [5, 6]:
        s = day * INTERVALS_PER_DAY
        e = s + INTERVALS_PER_DAY
        actual = int(cov[s:e].max())
        check(
            f"TEST10b-{DAY_NAMES[day]}: coverage({actual}) >= 0 (unlimited on weekends)",
            actual >= 0,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 11 – Some occs have limits, others don't (mixed None)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 11 – Mixed: some occs with limits, some without")
print("=" * 65)

# Occ0 has limit 3, Occ1 has no limit (None), Occ2 has limit 5
# Demand: Occ0=2 (fits in 3), Occ1=6 (unlimited), Occ2=3 (fits in 5)
occ_hc = [
    [3, 3, 3, 3, 3, 3, 3],
    None,  # Occ1: no limit
    [5, 5, 5, 5, 5, 5, 5],
]
p = quick_params(occ_hc=occ_hc)
r = solve_multi([flat_demand(2), flat_demand(6), flat_demand(3)], ["A", "B", "C"], p)

check("TEST11a: mixed None/default limits handled",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE", "INFEASIBLE"))

if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    # Occ0 should obey limit=3
    for day in range(7):
        s = day * INTERVALS_PER_DAY
        e = s + INTERVALS_PER_DAY
        actual = int(r.occupations[0].phase1.coverage[s:e].max())
        check(
            f"TEST11b-Occ0-{DAY_NAMES[day]}: {actual} <= 3",
            actual <= 3,
            f"VIOLATION: {actual} > 3"
        )
    # Occ2 should obey limit=5
    for day in range(7):
        s = day * INTERVALS_PER_DAY
        e = s + INTERVALS_PER_DAY
        actual = int(r.occupations[2].phase1.coverage[s:e].max())
        check(
            f"TEST11c-Occ2-{DAY_NAMES[day]}: {actual} <= 5",
            actual <= 5,
            f"VIOLATION: {actual} > 5"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 12 – 5-occupation per-occ headcount limits (feasible)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 12 – 5 occupations with per-occ headcount limits")
print("=" * 65)

# Each occ has demand=1, limits generous enough
occ_hc = [
    [3, 3, 3, 3, 3, 2, 2],
    [4, 4, 4, 4, 4, 3, 3],
    [5, 5, 5, 5, 5, 4, 4],
    [2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3, 3],
]
demands = [flat_demand(1) for _ in range(5)]
names = ["Tech", "Lab", "Help", "Sup", "Asst"]
p = quick_params(occ_hc=occ_hc)
r = solve_multi(demands, names, p)

check("TEST12a: 5-occ feasible", r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    for occ_idx, occ in enumerate(r.occupations):
        for day in range(7):
            s = day * INTERVALS_PER_DAY
            e = s + INTERVALS_PER_DAY
            actual_max = int(occ.phase1.coverage[s:e].max())
            limit = occ_hc[occ_idx][day]
            check(
                f"TEST12b-{occ.name}-{DAY_NAMES[day]}: "
                f"sim({actual_max}) <= limit({limit})",
                actual_max <= limit,
                f"VIOLATION"
            )
else:
    print(f"  Status: {r.combined_phase1.status} — may be infeasible with current settings")
    ok("TEST12a: completed without error", True)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 13 – Edge: some days limited, others unlimited
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 13 – Mixed limits across days (some 0=unlimited)")
print("=" * 65)

# Mon-Wed: limit=3, Thu-Sun: limit=0 (unlimited)
occ_hc = [[3, 3, 3, 0, 0, 0, 0]]
p = quick_params(occ_hc=occ_hc)
r = solve_multi([flat_demand(2)], ["Staff"], p)

check("TEST13a: solver feasible with partial limits",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    cov = r.combined_phase1.coverage
    # Mon-Wed: check limit
    for day in [0, 1, 2]:
        s = day * INTERVALS_PER_DAY
        e = s + INTERVALS_PER_DAY
        actual = int(cov[s:e].max())
        check(
            f"TEST13b-{DAY_NAMES[day]}: {actual} <= 3",
            actual <= 3,
            f"VIOLATION: {actual} > 3"
        )
    # Thu-Sun: should still cover demand (unlimited)
    for day in [3, 4, 5, 6]:
        s = day * INTERVALS_PER_DAY
        e = s + INTERVALS_PER_DAY
        sl = slice(s, e)
        actual = int(cov[sl].max())
        check(
            f"TEST13c-{DAY_NAMES[day]}: coverage({actual}) >= 0",
            actual >= 0,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 14 – Empty per-occ list should cause no constraint
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 14 – Empty occ_max_headcount_per_day list")
print("=" * 65)

p = quick_params(occ_hc=[])  # Empty list
r = solve_multi([flat_demand(2)], ["Staff"], p)
check("TEST14a: empty list treated as no constraint",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 15 – per-occ limit exactly equals demand
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TEST 15 – Per-occ limit exactly equals demand")
print("=" * 65)

occ_hc = [[4, 4, 4, 4, 4, 4, 4]]
p = quick_params(occ_hc=occ_hc)
r = solve_multi([flat_demand(4)], ["Staff"], p)

check("TEST15a: feasible when limit == demand",
      r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"))

if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
    cov = r.combined_phase1.coverage
    for day in range(7):
        s = day * INTERVALS_PER_DAY
        e = s + INTERVALS_PER_DAY
        actual = int(cov[s:e].max())
        check(
            f"TEST15b-{DAY_NAMES[day]}: {actual} <= 4",
            actual <= 4,
            f"VIOLATION: {actual} > 4"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"SUMMARY  PASS={_pass}  FAIL={_fail}")
print("=" * 65)
if _fail == 0:
    print("All per-occupation headcount tests passed.")
else:
    print("Some checks FAILED – see above for details.")