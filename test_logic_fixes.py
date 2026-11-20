"""Regression tests for logic fixes found in the review.

  L1  max_unique_shifts counts distinct shift CODES, not per-day instances
  L2  shift-code minimum above peak demand is feasible
  L3  combined daily minimum above peak demand is feasible
  L4  exits are grouped by the day/time the worker actually leaves
  (upload parser tests live in test_groundstar_import.py)
"""

import numpy as np
from solver import (SolverParams, solve_multi, generate_candidate_shifts,
                    daily_entry_headcount, TOTAL_INTERVALS)

_pass = _fail = 0


def check(label, cond, detail=""):
    global _pass, _fail
    if cond:
        _pass += 1; print(f"  [PASS] {label}")
    else:
        _fail += 1; print(f"  [FAIL] {label}  {detail}")


def iv(day, hhmm):
    h, m = map(int, hhmm.split(":"))
    return day * 288 + (h * 60 + m) // 5


def codes(r):
    return {s.shift_code for s, c in zip(r.combined_phase1.shifts,
                                         r.combined_phase1.counts) if c}


def deficit(r):
    return int(np.maximum(r.combined_demand - r.combined_phase1.coverage, 0).sum())


OK = ("OPTIMAL", "FEASIBLE")
daily = np.zeros(TOTAL_INTERVALS, int)
for day in range(7):
    daily[iv(day, "08:00"):iv(day, "16:00")] = 3

print("\n=== L1: max unique shift codes ===")
r = solve_multi([daily], ["S"], SolverParams(
    min_shift_hours=8, max_shift_hours=8, max_unique_shifts=1, solver_time_limit_sec=20))
check("one code reused on 7 days fits max_unique=1", r.combined_phase1.status in OK,
      r.combined_phase1.status)
check("…and uses exactly 1 code", codes(r) == {"0800-1600"}, codes(r))
two = np.zeros(TOTAL_INTERVALS, int)
for day in range(7):
    two[iv(day, "06:00"):iv(day, "22:00")] = 2
r = solve_multi([two], ["S"], SolverParams(
    min_shift_hours=4, max_shift_hours=12, max_unique_shifts=2, solver_time_limit_sec=30))
check("16 h op with max 2 codes solves", r.combined_phase1.status in OK, r.combined_phase1.status)
check("…uses ≤ 2 codes", len(codes(r)) <= 2, codes(r))
check("…covers demand", deficit(r) == 0)

print("\n=== L2: shift-code min above peak ===")
r = solve_multi([daily], ["S"], SolverParams(
    min_shift_hours=8, max_shift_hours=8, solver_time_limit_sec=20,
    occ_headcount_per_shift_code=[{"0800-1600": {"min": [5] * 7}}]))
check("feasible", r.combined_phase1.status in OK, r.combined_phase1.status)
check("5 workers on 0800-1600 every day",
      all(v >= 5 for v in daily_entry_headcount(r.combined_phase1)))

print("\n=== L3: combined daily min above peak, single allowed shift ===")
r = solve_multi([daily], ["S"], SolverParams(
    min_shift_hours=8, max_shift_hours=8, solver_time_limit_sec=20,
    allowed_slot_minutes=[480, 960], min_headcount_per_day=[5] * 7))
check("feasible", r.combined_phase1.status in OK, r.combined_phase1.status)
check("≥5 entries every day", all(v >= 5 for v in daily_entry_headcount(r.combined_phase1)))

print("\n=== L4: exits grouped by actual exit day ===")
# (a) Monday: day op ends 16:00, night op Mon 22:00 → Tue 06:00.
#     Monday has ONE real exit time (16:00); the night crew leaves Tuesday.
#     Old code booked the night exit on Monday → falsely infeasible.
na = np.zeros(TOTAL_INTERVALS, int)
na[iv(0, "08:00"):iv(0, "16:00")] = 2
na[iv(0, "22:00"):iv(1, "06:00")] = 2
r = solve_multi([na], ["S"], SolverParams(
    min_shift_hours=8, max_shift_hours=8, solver_time_limit_sec=20,
    max_exits_per_day=[1, 0, 0, 0, 0, 0, 0]))
check("Mon limit 1: day exit Mon + night exit Tue is feasible",
      r.combined_phase1.status in OK and deficit(r) == 0, r.combined_phase1.status)
# (b) Tuesday: night crew leaves Tue morning, day crew Tue 14:00 → 2 exit
#     times on Tuesday. Old code hid the night exit on Monday and accepted
#     limit 1; the correct answer is infeasible.
nb = np.zeros(TOTAL_INTERVALS, int)
nb[iv(0, "22:00"):iv(1, "06:00")] = 2
nb[iv(1, "08:00"):iv(1, "14:00")] = 2
r = solve_multi([nb], ["S"], SolverParams(
    min_shift_hours=6, max_shift_hours=10, solver_time_limit_sec=20,
    max_exits_per_day=[0, 1, 0, 0, 0, 0, 0]))
check("Tue limit 1 with 2 real Tuesday exit times is infeasible",
      r.combined_phase1.status == "INFEASIBLE", r.combined_phase1.status)
r = solve_multi([nb], ["S"], SolverParams(
    min_shift_hours=6, max_shift_hours=10, solver_time_limit_sec=20,
    max_exits_per_day=[0, 2, 0, 0, 0, 0, 0]))
check("…and feasible with Tue limit 2", r.combined_phase1.status in OK and deficit(r) == 0)
sh = generate_candidate_shifts(SolverParams(min_shift_hours=8, max_shift_hours=8))
s = next(x for x in sh if x.day == 0 and x.start_interval == iv(0, "16:00"))
check("16:00–24:00 is a Monday exit (not Tuesday 00:00)", (s.global_end - 1) // 288 == 0)

# L5 (upload parser) moved to test_groundstar_import.py

print(f"\nTOTAL: {_pass + _fail} tests | PASSED: {_pass} | FAILED: {_fail}")
