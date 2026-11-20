"""Roster step tests (roster.py).

  R1  each off mode reaches its lower bound on a simple week
  R2  no "Sunday-only" tail: flat demand → everyone gets a full week
  R3  12 h rest: exactly 12 h allowed, 11.5 h not
  R4  1-off needs one rest ≥ 24 h (23.5 h across the off is not enough);
      2-off people don't
  R5  1.5 off splits people half/half with scaled hour targets
  R6  max hours is a hard limit
  R7  part-time only when allowed, within its limits
  R8  non-circular week
  R9  plan → roster pipeline on a generated week, rules re-checked
"""
import numpy as np

from roster import (RosterParams, SlotType, Person, RosterResult, solve_roster,
                    verify_roster, lower_bounds, slots_from_phase1, WEEK_MIN)
from solver import SolverParams, solve_multi
from sample_data import generate_sample_demand

_pass = _fail = 0


def check(label, cond, detail=""):
    global _pass, _fail
    if cond:
        _pass += 1; print(f"  [PASS] {label}")
    else:
        _fail += 1; print(f"  [FAIL] {label}  {detail}")


def slot(day, code, count):
    a, b = code.split("-")
    s = int(a[:2]) * 60 + int(a[2:]); e = int(b[:2]) * 60 + int(b[2:])
    if e <= s:
        e += 1440
    length = (e - s) / 60
    brk = 60 if length > 7.5 else 30 if length > 4 else 0
    return SlotType(day, code, day * 1440 + s, day * 1440 + e, int(length * 60 - brk), count)


def run(slots, **kw):
    rp = RosterParams(**{"time_limit_sec": 20, **kw})
    r = solve_roster(slots, rp, "T")
    return r, verify_roster(r, slots)


print("\n=== R1: lower bound reached per off mode ===")
week = []
for d in range(7):
    a, b = (10, 5) if d < 5 else (6, 3)
    week += [slot(d, "0830-1700", a), slot(d, "1630-0100", b)]
for mode, expect in (("1", 16), ("1.5", 17), ("2", 19), ("auto", 16)):
    r, errs = run(week, off_mode=mode)
    check(f"off {mode}: {r.headcount} people = lower bound {expect}, rules ok",
          r.headcount == expect == r.lower_bound and not errs, (r.headcount, r.lower_bound, errs[:2]))

print("\n=== R2: flat demand, no Sunday-only tail ===")
flat = [slot(d, "0830-1700", 10) for d in range(7)]           # 70 shifts
r, errs = run(flat, off_mode="2")
hrs = sorted(q.paid_hours for q in r.people)
check("2 off: 14 people, all exactly 5 shifts / 37.5 h",
      r.headcount == 14 and set(hrs) == {37.5} and not errs, (r.headcount, hrs[:3], errs[:2]))
r, errs = run(flat, off_mode="1")
hrs = sorted(q.paid_hours for q in r.people)
check("1 off: 12 people, nobody under 5 shifts (37.5 h)",
      r.headcount == 12 and hrs[0] >= 37.5 and not errs, (r.headcount, hrs[:3]))

print("\n=== R3: 12 h rest is inclusive ===")
# one person a day; Mon-Fri 0800-2000 then next day 0800 → exactly 12 h rest
ok12 = [slot(d, "0800-2000", 1) for d in range(5)]
r, errs = run(ok12, off_mode="2", circular_week=False, max_hours=60)
check("12 h gap allowed → 1 person covers 5 days", r.headcount == 1 and not errs, (r.headcount, errs))
bad = [slot(d, "0830-2030", 1) if d % 2 == 0 else slot(d, "0800-2000", 1) for d in range(5)]
r, errs = run(bad, off_mode="2", circular_week=False, max_hours=60)
check("11.5 h gap forbidden → 2 people", r.headcount == 2 and not errs, (r.headcount, errs))

print("\n=== R4: 1-off 24 h rest ===")


def person(kind, codes):
    days = [slot(d, c, 1) if c else None for d, c in enumerate(codes)]
    return Person("p", kind, days, sum(t.paid for t in days if t))


def check_one(kind, codes, circular=True):
    p = person(kind, codes)
    sl = [t for t in p.shifts if t]
    rp = RosterParams(off_mode="auto", circular_week=circular, max_hours=70)
    res = RosterResult("T", [p], "", {}, rp, 0, {}, {"FT 1-off": (0, 70), "FT 2-off": (0, 70), "PT": (0, 70)})
    return verify_roster(res, sl)


six_235 = ["1630-0100", "1630-0100", None, "0030-0900", "0030-0900", "0030-0900", "0030-0900"]
errs = check_one("FT 1-off", six_235, circular=False)
check("1-off with only 23.5 h around its off → rejected",
      any("1-off" in e for e in errs), errs)
six_ok = ["0830-1700", "0830-1700", None, "0830-1700", "0830-1700", "0830-1700", "0830-1700"]
check("1-off with 39.5 h across the off → ok", not check_one("FT 1-off", six_ok))
five = ["1630-0100", None, "0030-0900", "1630-0100", None, "0030-0900", "0830-1700"]
check("2-off with 23.5 h gaps → ok (no 24 h rule)", not check_one("FT 2-off", five, False))
# solver honours it: night block that would only give 23.5 h
nights = []
for d in range(7):
    nights.append(slot(d, "1630-0100", 2))
    nights.append(slot(d, "0030-0900", 2))
r, errs = run(nights, off_mode="1")
check("1-off solve on night waves respects 24 h", r.people and not errs, errs[:2])

print("\n=== R5: 1.5 off ===")
r, errs = run(week, off_mode="1.5")
a, b = r.count("FT 1-off"), r.count("FT 2-off")
check("split within ±1", abs(a - b) <= 1, (a, b))
check("targets scaled 45·6/5.5 and 45·5/5.5",
      abs(r.target_by_kind["FT 1-off"] - 49.09) < 0.01 and abs(r.target_by_kind["FT 2-off"] - 40.91) < 0.01)
check("rules ok", not errs, errs[:2])

print("\n=== R6: max hours hard ===")
r, errs = run(flat, off_mode="1", max_hours=40, min_hours=30)
check("max 40 h → nobody above, more people than 12",
      max(q.paid_hours for q in r.people) <= 40 and r.headcount > 12 and not errs,
      (r.headcount, max(q.paid_hours for q in r.people)))

print("\n=== R7: part-time ===")
r, errs = run(week, off_mode="2")
check("not allowed → no part-timers", r.count("PT") == 0)
r2, errs2 = run(week, off_mode="2", allow_part_time=True, pt_max_shifts=3,
                pt_min_hours=15, pt_max_hours=25)
check("allowed → never more people than without", r2.headcount <= r.headcount,
      (r2.headcount, r.headcount))
pts = [q for q in r2.people if q.kind == "PT"]
check("part-timers within shift / hours limits",
      all(q.n_shifts <= 3 and q.paid_hours <= 25 for q in pts) and not errs2, errs2[:2])

print("\n=== R8: non-circular week ===")
r, errs = run(week, off_mode="2", circular_week=False)
check("solves, rules ok", r.people and not errs, errs[:2])

print("\n=== R9: plan → roster pipeline ===")
d = generate_sample_demand(peak_agents=18, base_agents=2, seed=3)
res = solve_multi([d], ["Staff"], SolverParams(
    min_shift_hours=6, max_shift_hours=11, solver_time_limit_sec=30, circular_week=True,
    prep_lead_min=30, exit_lag_min=30, handover_overlap_min=30))
sl = slots_from_phase1(res.occupations[0].phase1)
for mode in ("2", "1.5"):
    r = solve_roster(sl, RosterParams(off_mode=mode, time_limit_sec=40), "Staff")
    errs = verify_roster(r, sl)
    print(f"    off {mode}: {r.headcount} people (bound {r.lower_bound}), "
          f"paid h {min(q.paid_hours for q in r.people):.1f}–{max(q.paid_hours for q in r.people):.1f}")
    check(f"off {mode}: every planned shift staffed, rules ok", not errs, errs[:2])
    check(f"off {mode}: within 10 % of the lower bound",
          r.headcount <= max(r.lower_bound + 2, 1.10 * r.lower_bound), (r.headcount, r.lower_bound))

print(f"\nTOTAL: {_pass + _fail} tests | PASSED: {_pass} | FAILED: {_fail}")
