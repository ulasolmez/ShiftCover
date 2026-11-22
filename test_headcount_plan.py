"""Full-time headcount plan (SolverParams.plan_for_headcount).

  H1  classic plan → roster leaves people far below 40 h on a peaky week
  H2  headcount plan → same week, everyone in the hours window, demand covered,
      no more people than the classic roster
  H3  the plan's own aggregate rules hold (checked from the result, not the model)
  H4  1.5 off: groups at ~49 / ~41 h, two-week average ~45 h
  H5  non-circular week
"""
import numpy as np

from solver import SolverParams, solve_multi, TOTAL_INTERVALS, paid_intervals
from roster import (RosterParams, slots_from_phase1, solve_roster, verify_roster,
                    _rest_pair_bound)

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


# Peaky ramp-like week: day base, two banks, a late wave
d = np.zeros(TOTAL_INTERVALS, int)
for day in range(7):
    d[iv(day, "06:00"):iv(day, "22:00")] = 4
    d[iv(day, "07:00"):iv(day, "10:00")] += 6
    d[iv(day, "16:30"):iv(day, "19:30")] += 7 if day < 5 else 4
    d[iv(day, "22:00"):iv(day, "23:59")] = 3

BASE = dict(min_shift_hours=4.0, max_shift_hours=11.0, circular_week=True,
            prep_lead_min=30, exit_lag_min=30, handover_overlap_min=30,
            solver_time_limit_sec=60)


def plan(hc, off="2", **kw):
    spp = {"1": 6.0, "1.5": 5.5, "2": 5.0}[off]
    p = SolverParams(**{**BASE, **kw}, plan_for_headcount=hc, hc_shifts_per_person=spp,
                     hc_target_hours=45, hc_min_hours=40, hc_max_hours=50)
    return solve_multi([d], ["Ramp"], p)


def roster(res, off="2", **kw):
    sl = slots_from_phase1(res.occupations[0].phase1)
    r = solve_roster(sl, RosterParams(off_mode=off, time_limit_sec=60, **kw), "Ramp")
    return sl, r


def deficit(res):
    return int(np.maximum(res.combined_demand - res.combined_phase1.coverage, 0).sum())


print("\n=== H1: classic plan ===")
rc = plan(False)
slc, roc = roster(rc)
hc_ = [q.paid_hours for q in roc.people]
print(f"    classic: {roc.headcount} people, paid h {min(hc_):.1f}–{max(hc_):.1f}, "
      f"mean {np.mean(hc_):.1f}")
check("classic roster has people below 40 h", min(hc_) < 40)

print("\n=== H2: headcount plan, 2 off ===")
rh = plan(True)
p1 = rh.occupations[0].phase1
slh, roh = roster(rh)
hh = [q.paid_hours for q in roh.people]
print(f"    headcount: plan {p1.planned_people} people → roster {roh.headcount}, "
      f"paid h {min(hh):.1f}–{max(hh):.1f}, mean {np.mean(hh):.1f}")
check("plan solved", p1.status in ("OPTIMAL", "FEASIBLE"), p1.status)
check("raw demand fully covered", deficit(rh) == 0, deficit(rh))
check("roster rules ok", not verify_roster(roh, slh), verify_roster(roh, slh)[:2])
check("everyone 40–50 h", min(hh) >= 40 - 1e-6 and max(hh) <= 50 + 1e-6, (min(hh), max(hh)))
check("mean within 1.5 h of 45", abs(np.mean(hh) - 45) <= 1.5, np.mean(hh))
check("roster ≤ planned people + 2", roh.headcount <= p1.planned_people + 2,
      (roh.headcount, p1.planned_people))
check("not more people than the classic roster", roh.headcount <= roc.headcount,
      (roh.headcount, roc.headcount))

print("\n=== H3: plan's aggregate rules ===")
N = p1.planned_people
per_day = [sum(t.count for t in slh if t.day == x) for x in range(7)]
paid = sum(t.paid * t.count for t in slh) / 60
check("shift starts per day ≤ N", max(per_day) <= N, (per_day, N))
check("total shifts ≤ 5 N", sum(per_day) <= 5 * N, (sum(per_day), N))
check("40 N ≤ paid ≤ 50 N", 40 * N - 1e-6 <= paid <= 50 * N + 1e-6, (paid, N))
check("night-rest bound ≤ N", _rest_pair_bound(slh, RosterParams()) <= N,
      (_rest_pair_bound(slh, RosterParams()), N))
check("paid_intervals: 8.5 h shift → 7.5 h paid",
      paid_intervals(type("S", (), {"duration_intervals": 102})()) == 90)

print("\n=== H4: 1.5 off ===")
r15 = plan(True, off="1.5")
sl15, ro15 = roster(r15, off="1.5")
g1 = [q.paid_hours for q in ro15.people if q.kind == "FT 1-off"]
g2 = [q.paid_hours for q in ro15.people if q.kind == "FT 2-off"]
allh = g1 + g2
print(f"    1.5: plan {r15.occupations[0].phase1.planned_people} → roster {ro15.headcount} "
      f"(1-off {len(g1)}: {min(g1):.1f}–{max(g1):.1f} h, 2-off {len(g2)}: "
      f"{min(g2):.1f}–{max(g2):.1f} h)")
check("rules ok", not verify_roster(ro15, sl15), verify_roster(ro15, sl15)[:2])
check("split within ±1", abs(len(g1) - len(g2)) <= 1, (len(g1), len(g2)))
check("two-week average within 1.5 h of 45", abs(np.mean(allh) - 45) <= 1.5, np.mean(allh))
check("nobody below their week's minimum", not ro15.under_min(), len(ro15.under_min()))
check("demand covered", deficit(r15) == 0)

print("\n=== H5: non-circular week ===")
rn = plan(True, circular_week=False)
sln, ron = roster(rn, circular_week=False)
hn = [q.paid_hours for q in ron.people]
check("solves, covers demand, rules ok",
      deficit(rn) == 0 and ron.people and not verify_roster(ron, sln))
check("everyone ≥ 40 h", min(hn) >= 40 - 1e-6, min(hn))

print(f"\nTOTAL: {_pass + _fail} tests | PASSED: {_pass} | FAILED: {_fail}")
