"""Operational buffer tests: prep lead, exit lag, handover overlap.

Scenarios
  A  Single block 08:30–17:00            → shifts must start 08:00, end 17:30
  B  Long op 06:00–22:00, max shift 9 h  → relief must overlap outgoing 30 min
  C  Two-step ramp (5 → 9 → 3)           → ramp-up staff early, ramp-down staff late
  D  Realistic sample week               → headcount / FTE with vs without buffers
  E  Two occupations sample week         → same, multi-curve
"""

import numpy as np
from solver import (
    SolverParams, solve_multi, required_curve, buffer_violations,
    max_headcount, fte, INTERVALS_PER_DAY, INTERVALS_PER_HOUR, TOTAL_INTERVALS,
)
from sample_data import generate_sample_demand
import os
WEEK_LIMIT = int(os.environ.get("WEEK_LIMIT", "60"))

_pass = _fail = 0


def check(label, cond, detail=""):
    global _pass, _fail
    if cond:
        _pass += 1
        print(f"  [PASS] {label}")
    else:
        _fail += 1
        print(f"  [FAIL] {label}  {detail}")


def iv(day, hhmm):
    h, m = map(int, hhmm.split(":"))
    return day * INTERVALS_PER_DAY + (h * 60 + m) // 5


def block(day, start, end, n, d=None):
    d = np.zeros(TOTAL_INTERVALS, dtype=int) if d is None else d
    d[iv(day, start):iv(day, end)] = np.maximum(d[iv(day, start):iv(day, end)], n)
    return d


def used(p1):
    return [(s, c) for s, c in zip(p1.shifts, p1.counts) if c > 0]


BUF = dict(prep_lead_min=30, exit_lag_min=30, handover_overlap_min=30)
BASE = dict(min_shift_hours=3.0, max_shift_hours=12.0, transition_penalty=50,
            solver_time_limit_sec=30)


def run(demands, names, buffers, **kw):
    p = SolverParams(**{**BASE, **kw, **(BUF if buffers else {})})
    return solve_multi(demands, names, p)


# ── A: single operation block ───────────────────────────────────────────────
print("\n=== A: Operation 08:30–17:00, 5 workers (Monday) ===")
dA = block(0, "08:30", "17:00", 5)
rA0 = run([dA], ["Staff"], False)
rA1 = run([dA], ["Staff"], True)
for tag, r in (("no buffers", rA0), ("buffers", rA1)):
    print(f"  {tag}: " + ", ".join(f"{s.shift_code}×{c}" for s, c in used(r.combined_phase1)))
st0 = min(s.global_start for s, _ in used(rA0.combined_phase1))
st1 = min(s.global_start for s, _ in used(rA1.combined_phase1))
en1 = max(s.global_end for s, _ in used(rA1.combined_phase1))
check("baseline enters exactly at op start 08:30", st0 == iv(0, "08:30"))
check("buffered: earliest shift starts 08:00", st1 == iv(0, "08:00"))
check("buffered: last shift ends 17:30", en1 == iv(0, "17:30"))
cov = rA1.combined_phase1.coverage
check("buffered: all 5 on shift from 08:00", cov[iv(0, "08:00"):iv(0, "08:30")].min() >= 5)
check("buffered: all 5 still on shift 17:00–17:30", cov[iv(0, "17:00"):iv(0, "17:30")].min() >= 5)
check("buffered: buffer check clean", buffer_violations(rA1)["intervals"] == 0)

# ── B: long operation forcing a handover ────────────────────────────────────
print("\n=== B: Operation 06:00–22:00, 4 workers, max shift 9 h ===")
dB = block(0, "06:00", "22:00", 4)
rB0 = run([dB], ["Staff"], False, max_shift_hours=9.0)
rB1 = run([dB], ["Staff"], True, max_shift_hours=9.0)
for tag, r in (("no buffers", rB0), ("buffers", rB1)):
    print(f"  {tag}: " + ", ".join(f"{s.shift_code}×{c}" for s, c in used(r.combined_phase1)))


def handover_gaps(r, op_start, op_end, need):
    """For every exit inside the operation, minutes the leaver overlaps relief.

    Overlap = how long before the exit ``need`` other workers (who stay past
    the exit) were already on shift.
    """
    shifts = used(r.combined_phase1)
    res = []
    for s, c in shifts:
        e = s.global_end
        if not (op_start < e < op_end):
            continue
        stayers = sorted(
            [x.global_start for x, k in shifts if x.global_start <= e < x.global_end
             for _ in range(k)])
        if len(stayers) < need:
            res.append((s.shift_code, -1))
            continue
        latest_needed_arrival = stayers[need - 1]
        res.append((s.shift_code, (e - latest_needed_arrival) * 5))
    return res


g0 = handover_gaps(rB0, iv(0, "06:00"), iv(0, "22:00"), 4)
g1 = handover_gaps(rB1, iv(0, "06:00"), iv(0, "22:00"), 4)
print(f"  overlap at mid-op exits (min) – baseline: {g0}")
print(f"  overlap at mid-op exits (min) – buffered: {g1}")
check("baseline has a zero-overlap handover", any(o == 0 for _, o in g0))
check("buffered has at least one handover", len(g1) > 0)
check("buffered: every handover overlaps ≥ 30 min", all(o >= 30 for _, o in g1))
check("buffered: buffer check clean", buffer_violations(rB1)["intervals"] == 0)

# ── C: ramp up and ramp down inside one operation ───────────────────────────
print("\n=== C: 07:00–19:00 base 5, peak 9 at 11:00–14:00, drop to 3 after 16:00 ===")
dC = np.zeros(TOTAL_INTERVALS, dtype=int)
dC[iv(1, "07:00"):iv(1, "16:00")] = 5
dC[iv(1, "11:00"):iv(1, "14:00")] = 9
dC[iv(1, "16:00"):iv(1, "19:00")] = 3
rC1 = run([dC], ["Staff"], True)
print("  buffered: " + ", ".join(f"{s.shift_code}×{c}" for s, c in used(rC1.combined_phase1)))
cov = rC1.combined_phase1.coverage
check("9 on shift from 10:30 (prep before peak)", cov[iv(1, "10:30"):iv(1, "11:00")].min() >= 9)
check("9 still on shift until 14:30 (stay after peak)", cov[iv(1, "14:00"):iv(1, "14:30")].min() >= 9)
check("5 still on shift until 16:30", cov[iv(1, "16:00"):iv(1, "16:30")].min() >= 5)
check("3 on shift until 19:30", cov[iv(1, "19:00"):iv(1, "19:30")].min() >= 3)
check("buffered: buffer check clean", buffer_violations(rC1)["intervals"] == 0)

# ── required_curve unit checks ──────────────────────────────────────────────
print("\n=== required_curve() ===")
p = SolverParams(**BUF)
rq = required_curve(dA, p)
check("prep 30 = handover 30 → no extra look-ahead needed", rq[iv(0, "08:25")] == 0)
check("exit 30 → required until 17:30", rq[iv(0, "17:25")] == 5 and rq[iv(0, "17:30")] == 0)
p2 = SolverParams(prep_lead_min=60, handover_overlap_min=30)
rq2 = required_curve(dA, p2)
check("prep 60, handover 30 → 30 extra look-ahead", rq2[iv(0, "08:00")] == 5 and rq2[iv(0, "07:55")] == 0)
p3 = SolverParams(prep_lead_min=30, circular_week=True)
d_wrap = np.zeros(TOTAL_INTERVALS, dtype=int); d_wrap[0:12] = 2
check("circular: Monday 00:00 demand pulls prep into Sunday 23:30",
      required_curve(d_wrap, p3)[TOTAL_INTERVALS - 6] == 2)

# ── D/E: realistic weeks – headcount & FTE impact ───────────────────────────
rows = []


def compare(label, demands, names, **kw):
    r0 = run(demands, names, False, **kw)
    r1 = run(demands, names, True, **kw)
    out = []
    for tag, r in (("no buffers", r0), ("30/30/30", r1)):
        p1 = r.combined_phase1
        bv = buffer_violations(r, SolverParams(**{**BASE, **BUF}))
        deficit = int(np.maximum(r.combined_demand - p1.coverage, 0).sum())
        out.append(dict(scenario=label, mode=tag, status=p1.status,
                        headcount=max_headcount(p1),
                        worker_h=p1.total_worker_intervals / INTERVALS_PER_HOUR,
                        fte=fte(p1), peak=int(p1.coverage.max()),
                        shifts=sum(1 for c in p1.counts if c > 0),
                        deficit=deficit, buf_viol=bv["intervals"]))
    rows.extend(out)
    return r0, r1, out


print("\n=== D: sample week, 1 occupation ===")
dD = generate_sample_demand(peak_agents=25, base_agents=2, seed=42)
rD0, rD1, oD = compare("sample 1-occ", [dD], ["Staff"], solver_time_limit_sec=WEEK_LIMIT)
check("D baseline covers raw demand", oD[0]["deficit"] == 0)
check("D buffered covers raw demand", oD[1]["deficit"] == 0)
check("D buffered respects buffers", oD[1]["buf_viol"] == 0)
_bufp = SolverParams(**{**BASE, **BUF})
_bv0 = buffer_violations(rD0, _bufp)
print(f"  baseline scored against 30/30/30 rule: {_bv0['intervals']} intervals, "
      f"{_bv0['short_wh']:.0f} worker-hours unprepared/unhanded-over")
check("D baseline violates buffers (proves the fix matters)", _bv0["intervals"] > 0)
check("D buffered needs ≥ worker-hours", oD[1]["worker_h"] >= oD[0]["worker_h"])

print("\n=== E: sample week, 2 occupations ===")
dE = [generate_sample_demand(25, 3, seed=7), generate_sample_demand(12, 1, seed=8)]
rE0, rE1, oE = compare("sample 2-occ", dE, ["Technician", "Labourer"],
                       solver_time_limit_sec=WEEK_LIMIT)
check("E buffered covers raw demand", oE[1]["deficit"] == 0)
check("E buffered respects buffers", oE[1]["buf_viol"] == 0)

print("\n=== F: airport-style week (bank structure, 2 waves/day) ===")
dF = np.zeros(TOTAL_INTERVALS, dtype=int)
for day in range(7):
    dF = block(day, "05:30", "09:00", 8, dF)
    dF = block(day, "09:00", "12:30", 4, dF)
    dF = block(day, "12:30", "15:00", 10, dF)
    dF = block(day, "15:00", "21:30", 6, dF)
    dF = block(day, "21:30", "23:30", 9, dF)
rF0, rF1, oF = compare("airport waves", [dF], ["Ramp"], solver_time_limit_sec=WEEK_LIMIT)
check("F buffered covers raw demand", oF[1]["deficit"] == 0)
check("F buffered respects buffers", oF[1]["buf_viol"] == 0)

print("\n=== Headcount / FTE summary ===")
hdr = f"{'scenario':<15}{'mode':<12}{'status':<10}{'HC':>5}{'FTE':>8}{'wk-h':>9}{'peak':>6}{'shifts':>8}{'deficit':>9}{'buf_viol':>10}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['scenario']:<15}{r['mode']:<12}{r['status']:<10}{r['headcount']:>5}"
          f"{r['fte']:>8.1f}{r['worker_h']:>9.0f}{r['peak']:>6}{r['shifts']:>8}"
          f"{r['deficit']:>9}{r['buf_viol']:>10}")

print(f"\nTOTAL: {_pass + _fail} tests | PASSED: {_pass} | FAILED: {_fail}")
