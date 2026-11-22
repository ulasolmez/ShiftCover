"""Roster step: turn the shift plan into people.

Phase 1 decides *which shifts* run each day ("Tue 0600-1430 × 8").  This step
assigns every one of those shift slots to a person, using as few people as
possible and then making their paid hours as even as possible.

Rules (per occupation, people are interchangeable within it)
------------------------------------------------------------
* at most one shift starting per person per calendar day
* rest between consecutive shifts ≥ ``min_rest_hours`` (12 h; exactly 12 ok)
* full-time 1-off person: ≤ 6 shifts and at least one rest ≥ ``off_rest_hours``
  (24 h) somewhere in the week — calendar days don't matter
* full-time 2-off person: ≤ 5 shifts (offs needn't be consecutive)
* off modes:  "1"   everyone 1-off
              "2"   everyone 2-off
              "1.5" half 1-off / half 2-off this week (± 1); the groups swap
                    next week, so hours targets are 45·6/5.5 and 45·5/5.5
              "auto" solver picks 1-off or 2-off per person
* part-time (optional): ≤ ``pt_max_shifts`` shifts, own hours window
* paid hours = shift length − break (GroundStar rule, see solver.break_hours)
* weekly paid hours: max is hard, min is soft (penalised), target soft

Lower bounds: paid hours ÷ max hours · shifts ÷ shifts per person · busiest
day · consecutive-day pairs under the rest rule (see _rest_pair_bound).

Objective (two CP-SAT passes)
-----------------------------
1. fewest people (full-time preferred over part-time on ties)
2. with that headcount fixed: smallest worst-case gap to the hours target,
   then smallest total gap, then fewest hours below the minimum
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from solver import effective_hours, INTERVALS_PER_HOUR

DAY_MIN = 1440
WEEK_MIN = 7 * DAY_MIN
DAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
OFF_MODES = ("1", "1.5", "2", "auto")


@dataclass
class RosterParams:
    off_mode: str = "2"
    target_hours: float = 45.0
    min_hours: float = 40.0
    max_hours: float = 50.0
    min_rest_hours: float = 12.0
    off_rest_hours: float = 24.0
    allow_part_time: bool = False
    pt_min_hours: float = 20.0
    pt_max_hours: float = 30.0
    pt_max_shifts: int = 4
    circular_week: bool = True
    time_limit_sec: float = 60.0
    num_workers: int = 0          # 0 = all cores (4–16)


@dataclass
class SlotType:
    """One shift code on one day, with how many people it needs."""
    day: int
    code: str
    start: int          # absolute minutes from Mon 00:00
    end: int
    paid: int           # paid minutes
    count: int


@dataclass
class Person:
    pid: str
    kind: str                           # "FT 1-off" | "FT 2-off" | "PT"
    shifts: List[Optional[SlotType]]    # index = day
    paid_min: int

    @property
    def paid_hours(self) -> float:
        return self.paid_min / 60

    @property
    def n_shifts(self) -> int:
        return sum(1 for s in self.shifts if s is not None)


@dataclass
class RosterResult:
    occupation: str
    people: List[Person]
    status: str
    lower_bounds: Dict[str, int]
    params: RosterParams
    elapsed_sec: float
    target_by_kind: Dict[str, float] = field(default_factory=dict)
    window_by_kind: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    message: str = ""

    @property
    def headcount(self) -> int:
        return len(self.people)

    @property
    def lower_bound(self) -> int:
        return max(self.lower_bounds.values()) if self.lower_bounds else 0

    def count(self, kind: str) -> int:
        return sum(1 for p in self.people if p.kind == kind)

    def under_min(self) -> List[Person]:
        return [p for p in self.people
                if p.paid_hours + 1e-6 < self.window_by_kind.get(p.kind, (0, 0))[0]]


# ── Inputs ───────────────────────────────────────────────────────────────────
def slots_from_phase1(p1) -> List[SlotType]:
    """Aggregate a PhaseOneResult into (day, code) slot types."""
    agg: Dict[Tuple[int, str], SlotType] = {}
    for s, c in zip(p1.shifts, p1.counts):
        if c <= 0:
            continue
        start = s.global_start * 5
        end = s.global_end * 5
        paid = int(round(effective_hours(s.duration_intervals / INTERVALS_PER_HOUR) * 60))
        key = (s.day, s.shift_code)
        if key in agg:
            agg[key].count += int(c)
        else:
            agg[key] = SlotType(s.day, s.shift_code, start, end, paid, int(c))
    return sorted(agg.values(), key=lambda t: (t.day, t.start, t.end))


def _windows(rp: RosterParams):
    """Hours target and [min, max] window per person kind (in hours)."""
    if rp.off_mode == "1.5":
        f1, f2 = 6 / 5.5, 5 / 5.5           # two-week average stays on target
    else:
        f1 = f2 = 1.0
    w = {
        "FT 1-off": (rp.target_hours * f1, rp.min_hours * f1, rp.max_hours * f1),
        "FT 2-off": (rp.target_hours * f2, rp.min_hours * f2, rp.max_hours * f2),
        "PT": ((rp.pt_min_hours + rp.pt_max_hours) / 2, rp.pt_min_hours, rp.pt_max_hours),
    }
    return w


def lower_bounds(slots: List[SlotType], rp: RosterParams) -> Dict[str, int]:
    total_paid = sum(t.paid * t.count for t in slots) / 60
    total_shifts = sum(t.count for t in slots)
    per_day = [sum(t.count for t in slots if t.day == d) for d in range(7)]
    max_days = {"1": 6, "auto": 6, "1.5": 5.5, "2": 5}[rp.off_mode]
    w = _windows(rp)
    max_h = max(w["FT 1-off"][2] if rp.off_mode != "2" else 0,
                w["FT 2-off"][2] if rp.off_mode != "1" else 0)
    if rp.off_mode == "1.5":
        max_h = rp.max_hours                 # average of the two groups
    return {
        "hours": math.ceil(total_paid / max_h - 1e-9) if max_h > 0 else 0,
        "days": math.ceil(total_shifts / max_days - 1e-9),
        "busiest day": max(per_day) if per_day else 0,
        "rest (day pairs)": _rest_pair_bound(slots, rp),
    }


def _rest_pair_bound(slots: List[SlotType], rp: RosterParams) -> int:
    """People needed for two consecutive days under the daily-rest rule.

    A person works at most one shift per day, so days d and d+1 need
    n_d + n_{d+1} − (max number of day-d shifts that can be followed by a
    day-(d+1) shift with ≥ min rest).  The max is a bipartite matching on
    the shift counts (a small max-flow).  Night waves make this the binding
    bound: people finishing at 04:00 can't take tomorrow's 06:00 shifts.
    """
    rest = rp.min_rest_hours * 60
    by_day = [[t for t in slots if t.day == d] for d in range(7)]
    best = 0
    for d in range(7):
        n, off = _nxt(d, rp.circular_week)
        if n is None:
            continue
        A, B = by_day[d], by_day[n]
        na, nb = sum(t.count for t in A), sum(t.count for t in B)
        if not na or not nb:
            best = max(best, na, nb)
            continue
        # max flow: source → A (cap count) → B (compatible) → sink (cap count)
        # A and B are sorted by time, so a greedy works: compatibility is
        # "B.start ≥ A.end + rest" — threshold-type, so match the earliest
        # finishing A shifts to the earliest compatible B shifts.
        a_list = sorted(((t.end, t.count) for t in A))
        b_list = sorted(((t.start + off, t.count) for t in B))
        matched, j, b_left = 0, 0, [c for _, c in b_list]
        for end, cnt in a_list:
            while cnt and j < len(b_list):
                if b_list[j][0] - end >= rest and b_left[j] > 0:
                    take = min(cnt, b_left[j])
                    cnt -= take; b_left[j] -= take; matched += take
                    if b_left[j] == 0:
                        j += 1
                elif b_list[j][0] - end < rest:
                    j += 1
                else:
                    j += 1
            # B shifts skipped for this A can't serve later (later-ending) A either
        best = max(best, na + nb - matched)
    return best


def precheck(slots: List[SlotType], rp: RosterParams) -> List[str]:
    """Warnings when the hours window can't be reached with these shifts."""
    msgs = []
    if not slots:
        return msgs
    paid = sorted({t.paid for t in slots})
    longest = paid[-1] / 60
    w = _windows(rp)
    kinds = {"1": ["FT 1-off"], "2": ["FT 2-off"], "1.5": ["FT 1-off", "FT 2-off"],
             "auto": ["FT 1-off", "FT 2-off"]}[rp.off_mode]
    for k in kinds:
        n = 6 if k == "FT 1-off" else 5
        if n * longest + 1e-6 < w[k][1]:
            msgs.append(f"{k}: {n} shifts × longest paid shift {longest:.1f} h = "
                        f"{n * longest:.1f} h < min {w[k][1]:.1f} h — nobody in this "
                        f"group can reach the minimum.")
    return msgs


# ── Model ────────────────────────────────────────────────────────────────────
def _nxt(d, circular):
    if d < 6:
        return d + 1, 0
    return (0, WEEK_MIN) if circular else (None, 0)


def _prv(d, circular):
    if d > 0:
        return d - 1, 0
    return (6, -WEEK_MIN) if circular else (None, 0)


def _pair_tables(slots, circular):
    """Rest gaps (minutes) between slot types on day d → d+1 and d-1 → d+1."""
    by_day = [[k for k, t in enumerate(slots) if t.day == d] for d in range(7)]
    nxt, skip = [], []
    for d in range(7):
        n, off = _nxt(d, circular)
        nxt.append([] if n is None else
                   [(ka, kb, slots[kb].start + off - slots[ka].end)
                    for ka in by_day[d] for kb in by_day[n]])
        pv, poff = _prv(d, circular)
        nx, noff = _nxt(d, circular)
        skip.append([] if pv is None or nx is None else
                    [(ka, kb, slots[kb].start + noff - (slots[ka].end + poff))
                     for ka in by_day[pv] for kb in by_day[nx]])
    return by_day, nxt, skip


def _build(slots, rp: RosterParams, P: int, fix_people=None, fix_pt=None,
           sort_by_hours=True):
    m = cp_model.CpModel()
    w = _windows(rp)
    rest = int(round(rp.min_rest_hours * 60))
    rest_off = int(round(rp.off_rest_hours * 60))
    by_day, pairs_next, pairs_skip = _pair_tables(slots, rp.circular_week)

    y = {}
    for p in range(P):
        for k in range(len(slots)):
            y[p, k] = m.new_bool_var(f"y{p}_{k}")
    for k, t in enumerate(slots):
        m.add(sum(y[p, k] for p in range(P)) == t.count)

    used, t1, t2, tp, H, dev, under = [], [], [], [], [], [], []
    max_paid_week = 7 * max(t.paid for t in slots)
    maxdev = m.new_int_var(0, max_paid_week, "maxdev")
    pt_ok = rp.allow_part_time

    for p in range(P):
        u = m.new_bool_var(f"used{p}")
        a = m.new_bool_var(f"t1_{p}")
        b = m.new_bool_var(f"t2_{p}")
        c = m.new_bool_var(f"tp_{p}")
        m.add(a + b + c == u)
        if rp.off_mode == "1":
            m.add(b == 0)
        if rp.off_mode == "2":
            m.add(a == 0)
        if not pt_ok:
            m.add(c == 0)
        used.append(u); t1.append(a); t2.append(b); tp.append(c)

        work = []
        for d in range(7):
            ks = by_day[d]
            wd = m.new_bool_var(f"w{p}_{d}")
            if ks:
                m.add(sum(y[p, k] for k in ks) == wd)
            else:
                m.add(wd == 0)
            work.append(wd)
            m.add_implication(wd, u)

        n_sh = sum(work)
        m.add(n_sh >= u)
        m.add(n_sh <= 6 * a + 5 * b + rp.pt_max_shifts * c)

        long_rests = []
        for d in range(7):
            n, off = _nxt(d, rp.circular_week)
            if n is not None:
                lr = m.new_bool_var(f"long{p}_{d}") if rest_off > rest else None
                if lr is not None:
                    m.add_implication(lr, work[d]); m.add_implication(lr, work[n])
                    long_rests.append(lr)
                for ka, kb, gap in pairs_next[d]:
                    if gap < rest:                       # daily rest
                        m.add_bool_or([y[p, ka].Not(), y[p, kb].Not()])
                    elif lr is not None and gap < rest_off:
                        m.add_bool_or([y[p, ka].Not(), y[p, kb].Not(), lr.Not()])
            # ≥ 24 h rest across an empty day d
            o = m.new_bool_var(f"off{p}_{d}")
            m.add_implication(o, work[d].Not())
            for ka, kb, gap in pairs_skip[d]:
                if gap < rest_off:
                    m.add_bool_or([y[p, ka].Not(), y[p, kb].Not(), o.Not()])
            if not rp.circular_week and d == 6:
                # off on the last day of a standalone week: the rest runs from
                # Saturday's shift to the end of the week
                for ka in by_day[5]:
                    if WEEK_MIN - slots[ka].end < rest_off:
                        m.add_bool_or([y[p, ka].Not(), o.Not()])
            long_rests.append(o)
        # 1-off people need at least one ≥24 h rest
        m.add(sum(long_rests) >= 1).only_enforce_if(a)

        h = m.new_int_var(0, max_paid_week, f"H{p}")
        m.add(h == sum(slots[k].paid * y[p, k] for k in range(len(slots))))
        H.append(h)
        tgt = {k: int(round(v[0] * 60)) for k, v in w.items()}
        lo = {k: int(round(v[1] * 60)) for k, v in w.items()}
        hi = {k: int(round(v[2] * 60)) for k, v in w.items()}
        m.add(h <= hi["FT 1-off"] * a + hi["FT 2-off"] * b + hi["PT"] * c)
        un = m.new_int_var(0, max_paid_week, f"under{p}")
        m.add(un >= lo["FT 1-off"] * a + lo["FT 2-off"] * b + lo["PT"] * c - h)
        under.append(un)
        T = tgt["FT 1-off"] * a + tgt["FT 2-off"] * b + tgt["PT"] * c
        dv = m.new_int_var(0, max_paid_week, f"dev{p}")
        m.add(dv >= h - T)
        m.add(dv >= T - h)
        m.add(maxdev >= dv)
        dev.append(dv)

    # symmetry breaking: people sorted by paid hours (always possible because
    # people are interchangeable) — unused people (H = 0) end up last
    for p in range(P - 1):
        if sort_by_hours:
            m.add(H[p] >= H[p + 1])
        m.add(used[p] >= used[p + 1])

    if rp.off_mode == "1.5":
        m.add(sum(t1) - sum(t2) <= 1)
        m.add(sum(t2) - sum(t1) <= 1)

    if fix_people is not None:
        m.add(sum(used) == fix_people)
        m.add(sum(tp) == fix_pt)

    vars_ = dict(y=y, used=used, t1=t1, t2=t2, tp=tp, H=H, dev=dev,
                 under=under, maxdev=maxdev)
    return m, vars_


def _greedy(slots: List[SlotType], rp: RosterParams):
    """Quick valid roster used as a warm start (and fallback).

    Days are filled Monday → Sunday; each shift goes to the eligible person
    who has worked least so far (then: longest rest), which spreads days off
    over the week instead of exhausting everyone before Sunday.  A new person
    is opened only when nobody can legally take the shift.

    Returns list of (kind, [slot index, ...]) or None if a rule would be broken.
    """
    rest = rp.min_rest_hours * 60
    rest_off = rp.off_rest_hours * 60
    w = _windows(rp)
    people = []          # dict(kind, ks, days:set, paid, last_end per day)

    def new_kind():
        if rp.off_mode == "2":
            return "FT 2-off"
        if rp.off_mode == "1.5":
            n1 = sum(1 for q in people if q["kind"] == "FT 1-off")
            return "FT 1-off" if n1 <= len(people) - n1 else "FT 2-off"
        return "FT 1-off"

    def limit(q):
        return 6 if q["kind"] == "FT 1-off" else 5

    def ok(q, t):
        if t.day in q["days"] or len(q["ks"]) >= limit(q):
            return False
        if q["paid"] + t.paid > w[q["kind"]][2] * 60 + 1e-6:
            return False
        for k in q["ks"]:
            o = slots[k]
            a, b = (o, t) if o.start < t.start else (t, o)
            if b.start - a.end < rest:
                return False
            if rp.circular_week and (a.start + WEEK_MIN - b.end) < rest:
                return False
        if q["kind"] == "FT 1-off" and len(q["ks"]) + 1 == 6:
            sh = sorted([slots[k] for k in q["ks"]] + [t], key=lambda x: x.start)
            gaps = [b.start - a.end for a, b in zip(sh, sh[1:])]
            if rp.circular_week:
                gaps.append(sh[0].start + WEEK_MIN - sh[-1].end)
            else:
                gaps += [sh[0].start, WEEK_MIN - sh[-1].end]
            if max(gaps) < rest_off:
                return False
        return True

    order = sorted(range(len(slots)), key=lambda k: (slots[k].day, slots[k].start))
    for k in order:
        t = slots[k]
        for _ in range(t.count):
            cand = [q for q in people if ok(q, t)]
            if cand:
                q = min(cand, key=lambda q: (len(q["ks"]), q["paid"]))
            else:
                q = dict(kind=new_kind(), ks=[], days=set(), paid=0)
                people.append(q)
                if not ok(q, t):
                    return None
            q["ks"].append(k); q["days"].add(t.day); q["paid"] += t.paid
    return [(q["kind"], q["ks"]) for q in people]


def _match_slots(insts, cands, compatible):
    """Kuhn matching: every slot instance gets a person.

    ``cands`` are people in priority order; each instance tries them in that
    order, so higher-priority people are used first.  Returns {inst_pos: p}
    or None if some instance can't be staffed.
    """
    owner = {}                 # person -> inst position
    adj = [[p for p in cands if compatible(p, k)] for k in insts]

    def aug(j, seen):
        for p in adj[j]:
            if p in seen:
                continue
            seen.add(p)
            if p not in owner or aug(owner[p], seen):
                owner[p] = j
                return True
        return False

    # hardest instances first (fewest candidates)
    for j in sorted(range(len(insts)), key=lambda j: len(adj[j])):
        if not aug(j, set()):
            return None
    return {j: p for p, j in owner.items()}


def _greedy_daywise(slots: List[SlotType], rp: RosterParams, P: int, variant: int = 0):
    """Day-by-day roster with planned capacity (warm start for CP-SAT).

    Each day the day's shift slots are matched to people in one go:
    a person can take a slot if the rest since their last shift is enough
    (12 h; 24 h when a 1-off person comes back from their off day without
    having had a long rest yet), the week's Sunday → Monday wrap is respected,
    and their shift and hours limits allow it.  People with the most
    remaining shift capacity are preferred — the constructive rule that keeps
    enough capacity for the rest of the week, so nobody "runs out of days"
    before Sunday.

    Returns [(kind, [k, ...]), ...] or None.
    """
    rest = rp.min_rest_hours * 60
    rest_off = rp.off_rest_hours * 60
    w = _windows(rp)
    if rp.off_mode == "2":
        kinds = ["FT 2-off"] * P
    elif rp.off_mode == "1.5":
        kinds = ["FT 1-off" if i % 2 == 0 else "FT 2-off" for i in range(P)]
    else:
        kinds = ["FT 1-off"] * P
    left = [6 if k == "FT 1-off" else 5 for k in kinds]
    max_paid = [w[k][2] * 60 for k in kinds]
    inst = [[] for _ in range(7)]
    for k, t in enumerate(slots):
        inst[t.day].extend([k] * t.count)
    if max(len(x) for x in inst) > P or sum(len(x) for x in inst) > sum(left):
        return None

    got = [[None] * 7 for _ in range(P)]
    paid = [0] * P
    last_end = [None] * P          # absolute minutes
    last_day = [None] * P
    had_long = [False] * P
    first_slot = [None] * P        # Monday-side shift for the circular wrap

    for d in range(7):
        future_need = sum(len(inst[x]) for x in range(d, 7))

        def compatible(p, k, d=d):
            t = slots[k]
            if left[p] <= 0 or paid[p] + t.paid > max_paid[p] + 1e-6:
                return False
            if last_end[p] is not None:
                gap = t.start - last_end[p]
                need_gap = rest
                if (kinds[p] == "FT 1-off" and not had_long[p]
                        and last_day[p] is not None and d - last_day[p] >= 2):
                    need_gap = rest_off
                if gap < need_gap:
                    return False
            # circular wrap: rest from this shift to the person's Monday shift
            if rp.circular_week and first_slot[p] is not None and d >= 5:
                f = slots[first_slot[p]]
                days_between = 7 - d + f.day
                wrap_gap = f.start + WEEK_MIN - t.end
                need = rest
                if (kinds[p] == "FT 1-off" and not had_long[p] and days_between >= 2
                        and left[p] - 1 <= 0):
                    need = rest_off
                if d == 6 or days_between == 2:
                    if wrap_gap < need:
                        return False
            return True

        rested = lambda p: last_day[p] is None or last_day[p] < d - 1
        if variant == 0:
            key = lambda p: (-left[p], rested(p), p)
        elif variant == 1:
            key = lambda p: (-left[p], not rested(p), p)
        elif variant == 2:
            key = lambda p: (-left[p], -(last_end[p] or 0), p)
        else:
            key = lambda p: (-left[p], (last_end[p] or 0), p)
        cands = sorted(range(P), key=key)
        m = _match_slots(inst[d], cands, compatible)
        if m is None:
            return None
        for j, p in m.items():
            k = inst[d][j]
            t = slots[k]
            if last_end[p] is not None and t.start - last_end[p] >= rest_off:
                had_long[p] = True
            got[p][d] = k
            left[p] -= 1
            paid[p] += t.paid
            last_end[p] = t.end
            last_day[p] = d
            if first_slot[p] is None:
                first_slot[p] = k
        # capacity check for the rest of the week
        if d < 6 and sum(left) < future_need - len(inst[d]):
            return None

    out = []
    for p in range(P):
        ks = [got[p][d] for d in range(7) if got[p][d] is not None]
        if ks:
            out.append((kinds[p], ks))
    return out


def _best_greedy(slots: List[SlotType], rp: RosterParams, lb: int, deadline: float):
    """Smallest valid greedy roster: day-wise matching heuristic from the lower
    bound up (verified), falling back to the simple least-worked-first one."""
    simple = _greedy(slots, rp)
    cap = len(simple) if simple else sum(t.count for t in slots)
    for P in range(lb, cap):
        if time.time() > deadline:
            break
        for variant in range(4):
            g = _greedy_daywise(slots, rp, P, variant)
            if g is not None and _valid(g, slots, rp):
                return g
    return simple if simple is not None and _valid(simple, slots, rp) else None


def _valid(assign, slots, rp) -> bool:
    w = _windows(rp)
    res = RosterResult("", _people_from_assignment(assign, slots, "x"), "", {}, rp, 0,
                       {k: v[0] for k, v in w.items()},
                       {k: (v[1], v[2]) for k, v in w.items()})
    return not verify_roster(res, slots)


def _solve(m, rp, seconds, hint=None):
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = max(1.0, seconds)
    import os
    s.parameters.num_workers = rp.num_workers or max(4, min(16, os.cpu_count() or 4))
    if hint:
        for var, val in hint:
            m.add_hint(var, val)
    st = s.solve(m)
    return s, st


def _hint_from_assignment(v, assign, slots, P):
    """CP-SAT hint from [(kind, [k,...]), ...] sorted the way the model expects."""
    assign = sorted(assign, key=lambda q: -sum(slots[k].paid for k in q[1]))
    hint = []
    for p in range(P):
        kind, ks = assign[p] if p < len(assign) else (None, [])
        kset = set(ks)
        for k in range(len(slots)):
            hint.append((v["y"][p, k], int(k in kset)))
        hint += [(v["used"][p], int(kind is not None)),
                 (v["t1"][p], int(kind == "FT 1-off")),
                 (v["t2"][p], int(kind == "FT 2-off")),
                 (v["tp"][p], int(kind == "PT"))]
    return hint


def _people_from_assignment(assign, slots, occ):
    people = []
    for kind, ks in assign:
        days: List[Optional[SlotType]] = [None] * 7
        for k in ks:
            days[slots[k].day] = slots[k]
        people.append(Person("", kind, days, sum(slots[k].paid for k in ks)))
    people.sort(key=lambda q: ({"FT 1-off": 0, "FT 2-off": 1, "PT": 2}[q.kind], -q.paid_min))
    for i, q in enumerate(people, 1):
        q.pid = f"{occ[:10]}-{i:03d}"
    return people


def _assignment(s, v, slots, P):
    out = []
    for p in range(P):
        if not s.value(v["used"][p]):
            continue
        kind = ("FT 1-off" if s.value(v["t1"][p]) else
                "FT 2-off" if s.value(v["t2"][p]) else "PT")
        out.append((kind, [k for k in range(len(slots)) if s.value(v["y"][p, k])]))
    return out


def solve_roster(slots: List[SlotType], rp: RosterParams, occupation: str = "",
                 callback=None) -> RosterResult:
    t0 = time.time()
    lbs = lower_bounds(slots, rp)
    w = _windows(rp)
    targets = {k: v[0] for k, v in w.items()}
    windows = {k: (v[1], v[2]) for k, v in w.items()}
    if not slots:
        return RosterResult(occupation, [], "EMPTY", lbs, rp, 0.0, targets, windows)

    def result(assign, status, msg=""):
        return RosterResult(occupation, _people_from_assignment(assign, slots, occupation),
                            status, lbs, rp, time.time() - t0, targets, windows, msg)

    lb = max(lbs.values())
    total_shifts = sum(t.count for t in slots)

    # Warm start: greedy roster (valid by construction, checked again below)
    greedy = _best_greedy(slots, rp, lb, t0 + 0.15 * rp.time_limit_sec)
    if greedy is not None and verify_roster(result(greedy, "greedy"), slots):
        greedy = _greedy(slots, rp)            # planned heuristic broke a rule → simple one
        if greedy is not None and verify_roster(result(greedy, "greedy"), slots):
            greedy = None
    if callback:
        callback(f"{occupation}: greedy start — "
                 f"{len(greedy) if greedy else 'none'} people (lower bound {lb})")

    P = max(lb + max(3, math.ceil(0.10 * lb)), len(greedy) if greedy else 0)
    P = min(total_shifts, P)
    budget1 = rp.time_limit_sec * 0.6

    # Pass 1 — fewest people
    best = greedy
    status1 = "GREEDY" if greedy else "NONE"
    if not greedy or len(greedy) > lb:
        m, v = _build(slots, rp, P, sort_by_hours=False)
        m.add(sum(v["used"]) >= lb)
        if greedy:
            m.add(sum(v["used"]) <= len(greedy))
        m.minimize((P + 1) * sum(v["used"]) + sum(v["tp"]))
        if callback:
            callback(f"{occupation}: roster pass 1 — fewest people (pool {P})")
        s, st = _solve(m, rp, budget1,
                       _hint_from_assignment(v, greedy, slots, P) if greedy else None)
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            cand = _assignment(s, v, slots, P)
            if best is None or len(cand) <= len(best):
                best = cand
                status1 = s.status_name(st)
        elif best is None:
            return RosterResult(occupation, [], s.status_name(st), lbs, rp,
                                time.time() - t0, targets, windows,
                                message="No roster found — try a longer time limit, "
                                        "a wider hours window or another off mode.")
    else:
        status1 = "OPTIMAL (greedy = lower bound)"

    n_people = len(best)
    n_pt = sum(1 for kind, _ in best if kind == "PT")

    # Pass 2 — even hours at that headcount
    left = rp.time_limit_sec - (time.time() - t0)
    if left < 1:
        return result(best, f"{status1} / no time for balancing")
    if callback:
        callback(f"{occupation}: roster pass 2 — balancing hours for {n_people} people")
    P2 = n_people
    m2, v2 = _build(slots, rp, P2, fix_people=n_people, fix_pt=n_pt)
    m2.minimize(1000 * v2["maxdev"] + sum(v2["dev"]) + 20 * sum(v2["under"]))
    s2, st2 = _solve(m2, rp, left, _hint_from_assignment(v2, best, slots, P2))
    if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return result(_assignment(s2, v2, slots, P2), f"{status1} / {s2.status_name(st2)}")
    return result(best, f"{status1} / balancing {s2.status_name(st2)}")


# ── Independent check ────────────────────────────────────────────────────────
def verify_roster(res: RosterResult, slots: List[SlotType]) -> List[str]:
    """Re-check a roster against the rules without the model. [] = all good."""
    rp = res.params
    errs = []
    need = {(t.day, t.code): t.count for t in slots}
    got: Dict[Tuple[int, str], int] = {}
    for q in res.people:
        for d, t in enumerate(q.shifts):
            if t is not None:
                if t.day != d:
                    errs.append(f"{q.pid}: shift {t.code} filed under wrong day")
                got[(t.day, t.code)] = got.get((t.day, t.code), 0) + 1
        sh = sorted((t for t in q.shifts if t is not None), key=lambda t: t.start)
        if not sh:
            errs.append(f"{q.pid}: no shifts"); continue
        gaps = [b.start - a.end for a, b in zip(sh, sh[1:])]
        if rp.circular_week:
            gaps.append(sh[0].start + WEEK_MIN - sh[-1].end)
        if any(g < rp.min_rest_hours * 60 for g in gaps):
            errs.append(f"{q.pid}: rest {min(gaps) / 60:.1f} h < {rp.min_rest_hours} h")
        lim = {"FT 1-off": 6, "FT 2-off": 5, "PT": rp.pt_max_shifts}[q.kind]
        if len(sh) > lim:
            errs.append(f"{q.pid}: {len(sh)} shifts > {lim} ({q.kind})")
        if q.kind == "FT 1-off":
            longest = max(gaps) if gaps else WEEK_MIN
            if not rp.circular_week:   # rest before first / after last shift counts
                longest = max(longest, sh[0].start - 0, WEEK_MIN - sh[-1].end)
            if longest < rp.off_rest_hours * 60:
                errs.append(f"{q.pid}: longest rest {longest / 60:.1f} h < "
                            f"{rp.off_rest_hours} h (1-off)")
        hi = res.window_by_kind[q.kind][1]
        if q.paid_hours > hi + 1e-6:
            errs.append(f"{q.pid}: {q.paid_hours:.1f} h > max {hi:.1f} h")
        if q.kind == "PT" and not rp.allow_part_time:
            errs.append(f"{q.pid}: part-time not allowed")
        if rp.off_mode == "1" and q.kind == "FT 2-off" or rp.off_mode == "2" and q.kind == "FT 1-off":
            errs.append(f"{q.pid}: {q.kind} not allowed in off mode {rp.off_mode}")
    if got != need:
        errs.append(f"slot counts differ: need {sum(need.values())}, "
                    f"assigned {sum(got.values())}")
    if rp.off_mode == "1.5":
        a, b = res.count("FT 1-off"), res.count("FT 2-off")
        if abs(a - b) > 1:
            errs.append(f"1.5 off split {a}/{b} not balanced")
    return errs
