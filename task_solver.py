"""
Simplex – Task-based Shift Optimiser (OR-Tools CP-SAT)
======================================================
Same shift model as the curve-based ``solver.py`` (candidate shifts, shared
shift activation across occupations, entry/exit and headcount limits, force
include/exclude, transition penalty), but the input is a list of TASKS
instead of a workload curve.

Pipeline
  1. Shifts + task timing (MIP, SCIP bundled with OR-Tools)
       Every task gets one start time inside its window (Boolean option
       variables).  The chosen starts build the task load of each
       occupation, every 5 minutes, including the walk to the stand.
       Shifts must cover that load with PRODUCTIVE staff: on shift, after
       the briefing, before the close-out and not on break.
       Stage 1 minimises paid hours; stage 2 starts from that plan and adds
       the transition penalty (the curve solver's objective).
       SCIP's LP bound suits this covering model far better than CP-SAT
       (≈1 % gap in a minute for a full week vs. ≈50 % with CP-SAT).
  2. Assignment check (greedy, then CP-SAT if needed)
       Tasks are given to individual workers: one task at a time, the whole
       task inside one shift, travel time between stands, no task during
       the break.
  3. Cuts
       A task that could not be given to anyone gets a containment cut
       ("enough shifts must contain this task whole") and step 1 runs again
       from the previous solution.  After the last round any task still left
       gets an extra worker on the shortest fitting shift, so every task is
       always staffed.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

from solver import (
    INTERVALS_PER_DAY, INTERVALS_PER_HOUR, TOTAL_INTERVALS, DAY_NAMES,
    CandidateShift, SolverParams, PhaseOneResult, OccupationResult,
    MultiCurveResult, generate_candidate_shifts, break_hours,
    shift_type_summary, build_weekly_report_xlsx,
)
from tasks import Task, WEEK_MIN, DAY_ABBR3

STEP = 5  # minutes per interval


@dataclass
class TaskParams(SolverParams):
    """Shift rules (inherited) + task-based work rules."""
    briefing_min: int = 15        # start of shift: briefing / equipment
    closeout_min: int = 10        # end of shift: close-out / handover
    travel_min: int = 5           # walk between stands (0 at the same stand)
    breaks: bool = True           # break per break_hours(), placed mid-shift
    assign_rounds: int = 3        # assignment-check rounds with cuts
    max_task_options: int = 13    # start times tried per task (evenly spread)
    contain_all: bool = True      # task-window constraints for every task


# ── Results ──────────────────────────────────────────────────────────────────
@dataclass
class Worker:
    wid: str
    occupation: str
    shift: CandidateShift
    prod_start: int               # global interval, first productive interval
    prod_end: int                 # global interval, end (exclusive)
    break_start: int              # global interval (== break_end if no break)
    break_end: int
    extra: bool = False           # added by the repair step
    tasks: List[int] = field(default_factory=list)   # task ids in order
    extended_from: str = ""       # original shift code if the repair lengthened it

    @property
    def code(self) -> str:
        return self.shift.shift_code


@dataclass
class TaskResult:
    multi: MultiCurveResult              # shift plan in the curve-app format
    tasks: List[Task]
    start: Dict[int, int]                # task id → chosen start (minutes)
    assigned: Dict[int, List[str]]       # task id → worker ids
    workers: List[Worker]
    unassigned: List[int]                # task ids still short of staff
    productive: Dict[str, np.ndarray]    # occupation → productive staff curve
    rounds: int
    cuts: int
    extra_workers: int
    extended_shifts: int
    log: List[str]

    @property
    def status(self) -> str:
        return self.multi.combined_phase1.status

    @property
    def params(self) -> TaskParams:
        return self.multi.params  # type: ignore[return-value]


# ── Geometry helpers ─────────────────────────────────────────────────────────
def _iv(minutes: int) -> int:
    return int(math.ceil(minutes / STEP))


def shift_windows(s: CandidateShift, p: TaskParams) -> Optional[Tuple[int, int, int, int]]:
    """(prod_start, prod_end, break_start, break_end) in global intervals.

    None when the briefing and close-out leave no productive time.
    """
    dur = s.duration_intervals
    ps = s.global_start + _iv(p.briefing_min)
    pe = s.global_end - _iv(p.closeout_min)
    if pe - ps <= 0:
        return None
    bl = int(round(break_hours(s.duration_hours) * INTERVALS_PER_HOUR)) if p.breaks else 0
    if bl > 0:
        bs = s.global_start + max(0, (dur - bl) // 2)
        bs = min(max(bs, ps), pe - bl) if pe - ps > bl else ps
        be = bs + bl
    else:
        bs = be = ps
    return ps, pe, bs, be


def task_options(t: Task, p: TaskParams) -> List[int]:
    """Start intervals (global) the solver may pick for a task."""
    first = _iv(t.earliest)
    last = (t.latest_end - t.duration) // STEP
    if last < first:
        return [int(round(t.earliest / STEP))]
    opts = list(range(first, last + 1))
    k = max(1, p.max_task_options)
    if len(opts) > k:
        idx = np.unique(np.round(np.linspace(0, len(opts) - 1, k)).astype(int))
        opts = [opts[i] for i in idx]
    return opts


def occupancy(t: Task, start_iv: int, p: TaskParams) -> Tuple[int, int]:
    """Intervals a task keeps one worker busy: walk to the stand + the task."""
    return start_iv - _iv(p.travel_min), start_iv + _iv(t.duration)


def _occ_intervals(a: int, b: int, circular: bool):
    for k in range(a, b):
        if circular:
            yield k % TOTAL_INTERVALS
        elif 0 <= k < TOTAL_INTERVALS:
            yield k


# ── Coverage map with briefing / close-out / break ───────────────────────────
def shift_window_table(shifts: List[CandidateShift], p: TaskParams) -> np.ndarray:
    """Row s.idx = (prod_start, prod_end, break_start, break_end); -1 if unusable."""
    win = np.full((len(shifts), 4), -1, dtype=np.int64)
    for s in shifts:
        w = shift_windows(s, p)
        if w is not None:
            win[s.idx] = w
    return win


def build_task_coverage_map(shifts: List[CandidateShift], win: np.ndarray):
    """Interval → shifts whose PRODUCTIVE time covers it (off briefing,
    close-out and break)."""
    cov: Dict[int, List[int]] = {t: [] for t in range(TOTAL_INTERVALS)}
    usable = []
    for s in shifts:
        ps, pe, bs, be = (int(v) for v in win[s.idx])
        if ps < 0:
            continue
        usable.append(s.idx)
        for k in range(ps, pe):
            if bs <= k < be:
                continue
            cov[k % TOTAL_INTERVALS].append(s.idx)
    return cov, usable


def break_room_table(shifts: List[CandidateShift], win: np.ndarray) -> np.ndarray:
    """Row s.idx = (earliest break start, latest break end, break length).

    The break may move inside the middle half of the shift (see
    ``break_window``); this is the shift-level version used by the model.
    """
    out = np.zeros((len(shifts), 3), dtype=np.int64)
    for s in shifts:
        ps, pe, bs, be = (int(v) for v in win[s.idx])
        bl = be - bs
        if ps < 0 or bl <= 0:
            continue
        dur = s.duration_intervals
        lo = max(ps, s.global_start + dur // 4)
        hi = min(pe, s.global_end - dur // 4)
        if hi - lo < bl:
            lo, hi = bs, be
        out[s.idx] = (lo, hi, bl)
    return out


def _containing(win: np.ndarray, a: int, b: int, circular: bool,
                room: Optional[np.ndarray] = None) -> np.ndarray:
    """Shift indices whose productive window holds [a, b) whole.

    Without ``room`` the fixed mid-shift break must not touch the task; with
    it the break only needs space before or after the task in its window.
    """
    ps, pe, bs, be = win[:, 0], win[:, 1], win[:, 2], win[:, 3]
    ok = np.zeros(len(win), dtype=bool)
    offs = (0, TOTAL_INTERVALS) if circular else (0,)
    a = max(a, 0) if not circular else a
    for o in offs:
        aa, bb = a + o, b + o
        inside = (ps >= 0) & (ps <= aa) & (pe >= bb)
        if room is None:
            free = ~((bs < bb) & (be > aa) & (be > bs))
        else:
            lo, hi, bl = room[:, 0], room[:, 1], room[:, 2]
            free = (bl == 0) | (np.minimum(aa, hi) - lo >= bl) | (hi - np.maximum(bb, lo) >= bl)
        ok |= inside & free
    return np.nonzero(ok)[0]


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – Shifts + task timing (MIP, SCIP bundled with OR-Tools)
# ══════════════════════════════════════════════════════════════════════════════
_STATUS = {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE",
           pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
           pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
           pywraplp.Solver.ABNORMAL: "ABNORMAL",
           pywraplp.Solver.NOT_SOLVED: "UNKNOWN"}


class _Model:
    """Shift counts x[o][s], shared activation z[s], task start options y.

    Every shift rule of the curve-based solver is here in linear form:
    force include, max unique shift codes, entry/exit limits per day,
    headcount limits (combined, per occupation, per shift code) and the
    transition penalty on shift activation.
    """

    def __init__(self, tasks: List[Task], occ_names: List[str], p: TaskParams,
                 shifts: List[CandidateShift], win: np.ndarray, callback=None):
        cb = callback or (lambda *_: None)
        self.p, self.shifts, self.win = p, shifts, win
        self.occ_names = occ_names
        self.occ_of = {n: i for i, n in enumerate(occ_names)}
        n_occ = len(occ_names)
        S = pywraplp.Solver.CreateSolver("SCIP")
        if S is None:
            raise RuntimeError("SCIP is not available in this OR-Tools build")
        self.S = S
        self.cov, usable = build_task_coverage_map(shifts, win)
        coverable = np.array([bool(self.cov[k]) for k in range(TOTAL_INTERVALS)])

        # Start options that some shift can staff (drop e.g. Sun 23:55 with
        # a non-circular week)
        self.opts: Dict[int, List[int]] = {}
        self.dropped: List[int] = []
        for t in tasks:
            ok = [q for q in task_options(t, p)
                  if all(coverable[k] for k in _occ_intervals(*occupancy(t, q, p),
                                                               p.circular_week))]
            if ok:
                self.opts[t.tid] = ok
            else:
                self.dropped.append(t.tid)
        self.tasks = [t for t in tasks if t.tid in self.opts]
        if self.dropped:
            cb(f"{len(self.dropped)} task(s) cannot be staffed by any allowed shift "
               "(e.g. past the last shift end) — left out")

        # Most staff of an occupation ever needed at one time
        ub = np.zeros((n_occ, TOTAL_INTERVALS), dtype=int)
        for t in self.tasks:
            o = self.occ_of[t.occupation]
            lo = min(occupancy(t, q, p)[0] for q in self.opts[t.tid])
            hi = max(occupancy(t, q, p)[1] for q in self.opts[t.tid])
            for k in _occ_intervals(lo, hi, p.circular_week):
                ub[o, k] += t.staff
        base = [self._min_floor(o) for o in range(n_occ)]

        # x[o][s]: workers; bound = peak need inside the shift's productive time
        self.x: Dict[int, Dict[int, object]] = {o: {} for o in range(n_occ)}
        self.ub: Dict[Tuple[int, int], int] = {}
        for si in usable:
            ps, pe, bs, be = (int(v) for v in win[si])
            ks = [k % TOTAL_INTERVALS for k in range(ps, pe) if not (bs <= k < be)]
            for o in range(n_occ):
                m = max(int(ub[o, ks].max()) if ks else 0, base[o])
                if m > 0:
                    self.x[o][si] = S.IntVar(0, m, f"x_{o}_{si}")
                    self.ub[(o, si)] = m
        self.z: Dict[int, object] = {}
        for si in usable:
            terms = [(o, self.x[o][si]) for o in range(n_occ) if si in self.x[o]]
            if not terms:
                continue
            zv = S.BoolVar(f"z_{si}")
            self.z[si] = zv
            for o, xv in terms:
                S.Add(xv <= self.ub[(o, si)] * zv)
        cb(f"{sum(len(v) for v in self.x.values()):,} shift variables "
           f"({len(self.z):,} usable shifts × occupations)")
        self._add_rules(cb)

        # Task start options
        self.y: Dict[int, Dict[int, object]] = {}
        load: List[Dict[int, list]] = [defaultdict(list) for _ in range(n_occ)]
        n_y = 0
        for t in self.tasks:
            o = self.occ_of[t.occupation]
            ys = {}
            if len(self.opts[t.tid]) == 1:
                ys[self.opts[t.tid][0]] = 1
            else:
                for q in self.opts[t.tid]:
                    ys[q] = S.BoolVar(f"y_{t.tid}_{q}")
                    n_y += 1
                S.Add(sum(ys.values()) == 1)
            self.y[t.tid] = ys
            for q, v in ys.items():
                for k in _occ_intervals(*occupancy(t, q, p), p.circular_week):
                    load[o][k].append((t.staff, v))
        cb(f"{len(self.tasks):,} tasks, {n_y:,} start-time choices")

        n_con = 0
        for o in range(n_occ):
            xo = self.x[o]
            for k, terms in load[o].items():
                const = sum(c for c, v in terms if isinstance(v, int))
                rhs = sum(c * v for c, v in terms if not isinstance(v, int)) + const
                S.Add(sum(xo[si] for si in self.cov[k] if si in xo) >= rhs)
                n_con += 1
        cb(f"Task-load constraints: {n_con:,} (5-min slots × occupations)")

        self.room = break_room_table(shifts, win)
        self.n_cuts = 0
        self.cut_tasks: set = set()
        self.cut_keys: set = set()
        if p.contain_all:
            for t in self.tasks:
                self.add_containment_cut(t)
            cb(f"Task-window constraints: {self.n_cuts:,} (tasks done whole "
               "by one worker)")

        self.paid = sum(xv * self.shifts[si].duration_intervals
                        for o in self.x for si, xv in self.x[o].items())
        self.penalty = sum(self.z.values()) * (2 * p.transition_penalty)

    # ── shift rules (same meaning as solver._build_multi_curve_model) ────
    def _min_floor(self, o: int) -> int:
        p, v = self.p, 0
        if p.occ_min_headcount_per_day and p.occ_min_headcount_per_day[o]:
            v = max(v, max(p.occ_min_headcount_per_day[o]))
        if p.min_headcount_per_day:
            v = max(v, max(p.min_headcount_per_day))
        if p.occ_headcount_per_shift_code and p.occ_headcount_per_shift_code[o]:
            for spec in p.occ_headcount_per_shift_code[o].values():
                if isinstance(spec, dict) and spec.get("min"):
                    v = max(v, max(spec["min"]))
        return v

    def _add_rules(self, cb):
        p, S, shifts, x, z = self.p, self.S, self.shifts, self.x, self.z
        n_occ = len(self.occ_names)
        by_code: Dict[str, List[int]] = defaultdict(list)
        by_day: Dict[int, List[int]] = defaultdict(list)
        for si in z:
            by_code[shifts[si].shift_code].append(si)
            by_day[shifts[si].day].append(si)

        if p.force_include_shifts:
            for code in p.force_include_shifts:
                idxs = by_code.get(code, [])
                if idxs:
                    S.Add(sum(z[i] for i in idxs) >= 1)
                    for i in idxs:        # active means really used
                        S.Add(z[i] <= sum(x[o][i] for o in range(n_occ) if i in x[o]))
            cb(f"Force-include {len(p.force_include_shifts)} shift code(s)")

        if p.max_unique_shifts > 0:
            u = []
            for code, idxs in by_code.items():
                uv = S.BoolVar(f"u_{code}")
                for i in idxs:
                    S.Add(z[i] <= uv)
                u.append(uv)
            S.Add(sum(u) <= p.max_unique_shifts)
            cb(f"Max unique shift codes ≤ {p.max_unique_shifts}")

        def _limit_distinct(groups: Dict[int, Dict[int, List[int]]], limits, tag):
            for day in range(7):
                lim = limits[day]
                if lim <= 0 or len(groups[day]) <= lim:
                    continue
                flags = []
                for key, idxs in groups[day].items():
                    f = S.BoolVar(f"{tag}_{day}_{key}")
                    for i in idxs:
                        S.Add(z[i] <= f)
                    flags.append(f)
                S.Add(sum(flags) <= lim)

        if p.max_entries_per_day:
            g = {d: defaultdict(list) for d in range(7)}
            for si in z:
                g[shifts[si].day][shifts[si].start_interval].append(si)
            _limit_distinct(g, p.max_entries_per_day, "ent")
            cb(f"Max entries/day: {p.max_entries_per_day}")
        if p.max_exits_per_day:
            g = {d: defaultdict(list) for d in range(7)}
            for si in z:
                last = (shifts[si].global_end - 1) % TOTAL_INTERVALS
                g[last // INTERVALS_PER_DAY][last % INTERVALS_PER_DAY + 1].append(si)
            _limit_distinct(g, p.max_exits_per_day, "ext")
            cb(f"Max exits/day: {p.max_exits_per_day}")

        def _day_sum(occs, day, codes=None):
            return sum(x[o][i] for o in occs for i in by_day[day]
                       if i in x[o] and (codes is None or shifts[i].shift_code in codes))

        for o in range(n_occ):
            if p.occ_max_headcount_per_day and p.occ_max_headcount_per_day[o]:
                for d, lim in enumerate(p.occ_max_headcount_per_day[o]):
                    if lim > 0:
                        S.Add(_day_sum([o], d) <= lim)
            if p.occ_min_headcount_per_day and p.occ_min_headcount_per_day[o]:
                for d, lim in enumerate(p.occ_min_headcount_per_day[o]):
                    if lim > 0:
                        S.Add(_day_sum([o], d) >= lim)
            if p.occ_headcount_per_shift_code and p.occ_headcount_per_shift_code[o]:
                for code, spec in p.occ_headcount_per_shift_code[o].items():
                    mx = spec if isinstance(spec, list) else spec.get("max")
                    mn = None if isinstance(spec, list) else spec.get("min")
                    for d in range(7):
                        if mx and mx[d] > 0:
                            S.Add(_day_sum([o], d, {code}) <= mx[d])
                        if mn and mn[d] > 0:
                            S.Add(_day_sum([o], d, {code}) >= mn[d])
        allo = list(range(n_occ))
        for d in range(7):
            if p.max_headcount_per_day and p.max_headcount_per_day[d] > 0:
                S.Add(_day_sum(allo, d) <= p.max_headcount_per_day[d])
            if p.min_headcount_per_day and p.min_headcount_per_day[d] > 0:
                S.Add(_day_sum(allo, d) >= p.min_headcount_per_day[d])

    # ── cuts / solve / read ──────────────────────────────────────────────
    def _option_table(self):
        """Per occupation: arrays of option windows with staff and variable."""
        if hasattr(self, "_opt_tab"):
            return self._opt_tab
        tab = defaultdict(lambda: ([], [], [], []))
        for t in self.tasks:
            o = self.occ_of[t.occupation]
            for q, v in self.y[t.tid].items():
                a, b = occupancy(t, q, self.p)
                A, B, ST, V = tab[o]
                A.append(a); B.append(b); ST.append(t.staff); V.append(v)
        self._opt_tab = {o: (np.array(A), np.array(B), np.array(ST), V)
                         for o, (A, B, ST, V) in tab.items()}
        return self._opt_tab

    def add_window_cut(self, o: int, a: int, b: int):
        """Workers whose productive time spans [a, b) ≥ staff of every task
        option that spans [a, b).  (Hall's condition: those tasks can only be
        done by those workers.)"""
        key = (o, a, b)
        if key in self.cut_keys:
            return
        self.cut_keys.add(key)
        A, B, ST, V = self._option_table()[o]
        sel = np.nonzero((A <= a) & (B >= b))[0]
        if len(sel) == 0:
            return
        rhs = sum(int(ST[i]) * V[i] for i in sel)
        idxs = _containing(self.win, a, b, self.p.circular_week, self.room)
        lhs = sum(self.x[o][int(i)] for i in idxs if int(i) in self.x[o])
        self.S.Add(lhs >= rhs)
        self.n_cuts += 1

    def add_containment_cut(self, t: Task):
        """Window cuts on every start option of a task."""
        if t.tid in self.cut_tasks or t.tid not in self.y:
            return
        self.cut_tasks.add(t.tid)
        o = self.occ_of[t.occupation]
        for q in self.y[t.tid]:
            self.add_window_cut(o, *occupancy(t, q, self.p))

    def solve(self, time_limit: float, with_penalty: bool,
              hint: Optional[Tuple[dict, dict]] = None) -> str:
        S = self.S
        obj = self.paid + (self.penalty if with_penalty and self.p.transition_penalty > 0 else 0)
        S.Minimize(obj)
        if hint is not None:
            counts, starts = hint
            hv, hval = [], []
            for o in self.x:
                for si, xv in self.x[o].items():
                    hv.append(xv)
                    hval.append(float(counts[o].get(si, 0)))
            for si, zv in self.z.items():
                hv.append(zv)
                hval.append(float(any(counts[o].get(si, 0) for o in counts)))
            for tid, ys in self.y.items():
                for q, v in ys.items():
                    if not isinstance(v, int):
                        hv.append(v)
                        hval.append(float(starts.get(tid) == q))
            S.SetHint(hv, hval)
        S.SetTimeLimit(int(max(5.0, time_limit) * 1000))
        S.SetSolverSpecificParametersAsString("limits/gap = 0.002\n")
        code = S.Solve()
        self.last_bound = S.Objective().BestBound() if code in (0, 1) else None
        return _STATUS.get(code, "UNKNOWN")

    def read(self):
        counts = {o: {si: int(round(xv.solution_value()))
                      for si, xv in self.x[o].items() if xv.solution_value() > 0.5}
                  for o in self.x}
        starts = {}
        for t in self.tasks:
            for q, v in self.y[t.tid].items():
                if isinstance(v, int) or v.solution_value() > 0.5:
                    starts[t.tid] = q
                    break
        return counts, starts


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – Assignment check (greedy, best fit)
# ══════════════════════════════════════════════════════════════════════════════
def _make_workers(shifts, counts, occ_names, win, extra=None) -> List[Worker]:
    workers: List[Worker] = []
    extra = extra or {}
    for o, name in enumerate(occ_names):
        n = 0
        for s in shifts:
            c = counts[o].get(s.idx, 0)
            for k in range(c):
                ps, pe, bs, be = (int(v) for v in win[s.idx])
                n += 1
                workers.append(Worker(
                    f"{name[:3].upper()}-{DAY_ABBR3[s.day]}-{n:03d}", name, s,
                    ps, pe, bs, be,
                    extra=k >= c - extra.get((o, s.idx), 0)))
    return workers


def assign_tasks(tasks: List[Task], starts: Dict[int, int],
                 workers: List[Worker], p: TaskParams
                 ) -> Tuple[Dict[int, List[str]], List[int]]:
    """Give every task unit to one worker; returns (assigned, unassigned ids)."""
    tr = _iv(p.travel_min)
    by_occ: Dict[str, List[Worker]] = defaultdict(list)
    for w in workers:
        w.tasks = []
        by_occ[w.occupation].append(w)
    state = {w.wid: (None, None) for w in workers}   # last_end, last_loc
    order = sorted(tasks, key=lambda t: (starts[t.tid], -t.duration, t.tid))
    assigned: Dict[int, List[str]] = {}
    unassigned: List[int] = []
    for t in order:
        st = starts[t.tid]
        en = st + _iv(t.duration)
        got: List[str] = []
        for _ in range(t.staff):
            best, best_key = None, None
            for w in by_occ[t.occupation]:
                if w.wid in got:
                    continue
                for off in ((0, TOTAL_INTERVALS) if p.circular_week else (0,)):
                    s0, e0 = st + off, en + off
                    if e0 > w.prod_end or s0 < w.prod_start:
                        continue
                    last_end, last_loc = state[w.wid]
                    same = (last_end is not None and t.location
                            and last_loc == t.location)
                    need = 0 if same else tr
                    avail = w.prod_start if last_end is None else last_end
                    if last_end is None:
                        if s0 - tr < w.prod_start:
                            continue
                    elif avail + need > s0:
                        continue
                    if w.break_end > w.break_start and \
                            w.break_start < e0 and w.break_end > s0 - need:
                        continue
                    key = (w.prod_end, 0 if same else 1, -avail)
                    if best_key is None or key < best_key:
                        best, best_key = (w, e0), key
                    break
            if best is None:
                break
            w, e0 = best
            got.append(w.wid)
            w.tasks.append(t.tid)
            state[w.wid] = (e0, t.location)
        assigned[t.tid] = got
        if len(got) < t.staff:
            unassigned.append(t.tid)
    # keep each worker's task list in time order
    for w in workers:
        w.tasks.sort(key=lambda i: starts[i])
    return assigned, unassigned


# ══════════════════════════════════════════════════════════════════════════════
# Step 2b – Exact assignment check (CP-SAT), greedy result as hint
# ══════════════════════════════════════════════════════════════════════════════
def break_window(w: "Worker") -> Tuple[int, int, int]:
    """(earliest start, latest end, length) of the worker's break.

    The break may sit anywhere in the middle half of the shift (inside the
    productive time).  Length follows break_hours(); 0 = no break.
    """
    bl = w.break_end - w.break_start
    if bl <= 0:
        return 0, 0, 0
    dur = w.shift.duration_intervals
    lo = max(w.prod_start, w.shift.global_start + dur // 4)
    hi = min(w.prod_end, w.shift.global_end - dur // 4)
    if hi - lo < bl:
        return w.break_start, w.break_end, bl
    return lo, hi, bl


def assign_tasks_exact(tasks: List[Task], starts: Dict[int, int],
                       workers: List[Worker], p: TaskParams,
                       time_limit: float = 10.0,
                       greedy: Optional[Dict[int, List[str]]] = None
                       ) -> Tuple[Dict[int, List[str]], List[int]]:
    """Maximum number of staffed task places, one task at a time per worker.

    Every task keeps its walk-in time in front (conservative: also at the
    same stand), a task is done whole by one worker, and each worker's break
    is placed by the model anywhere in the middle half of the shift.
    """
    tr = _iv(p.travel_min)
    m = cp_model.CpModel()
    a_vars: Dict[Tuple[int, int], object] = {}
    per_worker: Dict[int, list] = defaultdict(list)
    per_task: Dict[int, list] = defaultdict(list)
    by_occ: Dict[str, List[int]] = defaultdict(list)
    brk: Dict[int, object] = {}
    for wi, w in enumerate(workers):
        by_occ[w.occupation].append(wi)
        lo, hi, bl = break_window(w)
        if bl > 0:
            bs = m.new_int_var(lo, hi - bl, f"b_{wi}")
            per_worker[wi].append(m.new_fixed_size_interval_var(bs, bl, f"bi_{wi}"))
            brk[wi] = bs
    wid_index = {w.wid: wi for wi, w in enumerate(workers)}
    for t in tasks:
        st = starts[t.tid]
        en = st + _iv(t.duration)
        for wi in by_occ[t.occupation]:
            w = workers[wi]
            for off in ((0, TOTAL_INTERVALS) if p.circular_week else (0,)):
                s0, e0 = st - tr + off, en + off
                if s0 < w.prod_start or e0 > w.prod_end:
                    continue
                v = m.new_bool_var(f"a_{t.tid}_{wi}")
                iv = m.new_optional_fixed_size_interval_var(
                    s0, e0 - s0, v, f"i_{t.tid}_{wi}")
                a_vars[(t.tid, wi)] = v
                per_worker[wi].append(iv)
                per_task[t.tid].append(v)
                break
    for wi, ivs in per_worker.items():
        if len(ivs) > 1:
            m.add_no_overlap(ivs)
    for t in tasks:
        if per_task[t.tid]:
            m.add(sum(per_task[t.tid]) <= t.staff)
    m.maximize(sum(a_vars.values()))
    if greedy:
        for tid, ws in greedy.items():
            for wid in ws:
                key = (tid, wid_index.get(wid, -1))
                if key in a_vars:
                    m.add_hint(a_vars[key], 1)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(2.0, time_limit)
    import os as _os
    solver.parameters.num_workers = max(4, min(16, _os.cpu_count() or 4))
    code = solver.solve(m)
    if code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {}, [t.tid for t in tasks]
    assigned: Dict[int, List[str]] = defaultdict(list)
    for wi, w in enumerate(workers):
        w.tasks = []
        if wi in brk:
            bl = w.break_end - w.break_start
            w.break_start = solver.value(brk[wi])
            w.break_end = w.break_start + bl
    for (tid, wi), v in a_vars.items():
        if solver.value(v):
            assigned[tid].append(workers[wi].wid)
            workers[wi].tasks.append(tid)
    for w in workers:
        w.tasks.sort(key=lambda i: starts[i])
    unassigned = [t.tid for t in tasks if len(assigned.get(t.tid, [])) < t.staff]
    return dict(assigned), unassigned


def _clusters(tasks: List[Task], starts: Dict[int, int], workers: List[Worker],
              p: TaskParams) -> List[Tuple[List[Task], List[Worker]]]:
    """Independent pieces: per occupation, groups of workers whose productive
    windows overlap in a chain.  A task belongs to the group whose time span
    holds it; no task can be done across two groups."""
    tr = _iv(p.travel_min)
    out = []
    by_occ: Dict[str, List[Worker]] = defaultdict(list)
    for w in workers:
        by_occ[w.occupation].append(w)
    tasks_by_occ: Dict[str, List[Task]] = defaultdict(list)
    for t in tasks:
        tasks_by_occ[t.occupation].append(t)
    for occ, ws in by_occ.items():
        ws = sorted(ws, key=lambda w: w.prod_start)
        groups: List[Tuple[int, int, List[Worker]]] = []
        for w in ws:
            if groups and w.prod_start < groups[-1][1]:
                lo, hi, g = groups[-1]
                groups[-1] = (lo, max(hi, w.prod_end), g + [w])
            else:
                groups.append((w.prod_start, w.prod_end, [w]))
        for lo, hi, g in groups:
            ts = [t for t in tasks_by_occ[occ]
                  if lo <= starts[t.tid] - tr and starts[t.tid] + _iv(t.duration) <= hi]
            out.append((ts, g))
    return out


def check_assignment(tasks, starts, workers, p, time_cap: float = 30.0):
    """Greedy first; groups with open tasks are re-solved exactly (CP-SAT)
    within ``time_cap`` seconds in total."""
    g_ass, g_un = assign_tasks(tasks, starts, workers, p)
    if not g_un:
        return g_ass, g_un
    failed = set(g_un)
    assigned = dict(g_ass)
    groups = [(ts, ws) for ts, ws in _clusters(tasks, starts, workers, p)
              if any(t.tid in failed for t in ts)]
    t_end = time.time() + time_cap
    # smallest groups first: they solve fast and to optimality
    groups.sort(key=lambda g: len(g[0]) * len(g[1]))
    for gi, (ts, ws) in enumerate(groups):
        remaining = t_end - time.time()
        if remaining < 1.0:
            break
        time_limit = max(1.5, remaining / (len(groups) - gi))
        snap = {w.wid: (list(w.tasks), w.break_start, w.break_end) for w in ws}
        hint = {t.tid: g_ass.get(t.tid, []) for t in ts}
        e_ass, e_un = assign_tasks_exact(ts, starts, ws, p, time_limit, hint)
        g_short = sum(t.staff - len(g_ass.get(t.tid, [])) for t in ts)
        e_short = sum(t.staff - len(e_ass.get(t.tid, [])) for t in ts)
        if e_ass and e_short < g_short:
            for t in ts:
                assigned[t.tid] = e_ass.get(t.tid, [])
        else:
            for w in ws:
                w.tasks, w.break_start, w.break_end = snap[w.wid]
    unassigned = [t.tid for t in tasks if len(assigned.get(t.tid, [])) < t.staff]
    return assigned, unassigned


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════
def solve_tasks(tasks: List[Task], params: Optional[TaskParams] = None,
                callback=None) -> TaskResult:
    p = params or TaskParams()
    log: List[str] = []

    def cb(msg):
        log.append(msg)
        if callback:
            callback(msg)

    t0 = time.time()
    occ_names: List[str] = []
    for t in tasks:
        if t.occupation not in occ_names:
            occ_names.append(t.occupation)
    budget = float(p.solver_time_limit_sec)

    shifts = generate_candidate_shifts(p)
    win = shift_window_table(shifts, p)
    cb(f"Generated {len(shifts):,} candidate shifts")
    cb(f"Work rules: briefing {p.briefing_min} min, close-out {p.closeout_min} min, "
       f"travel {p.travel_min} min, breaks {'mid-shift' if p.breaks else 'off'}")
    mdl = _Model(tasks, occ_names, p, shifts, win, cb)
    planned = mdl.tasks
    task_by_id = {t.tid: t for t in tasks}

    # Stage 1 – fewest paid hours
    pen = p.transition_penalty > 0
    cb("Model built – stage 1: fewest paid hours …")
    status = mdl.solve(budget * (0.5 if pen else 0.7), with_penalty=False)
    cb(f"Stage 1: {status}, {time.time() - t0:.1f}s")
    if status not in ("OPTIMAL", "FEASIBLE"):
        return _empty_result(tasks, occ_names, p, status, time.time() - t0, log)
    counts, starts = mdl.read()

    def objective(c):
        paid = sum(v * shifts[si].duration_intervals for o in c for si, v in c[o].items())
        act = len({si for o in c for si, v in c[o].items() if v})
        return paid + 2 * p.transition_penalty * act

    # Stage 2 – same objective as the curve solver (paid + transition penalty)
    if pen:
        st2 = mdl.solve(budget * 0.2, with_penalty=True, hint=(counts, starts))
        if st2 in ("OPTIMAL", "FEASIBLE"):
            c2, s2 = mdl.read()
            if objective(c2) < objective(counts):
                counts, starts, status = c2, s2, st2
        cb(f"Stage 2 (transition penalty {p.transition_penalty}): {st2}, "
           f"{time.time() - t0:.1f}s")

    cap = max(10.0, 0.15 * budget)
    workers = _make_workers(shifts, counts, occ_names, win)
    assigned, unassigned = check_assignment(planned, starts, workers, p, cap)
    cb(f"Assignment check: {len(planned) - len(unassigned):,}/{len(planned):,} tasks fully staffed")

    # Cut rounds – only while there are new cuts to add
    rounds = 0
    while unassigned and rounds < p.assign_rounds:
        before = mdl.n_cuts
        for tid in unassigned:
            mdl.add_containment_cut(task_by_id[tid])
        if mdl.n_cuts == before:
            break
        rounds += 1
        left = max(5.0, (budget - (time.time() - t0)) / (p.assign_rounds - rounds + 1))
        st3 = mdl.solve(left, with_penalty=pen, hint=(counts, starts))
        if st3 not in ("OPTIMAL", "FEASIBLE"):
            cb(f"Round {rounds}: {st3} — keeping the previous plan")
            break
        counts2, starts2 = mdl.read()
        workers2 = _make_workers(shifts, counts2, occ_names, win)
        ass2, un2 = check_assignment(planned, starts2, workers2, p, cap)
        better = (len(un2) < len(unassigned) or
                  (len(un2) == len(unassigned) and objective(counts2) < objective(counts)))
        cb(f"Round {rounds}: {mdl.n_cuts - before} new cuts → "
           f"{len(planned) - len(un2):,}/{len(planned):,} staffed"
           + ("" if better else " (previous plan kept)"))
        if better:
            status = st3
            counts, starts, workers, assigned, unassigned = (
                counts2, starts2, workers2, ass2, un2)

    # Patch – open tasks go to extra workers on the shortest fitting shift
    n_extra = n_ext = 0
    if unassigned:
        n_open = len(unassigned)
        n_extra, n_ext, unassigned = _patch(planned, starts, workers, assigned,
                                            unassigned, counts, shifts, win,
                                            mdl, p, cb)
        cb(f"Repair for {n_open} task(s): {n_ext} shift(s) lengthened, "
           f"{n_extra} worker(s) added")

    unassigned = list(unassigned) + list(mdl.dropped)
    elapsed = time.time() - t0
    res = _build_result(planned, tasks, occ_names, p, shifts, counts, starts,
                        assigned, unassigned, workers, win, status, elapsed,
                        rounds, n_extra, n_ext, mdl.n_cuts, log)
    cb(f"Done: {status}, paid hours "
       f"{res.multi.combined_phase1.total_worker_intervals / INTERVALS_PER_HOUR:,.0f}, "
       f"{len(tasks) - len(unassigned):,}/{len(tasks):,} tasks staffed, {elapsed:.1f}s")
    return res


def _break_slot(lo: int, hi: int, bl: int, busy: List[Tuple[int, int]]) -> Optional[int]:
    """First break start in [lo, hi-bl] that touches no busy interval."""
    if bl <= 0:
        return lo
    t = lo
    for a, b in sorted(busy):
        if b <= t:
            continue
        if a >= t + bl:
            break
        t = max(t, b)
    return t if t + bl <= hi else None


def _patch(tasks, starts, workers, assigned, unassigned, counts, shifts, win,
           mdl, p: TaskParams, cb):
    """Staff the task places the shift plan could not.

    For each open place the cheaper of
      * an extra worker already added here with free time (cost 0),
      * lengthening one worker's shift to a longer allowed shift that holds
        the task (cost = the extra hours), or
      * a new worker on the shortest allowed shift that holds it
    is taken.  The rest of the plan stays as it is.  Shift-structure limits
    (entries/exits, headcount) are not re-checked for these changes.
    Returns (n_changed_workers, still_open).
    """
    tmap = {t.tid: t for t in tasks}
    room = mdl.room
    ps_a, pe_a = win[:, 0], win[:, 1]
    dur_a = np.array([s.duration_intervals for s in shifts])
    active = {si for o in counts for si, v in counts[o].items() if v}
    by_occ: Dict[str, List[Worker]] = defaultdict(list)
    for w in workers:
        by_occ[w.occupation].append(w)
    extras: List[Worker] = []
    n_ext = 0
    still: List[int] = []

    def busy_of(w: Worker) -> List[Tuple[int, int]]:
        return [occupancy(tmap[i], starts[i], p) for i in w.tasks]

    def free(w: Worker, a: int, b: int) -> bool:
        return all(b2 <= a or a2 >= b for a2, b2 in busy_of(w))

    for tid in sorted(unassigned, key=lambda i: starts[i]):
        t = tmap[tid]
        o = mdl.occ_of[t.occupation]
        a, b = occupancy(t, starts[tid], p)
        got = assigned.setdefault(tid, [])
        while len(got) < t.staff:
            best = None            # (cost, kind, worker, shift idx, break start)
            for w in extras:       # free capacity on workers added before
                if w.occupation == t.occupation and w.wid not in got \
                        and a >= w.prod_start and b <= w.prod_end and free(w, a, b):
                    busy = busy_of(w) + [(a, b)]
                    lo, hi, bl = (int(v) for v in room[w.shift.idx])
                    bs = _break_slot(lo, hi, bl, busy)
                    if bs is not None:
                        best = (0, "use", w, w.shift.idx, bs)
                        break
            if best is None:       # lengthen someone's shift
                for w in by_occ[t.occupation]:
                    if w.wid in got or w.prod_end + 2 * INTERVALS_PER_DAY < a \
                            or w.prod_start > b + 2 * INTERVALS_PER_DAY:
                        continue
                    if abs(w.prod_start - a) > 12 * INTERVALS_PER_HOUR or not free(w, a, b):
                        continue
                    busy = busy_of(w) + [(a, b)]
                    L = min(x for x, _ in busy)
                    R = max(y for _, y in busy)
                    cand = np.nonzero((ps_a >= 0) & (ps_a <= L) & (pe_a >= R))[0]
                    if len(cand) == 0:
                        continue
                    for si in cand[np.argsort(dur_a[cand], kind="stable")][:8]:
                        cost = int(dur_a[si]) - w.shift.duration_intervals
                        if best is not None and cost >= best[0]:
                            break
                        lo, hi, bl = (int(v) for v in room[si])
                        bs = _break_slot(lo, hi, bl, busy)
                        if bs is not None:
                            best = (cost, "extend", w, int(si), bs)
                            break
            idxs = [int(i) for i in _containing(win, a, b, p.circular_week, room)]
            if idxs:               # a new worker
                pool = [i for i in idxs if i in active] or idxs
                si = min(pool, key=lambda i: (shifts[i].duration_intervals, i))
                if best is None or shifts[si].duration_intervals < best[0]:
                    lo, hi, bl = (int(v) for v in room[si])
                    bs = _break_slot(lo, hi, bl, [(a, b)])
                    if bs is not None:
                        best = (shifts[si].duration_intervals, "new", None, si, bs)
            if best is None:
                cb(f"{t.name} {t.flight} fits in no allowed shift — left unstaffed")
                break
            _, kind, w, si, bs = best
            if kind == "new":
                ps, pe, _, _ = (int(v) for v in win[si])
                bl = int(room[si][2])
                w = Worker(f"{t.occupation[:3].upper()}-{DAY_ABBR3[shifts[si].day]}-X{len(extras) + 1:02d}",
                           t.occupation, shifts[si], ps, pe, bs, bs + bl, extra=True)
                counts[o][si] = counts[o].get(si, 0) + 1
                extras.append(w)
                workers.append(w)
                by_occ[t.occupation].append(w)
            elif kind == "extend":
                old = w.shift.idx
                counts[o][old] -= 1
                counts[o][si] = counts[o].get(si, 0) + 1
                if not w.extended_from:
                    w.extended_from = w.code
                    n_ext += 1
                w.shift = shifts[si]
                w.prod_start, w.prod_end = int(win[si][0]), int(win[si][1])
                bl = int(room[si][2])
                w.break_start, w.break_end = bs, bs + bl
            else:
                bl = w.break_end - w.break_start
                w.break_start, w.break_end = bs, bs + bl
            w.tasks.append(tid)
            w.tasks.sort(key=lambda i: starts[i])
            got.append(w.wid)
        if len(got) < t.staff:
            still.append(tid)
    for o in counts:
        for si in [k for k, v in counts[o].items() if v <= 0]:
            del counts[o][si]
    return len(extras), n_ext, still


def _empty_result(tasks, occ_names, p, status, elapsed, log) -> TaskResult:
    z = np.zeros(TOTAL_INTERVALS, dtype=int)
    empty = PhaseOneResult([], [], 0, z, status, elapsed)
    occs = [OccupationResult(n, z.copy(), PhaseOneResult([], [], 0, z.copy(), status, elapsed))
            for n in occ_names]
    multi = MultiCurveResult(occs, empty, z.copy(), p, None)
    return TaskResult(multi, tasks, {}, {}, [], [t.tid for t in tasks], {},
                      0, 0, 0, 0, log)


def _build_result(tasks, all_tasks, occ_names, p, shifts, counts, starts,
                  assigned, unassigned, workers, win, status, elapsed, rounds,
                  n_extra, n_ext, n_cuts, log) -> TaskResult:
    occs: List[OccupationResult] = []
    comb_counts = [0] * len(shifts)
    comb_cov = np.zeros(TOTAL_INTERVALS, dtype=int)
    comb_load = np.zeros(TOTAL_INTERVALS, dtype=int)
    productive: Dict[str, np.ndarray] = {}
    for o, name in enumerate(occ_names):
        load = np.zeros(TOTAL_INTERVALS, dtype=int)
        for t in tasks:
            if t.occupation != name:
                continue
            a, b = occupancy(t, starts[t.tid], p)
            for k in _occ_intervals(a, b, p.circular_week):
                load[k] += t.staff
        cov = np.zeros(TOTAL_INTERVALS, dtype=int)
        prod = np.zeros(TOTAL_INTERVALS, dtype=int)
        for s in shifts:
            c = counts[o].get(s.idx, 0)
            if c <= 0:
                continue
            for k in range(s.global_start, s.global_end):
                cov[k % TOTAL_INTERVALS] += c
            comb_counts[s.idx] += c
        for w in workers:                 # actual breaks (may have moved)
            if w.occupation != name:
                continue
            for k in range(w.prod_start, w.prod_end):
                if not (w.break_start <= k < w.break_end):
                    prod[k % TOTAL_INTERVALS] += 1
        cl = [counts[o].get(s.idx, 0) for s in shifts]
        wi = sum(c * s.duration_intervals for s, c in zip(shifts, cl))
        occs.append(OccupationResult(name, load, PhaseOneResult(
            shifts, cl, wi, cov, status, elapsed)))
        productive[name] = prod
        comb_cov += cov
        comb_load += load
    comb_wi = sum(c * s.duration_intervals for s, c in zip(shifts, comb_counts))
    combined = PhaseOneResult(shifts, comb_counts, comb_wi, comb_cov, status, elapsed)
    multi = MultiCurveResult(occs, combined, comb_load, p, None)
    start_min = {tid: st * STEP for tid, st in starts.items()}
    return TaskResult(multi, all_tasks, start_min, assigned, workers, unassigned,
                      productive, rounds, n_cuts, n_extra, n_ext, log)


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════
def _fmt_iv(k: int) -> str:
    d, r = divmod(int(k) % TOTAL_INTERVALS, INTERVALS_PER_DAY)
    h, m = divmod(r * STEP, 60)
    return f"{DAY_ABBR3[d]} {h:02d}:{m:02d}"


def _fmt_min(mn: int) -> str:
    d, r = divmod(int(mn) % WEEK_MIN, 1440)
    return f"{DAY_ABBR3[d]} {r // 60:02d}:{r % 60:02d}"


def task_hours(res: TaskResult) -> float:
    return sum(t.duration * t.staff for t in res.tasks) / 60


def utilisation(res: TaskResult) -> float:
    paid = res.multi.combined_phase1.total_worker_intervals / INTERVALS_PER_HOUR
    return 100.0 * task_hours(res) / paid if paid else 0.0


def assignment_dataframe(res: TaskResult) -> pd.DataFrame:
    rows = []
    for t in sorted(res.tasks, key=lambda t: (res.start.get(t.tid, 0), t.tid)):
        st = res.start.get(t.tid)
        ws = res.assigned.get(t.tid, [])
        rows.append({
            "Task": t.name, "Occupation": t.occupation, "Flight": t.flight,
            "Location": t.location,
            "Window": f"{_fmt_min(t.earliest)}–{_fmt_min(t.latest_end)[4:]}",
            "Planned start": _fmt_min(st) if st is not None else "",
            "Planned end": _fmt_min(st + t.duration)[4:] if st is not None else "",
            "Duration (min)": t.duration, "Staff": t.staff,
            "Assigned": len(ws),
            "Workers": ", ".join(ws),
            "Status": "✅" if len(ws) >= t.staff else "❌ short",
        })
    return pd.DataFrame(rows)


def workers_dataframe(res: TaskResult) -> pd.DataFrame:
    tmap = {t.tid: t for t in res.tasks}
    rows = []
    for w in res.workers:
        busy = sum(tmap[i].duration for i in w.tasks)
        paid = w.shift.duration_intervals * STEP
        rows.append({
            "Worker": w.wid, "Occupation": w.occupation,
            "Day": DAY_NAMES[w.shift.day], "Shift": w.code,
            "Paid (h)": paid / 60,
            "Break": (f"{_fmt_iv(w.break_start)[4:]}–{_fmt_iv(w.break_end)[4:]}"
                      if w.break_end > w.break_start else ""),
            "Tasks": len(w.tasks), "Task hours": round(busy / 60, 2),
            "Utilisation %": round(100 * busy / paid, 1) if paid else 0.0,
            "Repair": ("added" if w.extra else
                       f"lengthened from {w.extended_from}" if w.extended_from else ""),
        })
    return pd.DataFrame(rows)


def build_task_report_xlsx(res: TaskResult) -> bytes:
    return build_weekly_report_xlsx(res.multi, extra_sheets={
        "Task Assignment": assignment_dataframe(res),
        "Workers": workers_dataframe(res),
    })


def shift_types(res: TaskResult) -> pd.DataFrame:
    parts = [shift_type_summary(o.phase1, o.name) for o in res.multi.occupations]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
