"""
ShiftCover – Weekly Shift-Covering Optimiser (OR-Tools CP-SAT)
==============================================================
Two-phase solver:
  Phase 1 – Set Covering : find the minimum-cost collection of candidate shifts
             that satisfies every 5-minute demand interval.
  Phase 2 – Worker Assignment : bin-pack the Phase-1 shifts into individual
             worker schedules respecting weekly-hour windows.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

# ── Constants ────────────────────────────────────────────────────────────────
INTERVALS_PER_HOUR = 12          # 60 / 5
INTERVALS_PER_DAY = 288          # 24 * 12
TOTAL_INTERVALS = 2016           # 7 * 288
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class CandidateShift:
    """One possible shift (day, start, duration)."""
    idx: int
    day: int                     # 0-6
    start_interval: int          # 0-287 within the day
    duration_intervals: int      # 36-144  (3 h – 12 h)

    @property
    def duration_hours(self) -> float:
        return self.duration_intervals / INTERVALS_PER_HOUR

    @property
    def global_start(self) -> int:
        return self.day * INTERVALS_PER_DAY + self.start_interval

    @property
    def global_end(self) -> int:
        return self.global_start + self.duration_intervals

    @property
    def start_time_str(self) -> str:
        h, m = divmod(self.start_interval * 5, 60)
        return f"{h:02d}:{m:02d}"

    @property
    def end_time_str(self) -> str:
        total_min = (self.start_interval + self.duration_intervals) * 5
        h, m = divmod(total_min, 60)
        # handle overnight display
        if h >= 24:
            h -= 24
        return f"{h:02d}:{m:02d}"

    def covers(self, global_interval: int) -> bool:
        return self.global_start <= global_interval < self.global_end


@dataclass
class SolverParams:
    """All tuneable knobs."""
    min_shift_hours: float = 3.0
    max_shift_hours: float = 12.0
    shift_start_granularity_min: int = 15      # candidate start every N min
    shift_duration_step_min: int = 30           # duration step
    min_weekly_hours: float = 40.0
    max_weekly_hours: float = 50.0
    max_shifts_per_day_per_worker: int = 1
    solver_time_limit_sec: int = 120
    # whether to allow overnight shifts within a day (ending after 24:00)
    allow_overnight: bool = False


@dataclass
class PhaseOneResult:
    shifts: List[CandidateShift]
    counts: List[int]             # x[s] for each shift
    total_worker_intervals: int
    coverage: np.ndarray          # actual coverage per interval
    status: str
    elapsed_sec: float


@dataclass
class PhaseTwoResult:
    worker_schedules: List[List[CandidateShift]]   # list per worker
    worker_hours: List[float]
    num_workers: int
    status: str
    elapsed_sec: float


@dataclass
class FullResult:
    phase1: PhaseOneResult
    phase2: PhaseTwoResult
    demand: np.ndarray


# ── Candidate shift generation ───────────────────────────────────────────────
def generate_candidate_shifts(params: SolverParams) -> List[CandidateShift]:
    """Build the universe of candidate shifts."""
    start_step = max(1, params.shift_start_granularity_min // 5)
    dur_step = max(1, params.shift_duration_step_min // 5)
    min_dur = int(params.min_shift_hours * INTERVALS_PER_HOUR)
    max_dur = int(params.max_shift_hours * INTERVALS_PER_HOUR)

    shifts: List[CandidateShift] = []
    idx = 0
    for day in range(7):
        for start in range(0, INTERVALS_PER_DAY, start_step):
            for dur in range(min_dur, max_dur + 1, dur_step):
                end_interval = start + dur
                # If overnight not allowed, shift must end within the day
                if not params.allow_overnight and end_interval > INTERVALS_PER_DAY:
                    continue
                # Even with overnight, cap at day boundary for simplicity
                if end_interval > INTERVALS_PER_DAY:
                    continue
                shifts.append(CandidateShift(idx, day, start, dur))
                idx += 1
    return shifts


# ── Pre-compute coverage matrix (sparse) ─────────────────────────────────────
def build_coverage_map(
    shifts: List[CandidateShift],
) -> Dict[int, List[int]]:
    """Return {global_interval: [shift indices covering it]}."""
    cov: Dict[int, List[int]] = {t: [] for t in range(TOTAL_INTERVALS)}
    for s in shifts:
        for t in range(s.global_start, s.global_end):
            if 0 <= t < TOTAL_INTERVALS:
                cov[t].append(s.idx)
    return cov


# ── Phase 1 – Set Covering ───────────────────────────────────────────────────
def solve_phase1(
    demand: np.ndarray,
    params: SolverParams,
    callback=None,
) -> PhaseOneResult:
    """
    Minimise total worker-intervals (≈ total labour cost) such that
    every 5-min interval is covered by at least demand[t] workers.
    """
    assert demand.shape == (TOTAL_INTERVALS,), \
        f"Demand must be length {TOTAL_INTERVALS}, got {demand.shape}"

    t0 = time.time()

    # ---- generate candidates ----
    shifts = generate_candidate_shifts(params)
    if callback:
        callback(f"Generated {len(shifts):,} candidate shifts")

    # ---- coverage map ----
    cov = build_coverage_map(shifts)
    if callback:
        callback("Built coverage map")

    # ---- identify intervals that actually need coverage ----
    active_intervals = [t for t in range(TOTAL_INTERVALS) if demand[t] > 0]
    max_demand = int(demand.max())

    # ---- model ----
    model = cp_model.CpModel()

    # decision variables: x[s] = how many workers on shift s
    x = {}
    for s in shifts:
        x[s.idx] = model.NewIntVar(0, max_demand, f"x_{s.idx}")

    # coverage constraints
    for t in active_intervals:
        covering = cov[t]
        if not covering:
            # infeasible interval – no candidate shift covers it
            continue
        model.Add(
            sum(x[s_idx] for s_idx in covering) >= int(demand[t])
        )

    # objective: minimise total worker-intervals (proportional to labour hours)
    model.Minimize(
        sum(x[s.idx] * s.duration_intervals for s in shifts)
    )

    if callback:
        callback("Model built – starting Phase 1 solve …")

    # ---- solve ----
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = params.solver_time_limit_sec
    solver.parameters.num_workers = 8     # parallelism
    solver.parameters.log_search_progress = False

    status_code = solver.Solve(model)
    status_str = solver.StatusName(status_code)

    elapsed = time.time() - t0

    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return PhaseOneResult(shifts, [], 0, np.zeros(TOTAL_INTERVALS),
                              status_str, elapsed)

    counts = [solver.Value(x[s.idx]) for s in shifts]

    # compute actual coverage
    coverage = np.zeros(TOTAL_INTERVALS, dtype=int)
    for s, cnt in zip(shifts, counts):
        if cnt > 0:
            coverage[s.global_start:s.global_end] += cnt

    total_wi = sum(c * s.duration_intervals for s, c in zip(shifts, counts))

    if callback:
        callback(f"Phase 1 done: status={status_str}, "
                 f"total worker-hours={total_wi / INTERVALS_PER_HOUR:.1f}")

    return PhaseOneResult(shifts, counts, total_wi, coverage,
                          status_str, elapsed)


# ── Phase 2 – Worker Assignment (Bin Packing) ────────────────────────────────
def solve_phase2(
    p1: PhaseOneResult,
    params: SolverParams,
    callback=None,
) -> PhaseTwoResult:
    """
    Assign the shift instances from Phase 1 to individual workers so that
    each worker's total weekly hours is in [min_weekly, max_weekly] and
    each worker has at most `max_shifts_per_day_per_worker` shifts per day.
    Minimise total number of workers.
    """
    t0 = time.time()

    # ---- flatten shift instances ----
    instances: List[CandidateShift] = []
    for s, cnt in zip(p1.shifts, p1.counts):
        for _ in range(cnt):
            instances.append(s)

    if not instances:
        return PhaseTwoResult([], [], 0, "NO_SHIFTS", 0.0)

    n_instances = len(instances)
    if callback:
        callback(f"Phase 2: assigning {n_instances} shift instances to workers")

    # upper bound on workers needed
    min_dur = min(s.duration_intervals for s in instances)
    total_intervals = sum(s.duration_intervals for s in instances)
    min_slots_per_worker = int(params.min_weekly_hours * INTERVALS_PER_HOUR)
    max_workers = min(
        n_instances,
        int(np.ceil(total_intervals / min_slots_per_worker)) + 5
    )
    max_workers = max(max_workers, 1)

    # ---- model ----
    model = cp_model.CpModel()

    # y[j, w] = 1 if instance j assigned to worker w
    y = {}
    for j in range(n_instances):
        for w in range(max_workers):
            y[j, w] = model.NewBoolVar(f"y_{j}_{w}")

    # worker_used[w] = 1 if any shift assigned to worker w
    worker_used = {}
    for w in range(max_workers):
        worker_used[w] = model.NewBoolVar(f"used_{w}")

    # each instance assigned to exactly one worker
    for j in range(n_instances):
        model.AddExactlyOne(y[j, w] for w in range(max_workers))

    # link worker_used
    for w in range(max_workers):
        model.AddMaxEquality(worker_used[w],
                             [y[j, w] for j in range(n_instances)])

    # weekly hours per worker
    min_intervals = int(params.min_weekly_hours * INTERVALS_PER_HOUR)
    max_intervals = int(params.max_weekly_hours * INTERVALS_PER_HOUR)

    for w in range(max_workers):
        total_w = sum(
            y[j, w] * instances[j].duration_intervals
            for j in range(n_instances)
        )
        # if worker is used, hours must be in range
        model.Add(total_w >= min_intervals).OnlyEnforceIf(worker_used[w])
        model.Add(total_w <= max_intervals).OnlyEnforceIf(worker_used[w])
        model.Add(total_w == 0).OnlyEnforceIf(worker_used[w].Not())

    # max shifts per day per worker  &  no overlapping shifts
    for w in range(max_workers):
        for day in range(7):
            day_instances = [j for j in range(n_instances)
                             if instances[j].day == day]
            if not day_instances:
                continue
            # limit number of shifts
            model.Add(
                sum(y[j, w] for j in day_instances)
                <= params.max_shifts_per_day_per_worker
            )
            # no overlap (for pairs on the same day)
            if params.max_shifts_per_day_per_worker == 1:
                continue  # only one shift → no overlap possible
            for j1, j2 in itertools.combinations(day_instances, 2):
                s1, s2 = instances[j1], instances[j2]
                # if they overlap, they can't both be assigned to same worker
                if not (s1.global_end <= s2.global_start or
                        s2.global_end <= s1.global_start):
                    model.Add(y[j1, w] + y[j2, w] <= 1)

    # symmetry breaking: prefer lower-index workers
    for w in range(1, max_workers):
        model.Add(worker_used[w] <= worker_used[w - 1])

    # objective: minimise workers
    model.Minimize(sum(worker_used[w] for w in range(max_workers)))

    if callback:
        callback(f"Phase 2 model built (max_workers cap={max_workers}) – solving …")

    # ---- solve ----
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = params.solver_time_limit_sec
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = False

    status_code = solver.Solve(model)
    status_str = solver.StatusName(status_code)
    elapsed = time.time() - t0

    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return PhaseTwoResult([], [], 0, status_str, elapsed)

    # ---- extract schedules ----
    schedules: List[List[CandidateShift]] = []
    hours_list: List[float] = []
    for w in range(max_workers):
        if solver.Value(worker_used[w]) == 0:
            continue
        w_shifts = [instances[j] for j in range(n_instances)
                    if solver.Value(y[j, w]) == 1]
        w_shifts.sort(key=lambda s: s.global_start)
        schedules.append(w_shifts)
        hours_list.append(
            sum(s.duration_intervals for s in w_shifts) / INTERVALS_PER_HOUR
        )

    if callback:
        callback(f"Phase 2 done: {len(schedules)} workers, "
                 f"status={status_str}")

    return PhaseTwoResult(schedules, hours_list, len(schedules),
                          status_str, elapsed)


# ── Convenience: run both phases ─────────────────────────────────────────────
def solve(
    demand: np.ndarray,
    params: SolverParams | None = None,
    callback=None,
) -> FullResult:
    if params is None:
        params = SolverParams()
    p1 = solve_phase1(demand, params, callback)
    if p1.status not in ("OPTIMAL", "FEASIBLE"):
        p2 = PhaseTwoResult([], [], 0, "SKIPPED", 0.0)
    else:
        p2 = solve_phase2(p1, params, callback)
    return FullResult(p1, p2, demand)


# ── Result → DataFrames ─────────────────────────────────────────────────────
def shifts_to_dataframe(p1: PhaseOneResult) -> pd.DataFrame:
    """Active shifts from Phase 1 as a tidy DataFrame."""
    rows = []
    for s, cnt in zip(p1.shifts, p1.counts):
        if cnt > 0:
            rows.append({
                "Day": DAY_NAMES[s.day],
                "DayNum": s.day,
                "Start": s.start_time_str,
                "End": s.end_time_str,
                "DurationHrs": s.duration_hours,
                "Workers": cnt,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["DayNum", "Start"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def schedules_to_dataframe(p2: PhaseTwoResult) -> pd.DataFrame:
    """Per-worker schedule from Phase 2."""
    rows = []
    for w_idx, (schedule, hrs) in enumerate(
            zip(p2.worker_schedules, p2.worker_hours)):
        for s in schedule:
            rows.append({
                "Worker": w_idx + 1,
                "Day": DAY_NAMES[s.day],
                "DayNum": s.day,
                "Start": s.start_time_str,
                "End": s.end_time_str,
                "DurationHrs": s.duration_hours,
                "WeeklyHrs": hrs,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["Worker", "DayNum", "Start"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def coverage_dataframe(result: FullResult) -> pd.DataFrame:
    """Demand vs. coverage for charting."""
    times = []
    for t in range(TOTAL_INTERVALS):
        day = t // INTERVALS_PER_DAY
        intra = t % INTERVALS_PER_DAY
        h, m = divmod(intra * 5, 60)
        times.append(f"{DAY_NAMES[day]} {h:02d}:{m:02d}")

    return pd.DataFrame({
        "Interval": range(TOTAL_INTERVALS),
        "Time": times,
        "Demand": result.demand.astype(int),
        "Coverage": result.phase1.coverage.astype(int),
    })
