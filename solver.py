"""
ShiftCover – Weekly Shift-Covering Optimiser (OR-Tools CP-SAT)
==============================================================
Supports 1-3 occupation curves with SHARED shift structures.

Two-phase solver
  Phase 1 – Multi-curve Set Covering via CP-SAT.
             Shift activation (z[s]) is shared across occupations so every
             occupation's workers enter/exit at the same boundaries
             (same shuttle schedule).
  Phase 2 – Per-occupation greedy worker assignment respecting weekly-hour
             windows and minimum-rest constraints.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import io as _io
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

# ── Constants ────────────────────────────────────────────────────────────────
INTERVALS_PER_HOUR = 12          # 60 / 5
INTERVALS_PER_DAY  = 288         # 24 * 12
TOTAL_INTERVALS    = 2016        # 7 * 288
DAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]
DAY_ABBR = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
OCC_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # blue, orange, green


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class CandidateShift:
    idx: int
    day: int                     # 0-6
    start_interval: int          # 0-287 within the day
    duration_intervals: int      # e.g. 36-144

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
        if h >= 24:
            h -= 24
        return f"{h:02d}:{m:02d}"

    @property
    def shift_code(self) -> str:
        sh, sm = divmod(self.start_interval * 5, 60)
        total_end = (self.start_interval + self.duration_intervals) * 5
        eh, em = divmod(total_end, 60)
        if eh >= 24:
            eh -= 24
        return f"{sh:02d}{sm:02d}-{eh:02d}{em:02d}"

    def covers(self, global_interval: int) -> bool:
        if self.global_end <= TOTAL_INTERVALS:
            return self.global_start <= global_interval < self.global_end
        # Wrapping shift (Sunday → Monday)
        return (global_interval >= self.global_start or
                global_interval < self.global_end - TOTAL_INTERVALS)


@dataclass
class SolverParams:
    min_shift_hours: float        = 3.0
    max_shift_hours: float        = 12.0
    shift_start_granularity_min: int = 15
    shift_duration_step_min: int  = 30
    min_weekly_hours: float       = 40.0
    max_weekly_hours: float       = 50.0
    min_rest_hours: float         = 12.0
    max_unique_shifts: int        = 0          # 0 = unlimited
    transition_penalty: int       = 50
    solver_time_limit_sec: int    = 120
    # Per-day limits on distinct entry/exit times (0 = unlimited)
    # Each is a list of 7 ints (Mon–Sun). If None → no constraint.
    max_entries_per_day: Optional[List[int]] = None
    max_exits_per_day: Optional[List[int]] = None
    # Per-day max simultaneous workers (0 = unlimited). List of 7 ints.
    max_headcount_per_day: Optional[List[int]] = None
    # Exclude shifts that fall entirely within the 20:00–06:00 window
    exclude_night_shifts: bool = False
    # Circular week: Sunday shifts can wrap into Monday
    circular_week: bool = False
    # Force include/exclude specific shift codes (e.g. ["0630-1430"])
    force_include_shifts: Optional[List[str]] = None
    force_exclude_shifts: Optional[List[str]] = None
    # Cost-based objective (0 = disabled, falls back to FTE minimisation)
    personnel_cost_per_fte: float = 0.0   # monetary cost per FTE (worker_h / 45)
    shuttle_cost_per_trip: float  = 0.0   # monetary cost per shuttle trip
    shuttle_capacity: int         = 14    # workers per shuttle
    max_total_cost: Optional[float] = None  # hard budget cap (FTE + shuttle); None = no cap


@dataclass
class PhaseOneResult:
    shifts: List[CandidateShift]
    counts: List[int]
    total_worker_intervals: int
    coverage: np.ndarray
    status: str
    elapsed_sec: float
    estimated_cost: float = 0.0


@dataclass
class PhaseTwoResult:
    worker_schedules: List[List[CandidateShift]]
    worker_hours: List[float]
    num_workers: int
    status: str
    elapsed_sec: float


@dataclass
class OccupationResult:
    name: str
    demand: np.ndarray
    phase1: PhaseOneResult
    phase2: PhaseTwoResult


@dataclass
class MultiCurveResult:
    occupations: List[OccupationResult]
    combined_phase1: PhaseOneResult
    combined_demand: np.ndarray
    params: SolverParams


# backward-compat alias
FullResult = MultiCurveResult


# ── Night / break helpers ────────────────────────────────────────────────────
_NIGHT_START = 240   # 20:00  (within-day interval)
_NIGHT_END   = 72    # 06:00  (within-day interval)


def night_overlap_intervals(start: int, end: int) -> int:
    """Return how many 5-min intervals of [start, end) overlap 20:00-06:00.

    ``start`` is a within-day interval [0, 288); ``end`` = start + duration,
    may exceed 288 for overnight shifts.
    """
    total = 0
    # Split into current-day [start, min(end,288)) and next-day [0, end-288)
    ranges = [(start, min(end, INTERVALS_PER_DAY))]
    if end > INTERVALS_PER_DAY:
        ranges.append((0, end - INTERVALS_PER_DAY))
    for a, b in ranges:
        # Night windows: morning [0, 72) and evening [240, 288)
        total += max(0, min(b, _NIGHT_END) - max(a, 0))
        total += max(0, min(b, INTERVALS_PER_DAY) - max(a, _NIGHT_START))
    return total


def night_overlap_hours(start: int, end: int) -> float:
    return night_overlap_intervals(start, end) / INTERVALS_PER_HOUR


def break_hours(duration_hours: float) -> float:
    """Mandatory break: 30 min for 4-7.5 h shifts, 1 h for ≥8 h shifts."""
    if duration_hours >= 8.0:
        return 1.0
    if duration_hours >= 4.0:
        return 0.5
    return 0.0


def effective_hours(duration_hours: float) -> float:
    return duration_hours - break_hours(duration_hours)


def is_night_shift(start: int, dur: int) -> bool:
    """True when duration ≥ 8 h AND night overlap > half the shift duration."""
    dur_h = dur / INTERVALS_PER_HOUR
    if dur_h < 8.0:
        return False
    night_h = night_overlap_hours(start, start + dur)
    return night_h > dur_h / 2.0


# ── Candidate-shift generation ───────────────────────────────────────────────
def _shift_code_from(start: int, dur: int) -> str:
    """Compute HHMM-HHMM code from start interval and duration intervals."""
    sh, sm = divmod(start * 5, 60)
    total_end = (start + dur) * 5
    eh, em = divmod(total_end, 60)
    if eh >= 24:
        eh -= 24
    return f"{sh:02d}{sm:02d}-{eh:02d}{em:02d}"


def list_possible_shift_codes(params: SolverParams) -> List[str]:
    """Return sorted list of unique HHMM-HHMM shift codes for given params."""
    start_step = max(1, params.shift_start_granularity_min // 5)
    dur_step   = max(1, params.shift_duration_step_min // 5)
    min_dur    = int(params.min_shift_hours * INTERVALS_PER_HOUR)
    max_dur    = int(params.max_shift_hours * INTERVALS_PER_HOUR)
    codes: set = set()
    for start in range(0, INTERVALS_PER_DAY, start_step):
        for dur in range(min_dur, max_dur + 1, dur_step):
            if params.exclude_night_shifts and is_night_shift(start, dur):
                continue
            codes.add(_shift_code_from(start, dur))
    return sorted(codes)


def generate_candidate_shifts(params: SolverParams) -> List[CandidateShift]:
    start_step = max(1, params.shift_start_granularity_min // 5)
    dur_step   = max(1, params.shift_duration_step_min // 5)
    min_dur    = int(params.min_shift_hours * INTERVALS_PER_HOUR)
    max_dur    = int(params.max_shift_hours * INTERVALS_PER_HOUR)

    _excl_set = set(params.force_exclude_shifts) if params.force_exclude_shifts else set()

    shifts: List[CandidateShift] = []
    idx = 0
    for day in range(7):
        for start in range(0, INTERVALS_PER_DAY, start_step):
            for dur in range(min_dur, max_dur + 1, dur_step):
                end_interval = start + dur
                is_overnight = end_interval > INTERVALS_PER_DAY
                is_sun_wrap = (day == 6 and is_overnight)

                if is_sun_wrap and not params.circular_week:
                    continue
                if is_overnight and not is_sun_wrap:
                    if day >= 6:
                        continue  # Sunday without circular

                if params.exclude_night_shifts and is_night_shift(start, dur):
                    continue

                if (params.force_exclude_shifts
                        and _shift_code_from(start, dur)
                        in _excl_set):
                    continue

                shifts.append(CandidateShift(idx, day, start, dur))
                idx += 1
    return shifts


def build_coverage_map(
    shifts: List[CandidateShift],
) -> Dict[int, List[int]]:
    cov: Dict[int, List[int]] = {t: [] for t in range(TOTAL_INTERVALS)}
    for s in shifts:
        for t in range(s.global_start, s.global_end):
            cov[t % TOTAL_INTERVALS].append(s.idx)
    return cov


# ── Phase 1 – Multi-Curve Set Covering ───────────────────────────────────────
def solve_phase1_multi(
    demands: List[np.ndarray],
    occ_names: List[str],
    params: SolverParams,
    callback=None,
) -> Tuple[List[PhaseOneResult], PhaseOneResult]:
    """
    Solve all occupation curves simultaneously with SHARED shift activation.
    Returns (per_occ_results, combined_result).
    """
    n_occ = len(demands)
    for i, d in enumerate(demands):
        assert d.shape == (TOTAL_INTERVALS,), \
            f"Demand[{i}] must have length {TOTAL_INTERVALS}"

    t0 = time.time()

    shifts = generate_candidate_shifts(params)
    if callback:
        callback(f"Generated {len(shifts):,} candidate shifts")

    cov = build_coverage_map(shifts)
    if callback:
        callback("Built coverage map")

    model = cp_model.CpModel()

    # Per-occupation decision vars: x[occ][s] = workers of that occ on shift s
    x: Dict[int, Dict[int, object]] = {}
    for occ in range(n_occ):
        x[occ] = {}
        max_d = int(demands[occ].max()) if demands[occ].max() > 0 else 1
        for s in shifts:
            x[occ][s.idx] = model.new_int_var(0, max_d, f"x_{occ}_{s.idx}")

    # Shared activation vars:  z[s] = 1  iff  any occupation uses shift s
    z: Dict[int, object] = {}
    for s in shifts:
        z[s.idx] = model.new_bool_var(f"z_{s.idx}")
        total_on = sum(x[occ][s.idx] for occ in range(n_occ))
        model.add(total_on >= 1).only_enforce_if(z[s.idx])
        model.add(total_on == 0).only_enforce_if(z[s.idx].negated())

    # Enforce: x[occ][s] > 0 only if z[s] = 1
    for occ in range(n_occ):
        for s in shifts:
            model.add(x[occ][s.idx] == 0).only_enforce_if(z[s.idx].negated())

    # Force-include shift codes: every matching code must be active
    if params.force_include_shifts:
        incl_set = set(params.force_include_shifts)
        # Group candidate indices by shift_code
        code_to_idxs: Dict[str, List[int]] = {}
        for s in shifts:
            if s.shift_code in incl_set:
                code_to_idxs.setdefault(s.shift_code, []).append(s.idx)
        for code, idxs in code_to_idxs.items():
            # At least one candidate with this code must be active
            model.add(sum(z[i] for i in idxs) >= 1)
        if callback:
            callback(f"Force-include {len(code_to_idxs)} shift code(s)")

    # Max unique shifts (shared cardinality)
    if params.max_unique_shifts > 0:
        model.add(sum(z[s.idx] for s in shifts) <= params.max_unique_shifts)
        if callback:
            callback(f"Max unique shifts ≤ {params.max_unique_shifts}")

    # Per-day max distinct entry times
    if params.max_entries_per_day:
        # Collect distinct start_intervals per day
        day_starts: Dict[int, set] = {d: set() for d in range(7)}
        for s in shifts:
            day_starts[s.day].add(s.start_interval)
        for day in range(7):
            limit = params.max_entries_per_day[day]
            if limit <= 0:
                continue
            starts = sorted(day_starts[day])
            if len(starts) <= limit:
                continue  # already within limit, no constraint needed
            # e_start[day][si] = 1 iff any shift on this day with start_interval==si is active
            e_vars = []
            for si in starts:
                e = model.new_bool_var(f"entry_d{day}_s{si}")
                matching = [s.idx for s in shifts
                            if s.day == day and s.start_interval == si]
                model.add(sum(z[idx] for idx in matching) >= 1).only_enforce_if(e)
                model.add(sum(z[idx] for idx in matching) == 0).only_enforce_if(e.negated())
                e_vars.append(e)
            model.add(sum(e_vars) <= limit)
        if callback:
            callback(f"Max entries/day: {params.max_entries_per_day}")

    # Per-day max distinct exit times
    if params.max_exits_per_day:
        day_ends: Dict[int, set] = {d: set() for d in range(7)}
        for s in shifts:
            end_ivl = s.start_interval + s.duration_intervals
            day_ends[s.day].add(end_ivl)
        for day in range(7):
            limit = params.max_exits_per_day[day]
            if limit <= 0:
                continue
            ends = sorted(day_ends[day])
            if len(ends) <= limit:
                continue
            x_vars = []
            for ei in ends:
                ev = model.new_bool_var(f"exit_d{day}_e{ei}")
                matching = [s.idx for s in shifts
                            if s.day == day
                            and s.start_interval + s.duration_intervals == ei]
                model.add(sum(z[idx] for idx in matching) >= 1).only_enforce_if(ev)
                model.add(sum(z[idx] for idx in matching) == 0).only_enforce_if(ev.negated())
                x_vars.append(ev)
            model.add(sum(x_vars) <= limit)
        if callback:
            callback(f"Max exits/day: {params.max_exits_per_day}")

    # Per-day max simultaneous headcount
    if params.max_headcount_per_day:
        for day in range(7):
            limit = params.max_headcount_per_day[day]
            if limit <= 0:
                continue
            day_start = day * INTERVALS_PER_DAY
            day_end = day_start + INTERVALS_PER_DAY
            for t in range(day_start, day_end):
                covering = cov[t]
                if not covering:
                    continue
                # sum across ALL occupations at interval t
                total_at_t = []
                for occ in range(n_occ):
                    total_at_t.extend(x[occ][si] for si in covering)
                model.add(sum(total_at_t) <= limit)
        if callback:
            callback(f"Max headcount/day: {params.max_headcount_per_day}")

    # Per-occupation coverage constraints
    for occ in range(n_occ):
        for t in range(TOTAL_INTERVALS):
            if demands[occ][t] <= 0:
                continue
            covering = cov[t]
            if not covering:
                continue
            model.add(
                sum(x[occ][si] for si in covering) >= int(demands[occ][t])
            )
    if callback:
        callback(f"Coverage constraints added for {n_occ} occupation(s)")

    # ---- objective ----
    _cost_mode = (
        params.personnel_cost_per_fte > 0 or params.shuttle_cost_per_trip > 0
    )
    # Integer scale: 1 unit = 0.01 monetary (cent precision)
    _SCALE = 100

    # cost_terms: real monetary cost (FTE + shuttle) — used for budget constraint
    # obj_terms:  full objective = cost_terms + transition penalty
    cost_terms = []
    obj_terms  = []

    if _cost_mode:
        # FTE component: cost per worker-interval = cost_per_fte * SCALE / (45 * IPH)
        _fte_coeff = int(round(
            params.personnel_cost_per_fte * _SCALE / (45.0 * INTERVALS_PER_HOUR)
        ))
        if _fte_coeff > 0:
            for occ in range(n_occ):
                for s in shifts:
                    cost_terms.append(x[occ][s.idx] * s.duration_intervals * _fte_coeff)
        # Shuttle component: group workers by (day, entry_time) and (exit_day, exit_time)
        _shu_coeff = int(round(params.shuttle_cost_per_trip * _SCALE))
        if _shu_coeff > 0:
            _cap = max(1, params.shuttle_capacity)
            _max_workers = int(sum(int(d.max()) for d in demands)) + 1
            _max_shuttles = (_max_workers + _cap - 1) // _cap + 1
            # build entry/exit groups: (day, time_ivl) -> list of shift indices
            _entry_grp: Dict[tuple, List[int]] = defaultdict(list)
            _exit_grp:  Dict[tuple, List[int]] = defaultdict(list)
            for s in shifts:
                _entry_grp[(s.day, s.start_interval)].append(s.idx)
                _exit_ivl = s.start_interval + s.duration_intervals
                _exit_day = s.day
                if _exit_ivl >= INTERVALS_PER_DAY:
                    _exit_ivl -= INTERVALS_PER_DAY
                    _exit_day = (_exit_day + 1) % 7
                _exit_grp[(_exit_day, _exit_ivl)].append(s.idx)
            for (d, t), idxs in _entry_grp.items():
                workers = sum(
                    x[occ][i] for occ in range(n_occ) for i in idxs)
                n_sh = model.new_int_var(0, _max_shuttles, f"she_{d}_{t}")
                model.add(n_sh * _cap >= workers)
                cost_terms.append(n_sh * _shu_coeff)
            for (d, t), idxs in _exit_grp.items():
                workers = sum(
                    x[occ][i] for occ in range(n_occ) for i in idxs)
                n_sh = model.new_int_var(0, _max_shuttles, f"shx_{d}_{t}")
                model.add(n_sh * _cap >= workers)
                cost_terms.append(n_sh * _shu_coeff)
        # Hard budget cap: FTE + shuttle cost <= max_total_cost
        # Use ceil so integer rounding never makes the constraint tighter than intended.
        if params.max_total_cost and params.max_total_cost > 0 and cost_terms:
            import math as _math
            _budget_scaled = _math.ceil(params.max_total_cost * _SCALE)
            model.add(sum(cost_terms) <= _budget_scaled)
            if callback:
                callback(f"Budget cap: {params.max_total_cost} (scaled {_budget_scaled})")
        obj_terms = list(cost_terms)
        if callback:
            callback(
                f"Cost mode: FTE @{params.personnel_cost_per_fte} "
                f"shuttle @{params.shuttle_cost_per_trip} cap={params.shuttle_capacity}"
            )
    else:
        # Default: minimise total worker-intervals (proportional to FTE)
        for occ in range(n_occ):
            for s in shifts:
                obj_terms.append(x[occ][s.idx] * s.duration_intervals)

    tp = params.transition_penalty
    if tp > 0:
        _tp_scaled = tp * _SCALE if _cost_mode else tp
        for s in shifts:
            obj_terms.append(z[s.idx] * _tp_scaled * 2)
        if callback:
            callback(f"Transition penalty {tp} on shared shift activation")

    model.minimize(sum(obj_terms))
    if callback:
        callback("Model built – starting Phase 1 solve …")

    # ---- solve ----
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = params.solver_time_limit_sec
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = False

    status_code = solver.solve(model)
    status_str  = solver.status_name(status_code)
    elapsed     = time.time() - t0

    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        empty = PhaseOneResult(shifts, [], 0,
                               np.zeros(TOTAL_INTERVALS, dtype=int),
                               status_str, elapsed)
        return [empty] * n_occ, empty

    # ---- extract per-occupation results ----
    per_occ: List[PhaseOneResult] = []
    combined_counts   = [0] * len(shifts)
    combined_coverage = np.zeros(TOTAL_INTERVALS, dtype=int)

    for occ in range(n_occ):
        counts   = [solver.value(x[occ][s.idx]) for s in shifts]
        coverage = np.zeros(TOTAL_INTERVALS, dtype=int)
        for s, cnt in zip(shifts, counts):
            if cnt > 0:
                for t in range(s.global_start, s.global_end):
                    coverage[t % TOTAL_INTERVALS] += cnt
        total_wi = sum(c * s.duration_intervals
                       for s, c in zip(shifts, counts))
        per_occ.append(PhaseOneResult(
            shifts, counts, total_wi, coverage, status_str, elapsed))
        for i, c in enumerate(counts):
            combined_counts[i] += c
        combined_coverage += coverage

    combined_wi = sum(c * s.duration_intervals
                      for s, c in zip(shifts, combined_counts))

    # Compute estimated monetary cost from solution values
    _est_cost = 0.0
    if _cost_mode:
        total_wh = combined_wi / INTERVALS_PER_HOUR
        _est_cost += params.personnel_cost_per_fte * (total_wh / 45.0)
        if params.shuttle_cost_per_trip > 0:
            _cap = max(1, params.shuttle_capacity)
            # count workers entering/exiting per (day, slot) from combined counts
            _entry_workers: Dict[tuple, int] = defaultdict(int)
            _exit_workers:  Dict[tuple, int] = defaultdict(int)
            for s, cnt in zip(shifts, combined_counts):
                if cnt == 0:
                    continue
                _entry_workers[(s.day, s.start_interval)] += cnt
                _exit_ivl = s.start_interval + s.duration_intervals
                _exit_day = s.day
                if _exit_ivl >= INTERVALS_PER_DAY:
                    _exit_ivl -= INTERVALS_PER_DAY
                    _exit_day = (_exit_day + 1) % 7
                _exit_workers[(_exit_day, _exit_ivl)] += cnt
            import math
            shuttle_trips = sum(
                math.ceil(w / _cap)
                for w in list(_entry_workers.values()) + list(_exit_workers.values())
            )
            _est_cost += shuttle_trips * params.shuttle_cost_per_trip

    combined_p1 = PhaseOneResult(
        shifts, combined_counts, combined_wi, combined_coverage,
        status_str, elapsed, estimated_cost=_est_cost)

    if callback:
        for occ in range(n_occ):
            wh = per_occ[occ].total_worker_intervals / INTERVALS_PER_HOUR
            callback(f"  {occ_names[occ]}: {wh:.0f} worker-hours")
        n_active = sum(1 for c in combined_counts if c > 0)
        msg = (f"Phase 1 done: status={status_str}, "
               f"total worker-hours={combined_wi/INTERVALS_PER_HOUR:.0f}, "
               f"unique shifts={n_active}")
        if _cost_mode:
            msg += f", estimated cost={_est_cost:,.0f}"
        callback(msg)

    return per_occ, combined_p1


# ── Phase 2 – Worker Assignment (greedy with rest constraint) ────────────────
def _shift_ranges(s: CandidateShift):
    """Return list of (start, end) interval ranges, handling week wrap."""
    if s.global_end <= TOTAL_INTERVALS:
        return [(s.global_start, s.global_end)]
    return [(s.global_start, TOTAL_INTERVALS),
            (0, s.global_end - TOTAL_INTERVALS)]


def _can_assign(
    worker_shifts: List[CandidateShift],
    shift: CandidateShift,
    params: SolverParams,
) -> bool:
    rest_ivl = int(params.min_rest_hours * INTERVALS_PER_HOUR)
    max_ivl  = int(params.max_weekly_hours * INTERVALS_PER_HOUR)

    current = sum(s.duration_intervals for s in worker_shifts)
    if current + shift.duration_intervals > max_ivl:
        return False

    # one shift per calendar day (based on start day only)
    for ex in worker_shifts:
        if ex.day == shift.day:
            return False

    s_ranges = _shift_ranges(shift)

    for ex in worker_shifts:
        ex_ranges = _shift_ranges(ex)

        # overlap check (compare all range pairs)
        overlap = False
        for (a1, a2) in s_ranges:
            for (b1, b2) in ex_ranges:
                if not (a2 <= b1 or b2 <= a1):
                    overlap = True
                    break
            if overlap:
                break
        if overlap:
            return False

        # rest gap check
        has_wrap = (shift.global_end > TOTAL_INTERVALS or
                    ex.global_end > TOTAL_INTERVALS)
        if has_wrap:
            # circular gap: effective ends wrapped
            e_ex = (ex.global_end % TOTAL_INTERVALS
                    if ex.global_end > TOTAL_INTERVALS
                    else ex.global_end)
            e_sh = (shift.global_end % TOTAL_INTERVALS
                    if shift.global_end > TOTAL_INTERVALS
                    else shift.global_end)
            gap = min((shift.global_start - e_ex) % TOTAL_INTERVALS,
                      (ex.global_start - e_sh) % TOTAL_INTERVALS)
        else:
            if ex.global_end <= shift.global_start:
                gap = shift.global_start - ex.global_end
            else:
                gap = ex.global_start - shift.global_end
        if gap < rest_ivl:
            return False
    return True


def solve_phase2(
    p1: PhaseOneResult,
    params: SolverParams,
    callback=None,
) -> PhaseTwoResult:
    t0 = time.time()

    instances: List[CandidateShift] = []
    for s, cnt in zip(p1.shifts, p1.counts):
        for _ in range(cnt):
            instances.append(s)

    if not instances:
        return PhaseTwoResult([], [], 0, "NO_SHIFTS", 0.0)

    instances.sort(key=lambda s: (s.global_start, s.duration_intervals))

    workers: List[List[CandidateShift]] = []
    worker_totals: List[int] = []
    max_ivl = int(params.max_weekly_hours * INTERVALS_PER_HOUR)
    min_ivl = int(params.min_weekly_hours * INTERVALS_PER_HOUR)

    for j, shift in enumerate(instances):
        best_w    = -1
        best_score = None

        for w_idx, (w_shifts, w_total) in enumerate(
                zip(workers, worker_totals)):
            new_total = w_total + shift.duration_intervals
            if new_total > max_ivl:
                continue
            if not _can_assign(w_shifts, shift, params):
                continue
            if w_total < min_ivl:          # tier 0 – fill under-min first
                tier, sec = 0, -w_total
            else:                          # tier 1 – most remaining capacity
                tier, sec = 1, -(max_ivl - new_total)
            score = (tier, sec)
            if best_score is None or score < best_score:
                best_w, best_score = w_idx, score

        if best_w >= 0:
            workers[best_w].append(shift)
            worker_totals[best_w] += shift.duration_intervals
        else:
            workers.append([shift])
            worker_totals.append(shift.duration_intervals)

        if callback and (j + 1) % 200 == 0:
            callback(f"  … {j+1}/{len(instances)} shifts, "
                     f"{len(workers)} workers")

    under_min = 0
    schedules: List[List[CandidateShift]] = []
    hours_list: List[float] = []
    for ws, wt in zip(workers, worker_totals):
        ws.sort(key=lambda s: s.global_start)
        schedules.append(ws)
        hrs = wt / INTERVALS_PER_HOUR
        hours_list.append(hrs)
        if wt < min_ivl:
            under_min += 1

    status = "FEASIBLE"
    if under_min > 0:
        status = f"FEASIBLE ({under_min} below min hours)"

    elapsed = time.time() - t0
    if callback:
        callback(f"  Phase 2: {len(schedules)} workers, "
                 f"status={status}, {elapsed:.1f}s")

    return PhaseTwoResult(schedules, hours_list, len(schedules),
                          status, elapsed)


# ── Convenience entry points ─────────────────────────────────────────────────
def solve_multi(
    demands: List[np.ndarray],
    occ_names: List[str],
    params: Optional[SolverParams] = None,
    callback=None,
) -> MultiCurveResult:
    if params is None:
        params = SolverParams()

    per_occ_p1, combined_p1 = solve_phase1_multi(
        demands, occ_names, params, callback)

    if combined_p1.status not in ("OPTIMAL", "FEASIBLE"):
        occ_results = [
            OccupationResult(n, d, p,
                             PhaseTwoResult([], [], 0, "SKIPPED", 0.0))
            for n, d, p in zip(occ_names, demands, per_occ_p1)
        ]
        return MultiCurveResult(occ_results, combined_p1,
                                sum(demands), params)

    occ_results = []
    for i, name in enumerate(occ_names):
        if callback:
            callback(f"Phase 2 for {name} …")
        p2 = solve_phase2(per_occ_p1[i], params, callback)
        occ_results.append(OccupationResult(name, demands[i],
                                            per_occ_p1[i], p2))

    return MultiCurveResult(occ_results, combined_p1,
                            sum(demands), params)


def solve(
    demand: np.ndarray,
    params: Optional[SolverParams] = None,
    callback=None,
) -> MultiCurveResult:
    """Backward-compatible single-curve wrapper."""
    return solve_multi([demand], ["Staff"], params, callback)


# ── Result → DataFrames ─────────────────────────────────────────────────────
def shifts_to_dataframe(p1: PhaseOneResult, label: str = "") -> pd.DataFrame:
    rows = []
    for s, cnt in zip(p1.shifts, p1.counts):
        if cnt > 0:
            row = {
                "Day": DAY_NAMES[s.day],
                "DayNum": s.day,
                "Start": s.start_time_str,
                "End": s.end_time_str,
                "DurationHrs": s.duration_hours,
                "Workers": cnt,
            }
            if label:
                row["Occupation"] = label
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["DayNum", "Start"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def schedules_to_dataframe(
    p2: PhaseTwoResult,
    label: str = "",
    emp_offset: int = 0,
) -> pd.DataFrame:
    rows = []
    for w_idx, (sched, hrs) in enumerate(
            zip(p2.worker_schedules, p2.worker_hours)):
        for s in sched:
            row = {
                "Worker": w_idx + 1 + emp_offset,
                "Day": DAY_NAMES[s.day],
                "DayNum": s.day,
                "Start": s.start_time_str,
                "End": s.end_time_str,
                "DurationHrs": s.duration_hours,
                "WeeklyHrs": hrs,
            }
            if label:
                row["Occupation"] = label
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["Worker", "DayNum", "Start"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def coverage_dataframe(result: MultiCurveResult) -> pd.DataFrame:
    times = []
    for t in range(TOTAL_INTERVALS):
        day = t // INTERVALS_PER_DAY
        intra = t % INTERVALS_PER_DAY
        h, m = divmod(intra * 5, 60)
        times.append(f"{DAY_NAMES[day]} {h:02d}:{m:02d}")

    data: Dict = {
        "Interval": range(TOTAL_INTERVALS),
        "Time": times,
        "TotalDemand": result.combined_demand.astype(int),
        "TotalCoverage": result.combined_phase1.coverage.astype(int),
    }
    for occ in result.occupations:
        data[f"Demand_{occ.name}"]    = occ.demand.astype(int)
        data[f"Coverage_{occ.name}"]  = occ.phase1.coverage.astype(int)

    return pd.DataFrame(data)


# ERP day mapping: 0=Sunday, 1=Monday, ..., 6=Saturday
_ERP_DAY = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0}  # solver day → ERP day

_SHIFTTYPE_HEADER = (
    '<Version=7.9.73082/><Class Name=CShifttype/>'
    '<Block Name=Shifttype/><Tab Name=Shifttype/>'
)


def _build_shifttype_rows(result: MultiCurveResult) -> List[Dict]:
    """Build ERP Shifttype rows from combined Phase 1 results."""
    p1 = result.combined_phase1

    # Collect per shift-code: days used (ERP numbering) + timing info
    code_days: Dict[str, set] = defaultdict(set)
    code_info: Dict[str, Dict] = {}

    for s, cnt in zip(p1.shifts, p1.counts):
        if cnt <= 0:
            continue
        code = s.shift_code
        code_days[code].add(_ERP_DAY[s.day])
        if code not in code_info:
            begin_min = s.start_interval * 5
            end_min = (s.start_interval + s.duration_intervals) * 5
            if end_min >= 1440:
                end_min -= 1440
            dur_min = s.duration_intervals * 5
            code_info[code] = {
                "begin": begin_min,
                "end": end_min,
                "duration": dur_min,
            }

    rows = []
    all_days = {0, 1, 2, 3, 4, 5, 6}
    for code in sorted(code_info):
        info = code_info[code]
        days_used = code_days[code]
        if days_used == all_days:
            multi = ""
        else:
            multi = "".join(str(d) for d in sorted(days_used, reverse=True))
        rows.append({
            "begin": info["begin"],
            "BGColor": 18,
            "code": code,
            "duration": info["duration"],
            "end": info["end"],
            "FGColor": 24,
            "fullTime": 1,
            "multiDays": multi,
        })
    return rows

def shift_type_summary(
    p1: PhaseOneResult,
    label: str = "",
) -> pd.DataFrame:
    type_day: Dict[str, Dict[int, int]] = defaultdict(
        lambda: {d: 0 for d in range(7)})
    type_dur: Dict[str, float] = {}

    for s, cnt in zip(p1.shifts, p1.counts):
        if cnt > 0:
            code = s.shift_code
            type_day[code][s.day] += cnt
            type_dur[code] = s.duration_hours

    rows = []
    for code in sorted(type_day):
        days_used = [DAY_ABBR[d] for d in range(7) if type_day[code][d] > 0]
        total = sum(type_day[code].values())
        row = {
            "ShiftType": code,
            "Duration(h)": type_dur[code],
            "Days": ",".join(days_used),
            "Total": total,
        }
        if label:
            row["Occupation"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def build_weekly_report_xlsx(result: MultiCurveResult) -> bytes:
    buf = _io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:

        # ── Roster ───────────────────────────────────────────────────────
        all_roster = []
        all_summary = []
        for occ in result.occupations:
            pfx = occ.name[:3].upper()
            for w_idx, (sched, hrs) in enumerate(
                    zip(occ.phase2.worker_schedules,
                        occ.phase2.worker_hours)):
                emp = f"{pfx}-{w_idx+1:03d}"
                for s in sched:
                    all_roster.append({
                        "Occupation": occ.name,
                        "Worker": emp,
                        "Day": DAY_NAMES[s.day],
                        "Shift": s.shift_code,
                        "Start": s.start_time_str,
                        "End": s.end_time_str,
                        "Hours": s.duration_hours,
                    })
                row: Dict = {
                    "Occupation": occ.name,
                    "Worker": emp,
                    "TotalHours": round(hrs, 1),
                    "FTE(÷45)": round(hrs / 45.0, 2),
                    "Shifts": len(sched),
                }
                for d in range(7):
                    day_s = [s for s in sched if s.day == d]
                    row[DAY_NAMES[d]] = (
                        ", ".join(s.shift_code for s in day_s)
                        if day_s else "OFF")
                all_summary.append(row)

        pd.DataFrame(all_roster).to_excel(writer, sheet_name="Roster",
                                          index=False)
        sdf = pd.DataFrame(all_summary)
        sdf.to_excel(writer, sheet_name="Worker Summary", index=False)
        if not sdf.empty:
            ws = writer.sheets["Worker Summary"]
            for i, col in enumerate(sdf.columns):
                mx = max(sdf[col].astype(str).str.len().max(),
                         len(col)) + 2
                ws.set_column(i, i, mx)

        # ── Shift Types ──────────────────────────────────────────────────
        all_st = [shift_type_summary(o.phase1, o.name)
                  for o in result.occupations]
        if all_st:
            pd.concat(all_st, ignore_index=True).to_excel(
                writer, sheet_name="Shift Types", index=False)

        # ── Coverage ─────────────────────────────────────────────────────
        coverage_dataframe(result).to_excel(
            writer, sheet_name="Coverage", index=False)

        # ── ERP Shifttype ────────────────────────────────────────────────
        st_rows = _build_shifttype_rows(result)
        if st_rows:
            st_df = pd.DataFrame(st_rows)
            st_df.to_excel(writer, sheet_name="Shifttype",
                           index=False, startrow=1)
            ws_st = writer.sheets["Shifttype"]
            ws_st.write(0, 0, _SHIFTTYPE_HEADER)

    return buf.getvalue()
