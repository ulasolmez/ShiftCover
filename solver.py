"""
ShiftCover – Weekly Shift-Covering Optimiser (OR-Tools CP-SAT)
==============================================================
Supports 1-5 occupation curves with SHARED shift structures.

Single-phase solver: Multi-curve Set Covering via CP-SAT.
Shift activation (z[s]) is shared across occupations so every
occupation's workers enter/exit at the same boundaries
(same shuttle schedule).
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
OCC_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # blue, orange, green, red, purple


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
    shift_start_granularity_min: int = 30
    shift_duration_step_min: int  = 30
    max_unique_shifts: int        = 0          # 0 = unlimited
    transition_penalty: int       = 50
    solver_time_limit_sec: int    = 120
    # Per-day limits on distinct entry/exit times (0 = unlimited)
    # Each is a list of 7 ints (Mon–Sun). If None → no constraint.
    max_entries_per_day: Optional[List[int]] = None
    max_exits_per_day: Optional[List[int]] = None
    # Per-day max workers entering shifts (0 = unlimited). List of 7 ints.
    # Counts total workers whose shift STARTS on that day (entry-based, not simultaneous).
    max_headcount_per_day: Optional[List[int]] = None
    # Per-day min workers entering shifts combined across ALL occupations (0 = no minimum). List of 7 ints.
    min_headcount_per_day: Optional[List[int]] = None
    # Per-occupation per-day max workers entering shifts.
    # List of length n_occ; each element is a list of 7 ints (0 = unlimited) or None.
    occ_max_headcount_per_day: Optional[List[Optional[List[int]]]] = None
    # Per-occupation per-day min workers entering shifts.
    # List of length n_occ; each element is a list of 7 ints (0 = no minimum) or None.
    occ_min_headcount_per_day: Optional[List[Optional[List[int]]]] = None
    # Per-occupation per-shift-code per-day max/min workers.
    # List of length n_occ; each element is None or a dict:
    #   {"HHMM-HHMM": [max_mon, ..., max_sun]}  or
    #   {"HHMM-HHMM": {"max": [7 ints], "min": [7 ints]}}  or None.
    # 0 = unlimited/no minimum.  Only codes present are constrained.
    occ_headcount_per_shift_code: Optional[List[Optional[Dict]]] = None
    # Exclude shifts that fall entirely within the 20:00–06:00 window
    exclude_night_shifts: bool = False
    # Circular week: Sunday shifts can wrap into Monday
    circular_week: bool = False
    # Force include/exclude specific shift codes (e.g. ["0630-1430"])
    force_include_shifts: Optional[List[str]] = None
    force_exclude_shifts: Optional[List[str]] = None
    # Allowed start/end minutes-of-day (e.g. [0, 60, 120, ...] for every hour).
    # Shifts are only generated when BOTH start and end fall in this set.
    # None = all times allowed.
    allowed_slot_minutes: Optional[List[int]] = None


@dataclass
class PhaseOneResult:
    shifts: List[CandidateShift]
    counts: List[int]
    total_worker_intervals: int
    coverage: np.ndarray
    status: str
    elapsed_sec: float


def daily_entry_headcount(p1: PhaseOneResult) -> List[int]:
    """Workers entering per day = sum of shift counts whose start day == d."""
    daily = [0] * 7
    for s, cnt in zip(p1.shifts, p1.counts):
        daily[s.day] += cnt
    return daily


def peak_daily_simultaneous(p1: PhaseOneResult) -> List[int]:
    """Peak simultaneous workers per day from the coverage array."""
    peaks = []
    for day in range(7):
        s = day * INTERVALS_PER_DAY
        e = s + INTERVALS_PER_DAY
        peaks.append(int(p1.coverage[s:e].max()))
    return peaks


def max_headcount(p1: PhaseOneResult) -> int:
    """Max daily entry headcount across the week (sum of workers starting shift each day)."""
    if p1.counts:  # non-empty result
        entries = daily_entry_headcount(p1)
        return max(entries) if entries else 0
    return 0


@dataclass
class OccupationResult:
    name: str
    demand: np.ndarray
    phase1: PhaseOneResult


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
    _allowed_mins = set(params.allowed_slot_minutes) if params.allowed_slot_minutes is not None else None
    codes: set = set()
    for start in range(0, INTERVALS_PER_DAY, start_step):
        for dur in range(min_dur, max_dur + 1, dur_step):
            if params.exclude_night_shifts and is_night_shift(start, dur):
                continue
            if _allowed_mins is not None:
                if (start * 5) % 1440 not in _allowed_mins:
                    continue
                if ((start + dur) * 5) % 1440 not in _allowed_mins:
                    continue
            codes.add(_shift_code_from(start, dur))
    return sorted(codes)


def generate_candidate_shifts(params: SolverParams) -> List[CandidateShift]:
    start_step = max(1, params.shift_start_granularity_min // 5)
    dur_step   = max(1, params.shift_duration_step_min // 5)
    min_dur    = int(params.min_shift_hours * INTERVALS_PER_HOUR)
    max_dur    = int(params.max_shift_hours * INTERVALS_PER_HOUR)

    _excl_set = set(params.force_exclude_shifts) if params.force_exclude_shifts else set()
    _allowed_mins = set(params.allowed_slot_minutes) if params.allowed_slot_minutes is not None else None

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

                if _allowed_mins is not None:
                    if (start * 5) % 1440 not in _allowed_mins:
                        continue
                    if ((start + dur) * 5) % 1440 not in _allowed_mins:
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


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 – Multi-Curve Model Builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_multi_curve_model(
    demands: List[np.ndarray],
    params: SolverParams,
    cov: Dict[int, List[int]],
    shifts: List[CandidateShift],
    n_occ: int,
    day_shift_idxs: Dict[int, List[int]],
    callback=None,
) -> Tuple[cp_model.CpModel, Dict[int, Dict[int, object]], Dict[int, object]]:
    """
    Build the CP-SAT model with all Phase 1 constraints.

    Returns:
        model: The configured CpModel.
        x:      x[occ][s_idx] = integer variable for workers of occupation occ
                assigned to shift s.
        z:      z[s_idx] = boolean activation variable (shared across occupations).
    """
    model = cp_model.CpModel()

    # Per-occupation decision vars: x[occ][s] = workers of that occ on shift s
    x: Dict[int, Dict[int, object]] = {}
    for occ in range(n_occ):
        x[occ] = {}
        peak_d = int(demands[occ].max()) if demands[occ].max() > 0 else 1
        bound = peak_d
        # respect per-occ min/max headcount limits when setting variable domain
        if params.occ_max_headcount_per_day and params.occ_max_headcount_per_day[occ]:
            vals = [v for v in params.occ_max_headcount_per_day[occ] if v > 0]
            if vals:
                bound = max(bound, max(vals))
        if params.occ_min_headcount_per_day and params.occ_min_headcount_per_day[occ]:
            vals = [v for v in params.occ_min_headcount_per_day[occ] if v > 0]
            if vals:
                bound = max(bound, max(vals))
        for s in shifts:
            x[occ][s.idx] = model.new_int_var(0, bound, f"x_{occ}_{s.idx}")

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

    # ── Force-include shift codes ─────────────────────────────────────────
    if params.force_include_shifts:
        incl_set = set(params.force_include_shifts)
        code_to_idxs: Dict[str, List[int]] = {}
        for s in shifts:
            if s.shift_code in incl_set:
                code_to_idxs.setdefault(s.shift_code, []).append(s.idx)
        for code, idxs in code_to_idxs.items():
            model.add(sum(z[i] for i in idxs) >= 1)
        if callback:
            callback(f"Force-include {len(code_to_idxs)} shift code(s)")

    # ── Max unique shifts ─────────────────────────────────────────────────
    if params.max_unique_shifts > 0:
        model.add(sum(z[s.idx] for s in shifts) <= params.max_unique_shifts)
        if callback:
            callback(f"Max unique shifts ≤ {params.max_unique_shifts}")

    # ── Per-day max distinct entry times ──────────────────────────────────
    if params.max_entries_per_day:
        day_starts: Dict[int, set] = {d: set() for d in range(7)}
        for s in shifts:
            day_starts[s.day].add(s.start_interval)
        for day in range(7):
            limit = params.max_entries_per_day[day]
            if limit <= 0:
                continue
            starts = sorted(day_starts[day])
            if len(starts) <= limit:
                continue
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

    # ── Per-day max distinct exit times ───────────────────────────────────
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

    # ── Per-occupation daily headcount limits (entry-based) ───────────────
    if params.occ_max_headcount_per_day:
        for occ in range(n_occ):
            occ_limits = params.occ_max_headcount_per_day[occ]
            if occ_limits is None:
                continue
            for day in range(7):
                limit = occ_limits[day]
                if limit <= 0:
                    continue
                idxs = day_shift_idxs[day]
                if not idxs:
                    continue
                model.add(
                    sum(x[occ][si] for si in idxs) <= limit
                )
        if callback:
            callback(f"Per-occ headcount limits set for {sum(1 for l in params.occ_max_headcount_per_day if l is not None)} occupation(s)")

    # ── Per-occupation daily minimum headcount (entry-based) ─────────────
    if params.occ_min_headcount_per_day:
        for occ in range(n_occ):
            occ_mins = params.occ_min_headcount_per_day[occ]
            if occ_mins is None:
                continue
            for day in range(7):
                min_limit = occ_mins[day]
                if min_limit <= 0:
                    continue
                idxs = day_shift_idxs[day]
                if not idxs:
                    continue
                model.add(
                    sum(x[occ][si] for si in idxs) >= min_limit
                )
        if callback:
            callback(f"Per-occ min headcount set for {sum(1 for l in params.occ_min_headcount_per_day if l is not None)} occupation(s)")

    # ── Per-occupation per-shift-code headcount limits ───────────────────
    if params.occ_headcount_per_shift_code:
        for occ in range(n_occ):
            sc_limits = params.occ_headcount_per_shift_code[occ]
            if sc_limits is None:
                continue
            for code, spec in sc_limits.items():
                # spec can be: list[7] (max only) or dict {"max": [...], "min": [...]}
                if isinstance(spec, list):
                    max_limits = spec
                    min_limits = None
                else:
                    max_limits = spec.get("max")
                    min_limits = spec.get("min")
                # Collect matching shift indices per day
                per_day: Dict[int, List[int]] = {d: [] for d in range(7)}
                for s in shifts:
                    if s.shift_code == code:
                        per_day[s.day].append(s.idx)
                for day in range(7):
                    idxs = per_day[day]
                    if not idxs:
                        continue
                    if max_limits:
                        limit = max_limits[day]
                        if limit > 0:
                            model.add(sum(x[occ][si] for si in idxs) <= limit)
                    if min_limits:
                        min_req = min_limits[day]
                        if min_req > 0:
                            model.add(sum(x[occ][si] for si in idxs) >= min_req)
        if callback:
            n_constrained = sum(1 for sc in params.occ_headcount_per_shift_code if sc is not None)
            callback(f"Shift-code headcount limits set for {n_constrained} occupation(s)")

    # ── Per-day max headcount (combined, entry-based) ────────────────────
    if params.max_headcount_per_day:
        for day in range(7):
            limit = params.max_headcount_per_day[day]
            if limit <= 0:
                continue
            idxs = day_shift_idxs[day]
            if not idxs:
                continue
            total_terms = []
            for occ in range(n_occ):
                total_terms.extend(x[occ][si] for si in idxs)
            model.add(sum(total_terms) <= limit)
        if callback:
            callback(f"Max headcount/day: {params.max_headcount_per_day}")

    # ── Per-day min headcount (combined, entry-based) ────────────────────
    if params.min_headcount_per_day:
        for day in range(7):
            min_limit = params.min_headcount_per_day[day]
            if min_limit <= 0:
                continue
            idxs = day_shift_idxs[day]
            if not idxs:
                continue
            total_terms = []
            for occ in range(n_occ):
                total_terms.extend(x[occ][si] for si in idxs)
            model.add(sum(total_terms) >= min_limit)
        if callback:
            callback(f"Min headcount/day: {params.min_headcount_per_day}")

    # ── Per-occupation coverage constraints ───────────────────────────────
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

    # ── Objective ─────────────────────────────────────────────────────────
    obj_terms = []
    for occ in range(n_occ):
        for s in shifts:
            obj_terms.append(x[occ][s.idx] * s.duration_intervals)

    tp = params.transition_penalty
    if tp > 0:
        for s in shifts:
            obj_terms.append(z[s.idx] * tp * 2)
        if callback:
            callback(f"Transition penalty {tp} on shared shift activation")

    model.minimize(sum(obj_terms))
    if callback:
        callback("Model built – starting Phase 1 solve …")

    return model, x, z


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 – Solver Runner
# ══════════════════════════════════════════════════════════════════════════════

def _solve_cp_sat(
    model: cp_model.CpModel,
    params: SolverParams,
    t0: float,
) -> Tuple[cp_model.CpSolver, int, str, float]:
    """
    Configure and run the CP-SAT solver.

    Returns:
        solver:     The CpSolver instance (can query .value() to extract results).
        status_code:  cp_model.OPTIMAL, FEASIBLE, INFEASIBLE, etc.
        status_str:   Human-readable status string.
        elapsed_sec:  Wall-clock seconds from t0 to solve completion.
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = params.solver_time_limit_sec
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = False

    status_code = solver.solve(model)
    status_str  = solver.status_name(status_code)
    elapsed     = time.time() - t0
    return solver, status_code, status_str, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 – Result Extractor
# ══════════════════════════════════════════════════════════════════════════════

def _extract_phase1_results(
    solver: cp_model.CpSolver,
    status_code: int,
    status_str: str,
    elapsed: float,
    x: Dict[int, Dict[int, object]],
    shifts: List[CandidateShift],
    n_occ: int,
    occ_names: List[str],
    callback=None,
) -> Tuple[List[PhaseOneResult], PhaseOneResult]:
    """
    Extract per-occupation and combined PhaseOneResult from a solved model.

    If the model is not OPTIMAL/FEASIBLE, returns empty results with the
    solver status preserved.

    Returns:
        per_occ:      List of n_occ PhaseOneResults.
        combined_p1:  Single PhaseOneResult aggregating all occupations.
    """
    from ortools.sat.python import cp_model as _cp

    if status_code not in (_cp.OPTIMAL, _cp.FEASIBLE):
        empty = PhaseOneResult(shifts, [], 0,
                               np.zeros(TOTAL_INTERVALS, dtype=int),
                               status_str, elapsed)
        per_occ_empty = [
            PhaseOneResult(shifts, [], 0,
                           np.zeros(TOTAL_INTERVALS, dtype=int),
                           status_str, elapsed)
            for _ in range(n_occ)
        ]
        return per_occ_empty, empty

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
    combined_p1 = PhaseOneResult(
        shifts, combined_counts, combined_wi, combined_coverage,
        status_str, elapsed)

    if callback:
        for occ in range(n_occ):
            wh = per_occ[occ].total_worker_intervals / INTERVALS_PER_HOUR
            callback(f"  {occ_names[occ]}: {wh:.0f} worker-hours")
        n_active = sum(1 for c in combined_counts if c > 0)
        callback(f"Phase 1 done: status={status_str}, "
                 f"total worker-hours={combined_wi/INTERVALS_PER_HOUR:.0f}, "
                 f"unique shifts={n_active}")

    return per_occ, combined_p1


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 – Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def solve_phase1_multi(
    demands: List[np.ndarray],
    occ_names: List[str],
    params: SolverParams,
    callback=None,
) -> Tuple[List[PhaseOneResult], PhaseOneResult]:
    """
    Solve all occupation curves simultaneously with SHARED shift activation.

    Orchestrates the three pipeline stages:
      1. _build_multi_curve_model  – construct CP-SAT model
      2. _solve_cp_sat              – run the solver
      3. _extract_phase1_results    – materialise results

    Returns (per_occ_results, combined_result).
    """
    n_occ = len(demands)
    for i, d in enumerate(demands):
        assert d.shape == (TOTAL_INTERVALS,), \
            f"Demand[{i}] must have length {TOTAL_INTERVALS}"

    t0 = time.time()

    # ── Stage 1: generate shifts + coverage map ──────────────────────────
    shifts = generate_candidate_shifts(params)
    if callback:
        callback(f"Generated {len(shifts):,} candidate shifts")

    cov = build_coverage_map(shifts)
    if callback:
        callback("Built coverage map")

    # Precompute: for each day, list of shift indices that start on that day
    day_shift_idxs: Dict[int, List[int]] = {d: [] for d in range(7)}
    for s in shifts:
        day_shift_idxs[s.day].append(s.idx)

    # ── Stage 2: build model ─────────────────────────────────────────────
    model, x, z = _build_multi_curve_model(
        demands, params, cov, shifts, n_occ, day_shift_idxs, callback)

    # ── Stage 3: solve ───────────────────────────────────────────────────
    solver, status_code, status_str, elapsed = _solve_cp_sat(model, params, t0)

    # ── Stage 4: extract results ─────────────────────────────────────────
    return _extract_phase1_results(
        solver, status_code, status_str, elapsed,
        x, shifts, n_occ, occ_names, callback)


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

    occ_results = [
        OccupationResult(n, d, p)
        for n, d, p in zip(occ_names, demands, per_occ_p1)
    ]
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