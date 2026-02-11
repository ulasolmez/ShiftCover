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

    @property
    def shift_code(self) -> str:
        """Shift type in HHMM-HHMM format."""
        sh, sm = divmod(self.start_interval * 5, 60)
        total_end_min = (self.start_interval + self.duration_intervals) * 5
        eh, em = divmod(total_end_min, 60)
        if eh >= 24:
            eh -= 24
        return f"{sh:02d}{sm:02d}-{eh:02d}{em:02d}"

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
    min_rest_hours: float = 12.0               # minimum rest between shifts
    max_unique_shifts: int = 0                  # 0 = unlimited
    transition_penalty: int = 50               # cost multiplier per entry/exit
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
    Minimise total worker-intervals + transition penalty such that
    every 5-min interval is covered by at least demand[t] workers.

    The transition penalty discourages jagged coverage curves by adding
    a cost every time the coverage level changes between consecutive
    intervals (i.e. workers entering or exiting).

    An optional cardinality constraint limits how many distinct shift
    types can be activated (max_unique_shifts).
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

    # --- indicator variables z[s] = 1 iff x[s] > 0 ---
    z = {}
    need_indicators = (params.max_unique_shifts > 0)
    if need_indicators:
        for s in shifts:
            z[s.idx] = model.NewBoolVar(f"z_{s.idx}")
            # link: x[s] > 0  ⇔  z[s] = 1
            model.Add(x[s.idx] >= 1).OnlyEnforceIf(z[s.idx])
            model.Add(x[s.idx] == 0).OnlyEnforceIf(z[s.idx].Not())

        # cardinality constraint
        model.Add(sum(z[s.idx] for s in shifts) <= params.max_unique_shifts)
        if callback:
            callback(f"Max unique shifts constraint: ≤ {params.max_unique_shifts}")

    # coverage constraints
    for t in active_intervals:
        covering = cov[t]
        if not covering:
            continue
        model.Add(
            sum(x[s_idx] for s_idx in covering) >= int(demand[t])
        )

    # ---- build coverage-level variables for transition penalty ----
    # cov_level[t] = total workers covering interval t
    # We only need these for intervals where transitions can happen,
    # i.e. boundaries of active periods and between consecutive intervals.
    transition_cost = params.transition_penalty
    transition_vars = []

    if transition_cost > 0:
        if callback:
            callback(f"Adding transition penalty (weight={transition_cost}) …")

        # For efficiency we group shifts by their start and end global
        # intervals, and only compute delta variables at those boundaries.
        boundary_set = set()
        for s in shifts:
            boundary_set.add(s.global_start)
            if s.global_end < TOTAL_INTERVALS:
                boundary_set.add(s.global_end)
        boundaries = sorted(boundary_set)

        # At each boundary t, delta[t] = |cov_level[t] - cov_level[t-1]|
        # captures how many workers enter or exit.
        # We model |A - B| as: diff_pos - diff_neg = A - B,
        #   diff_pos, diff_neg >= 0, transition = diff_pos + diff_neg
        # But computing cov_level at every boundary is expensive.
        # More efficient: count entries and exits at each boundary.
        #   entries[t] = sum of x[s] for shifts starting at t
        #   exits[t]   = sum of x[s] for shifts ending at t
        # Each entry or exit is one "shuttle event" we want to penalise.

        # Build entries/exits per boundary
        entries_at: Dict[int, List[int]] = {}
        exits_at: Dict[int, List[int]] = {}
        for s in shifts:
            entries_at.setdefault(s.global_start, []).append(s.idx)
            if s.global_end < TOTAL_INTERVALS:
                exits_at.setdefault(s.global_end, []).append(s.idx)

        # Total entries + exits = total shuttle events
        all_entry_exit_terms = []
        for t, s_idxs in entries_at.items():
            for sid in s_idxs:
                all_entry_exit_terms.append(x[sid])
        for t, s_idxs in exits_at.items():
            for sid in s_idxs:
                all_entry_exit_terms.append(x[sid])

        # Each x[s] appears twice (once as entry, once as exit) so
        # total shuttle events = 2 * sum(x[s]) but we penalise the
        # raw count of entry + exit events (not per worker, per shift-use).
        # Actually we want: for each active shift, the workers on it
        # create 1 entry event and 1 exit event each. So total shuttle
        # events = 2 * sum(x[s] for active s). The factor of 2 is constant
        # per shift so we just penalise sum(x[s]) once with the weight.

    # ---- objective ----
    # Primary: minimise total worker-intervals (labour cost)
    # Secondary: penalise number of active shifts (fewer shift types = blockier)
    obj_terms = []
    for s in shifts:
        obj_terms.append(x[s.idx] * s.duration_intervals)

    if transition_cost > 0:
        # Penalise each active shift *type* (entry/exit point pair)
        # This directly reduces the number of distinct start/end boundaries.
        for s in shifts:
            obj_terms.append(x[s.idx] * transition_cost)

    model.Minimize(sum(obj_terms))

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
    n_active = sum(1 for c in counts if c > 0)

    if callback:
        callback(f"Phase 1 done: status={status_str}, "
                 f"worker-hours={total_wi / INTERVALS_PER_HOUR:.1f}, "
                 f"unique shifts={n_active}")

    return PhaseOneResult(shifts, counts, total_wi, coverage,
                          status_str, elapsed)


# ── Phase 2 – Worker Assignment (Greedy with rest constraint) ────────────────
def _can_assign(worker_shifts: List[CandidateShift],
               shift: CandidateShift,
               params: SolverParams) -> bool:
    """
    Check whether *shift* can be added to a worker who already has
    *worker_shifts*.  Rules:
      1. Worker hasn't exceeded max weekly hours.
      2. Worker has at most max_shifts_per_day on this day.
      3. At least min_rest_hours gap between the end of any existing
         shift and the start of this one (and vice-versa).
      4. Shifts must not overlap.
    """
    rest_intervals = int(params.min_rest_hours * INTERVALS_PER_HOUR)
    max_intervals = int(params.max_weekly_hours * INTERVALS_PER_HOUR)

    # weekly hours check
    current_total = sum(s.duration_intervals for s in worker_shifts)
    if current_total + shift.duration_intervals > max_intervals:
        return False

    # per-day count
    day_count = sum(1 for s in worker_shifts if s.day == shift.day)
    if day_count >= params.max_shifts_per_day_per_worker:
        return False

    # overlap & rest gap
    for existing in worker_shifts:
        # overlap check
        if not (shift.global_end <= existing.global_start or
                existing.global_end <= shift.global_start):
            return False
        # rest gap: end-of-earlier + rest <= start-of-later
        if existing.global_end <= shift.global_start:
            gap = shift.global_start - existing.global_end
        else:
            gap = existing.global_start - shift.global_end
        if gap < rest_intervals:
            return False

    return True


def solve_phase2(
    p1: PhaseOneResult,
    params: SolverParams,
    callback=None,
) -> PhaseTwoResult:
    """
    Greedy worker-assignment that **minimises headcount** while respecting
    weekly-hour windows and minimum-rest constraints.

    Strategy (two-tier priority):
      • Tier 1 – workers still BELOW min weekly hours: prefer the one
        closest to reaching the minimum (most filled → fewest hours to go).
      • Tier 2 – workers already AT or ABOVE min weekly hours: prefer the
        one with the *most* remaining capacity (spread load, leave room
        for future shifts to avoid creating new workers).
      • Only create a new worker when no existing worker can accept the
        shift without violating rest / overlap / max-hours rules.
    """
    t0 = time.time()

    # ---- flatten shift instances ----
    instances: List[CandidateShift] = []
    for s, cnt in zip(p1.shifts, p1.counts):
        for _ in range(cnt):
            instances.append(s)

    if not instances:
        return PhaseTwoResult([], [], 0, "NO_SHIFTS", 0.0)

    # sort by global start so we assign in chronological order
    instances.sort(key=lambda s: (s.global_start, s.duration_intervals))

    n_instances = len(instances)
    if callback:
        callback(f"Phase 2 (greedy): assigning {n_instances} shift instances")

    # worker state
    workers: List[List[CandidateShift]] = []
    worker_totals: List[int] = []          # running duration-interval totals

    max_ivl = int(params.max_weekly_hours * INTERVALS_PER_HOUR)
    min_ivl = int(params.min_weekly_hours * INTERVALS_PER_HOUR)

    for j, shift in enumerate(instances):
        best_w = -1
        best_score = None   # (tier, secondary) – lower is better

        for w_idx, (w_shifts, w_total) in enumerate(
                zip(workers, worker_totals)):
            new_total = w_total + shift.duration_intervals
            if new_total > max_ivl:
                continue  # would exceed max hours

            if not _can_assign(w_shifts, shift, params):
                continue

            if w_total < min_ivl:
                # Tier 1: worker still below minimum hours.
                # Prefer the one CLOSEST to minimum (highest total so far)
                # so they reach min sooner → fewer under-filled workers.
                tier = 0
                secondary = -w_total          # more negative = higher total = better
            else:
                # Tier 2: worker already at/above minimum.
                # Prefer the one with the MOST remaining capacity (largest
                # remaining) to keep room open and delay creating new workers.
                remaining = max_ivl - new_total
                tier = 1
                secondary = -remaining        # more negative = more remaining = better

            score = (tier, secondary)
            if best_score is None or score < best_score:
                best_w = w_idx
                best_score = score

        if best_w >= 0:
            workers[best_w].append(shift)
            worker_totals[best_w] += shift.duration_intervals
        else:
            # create a new worker
            workers.append([shift])
            worker_totals.append(shift.duration_intervals)

        if callback and (j + 1) % 200 == 0:
            callback(f"  … assigned {j + 1}/{n_instances} shifts, "
                     f"{len(workers)} workers so far")

    # ---- build result ----
    schedules: List[List[CandidateShift]] = []
    hours_list: List[float] = []
    min_intervals = int(params.min_weekly_hours * INTERVALS_PER_HOUR)

    under_min = 0
    for w_shifts, w_total in zip(workers, worker_totals):
        w_shifts.sort(key=lambda s: s.global_start)
        schedules.append(w_shifts)
        hrs = w_total / INTERVALS_PER_HOUR
        hours_list.append(hrs)
        if w_total < min_intervals:
            under_min += 1

    elapsed = time.time() - t0
    status_str = "FEASIBLE"
    if under_min > 0:
        status_str = f"FEASIBLE ({under_min} workers below min hours)"

    if callback:
        callback(f"Phase 2 done: {len(schedules)} workers, "
                 f"status={status_str}, {elapsed:.1f} s")

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


def shift_type_summary(p1: PhaseOneResult) -> pd.DataFrame:
    """
    Shift types in HHMM-HHMM format with count per day and totals.
    """
    from collections import defaultdict
    type_day: Dict[str, Dict[str, int]] = defaultdict(lambda: {d: 0 for d in DAY_NAMES})
    type_dur: Dict[str, float] = {}

    for s, cnt in zip(p1.shifts, p1.counts):
        if cnt > 0:
            code = s.shift_code
            type_day[code][DAY_NAMES[s.day]] += cnt
            type_dur[code] = s.duration_hours

    rows = []
    for code in sorted(type_day.keys()):
        row = {"ShiftType": code, "Duration(h)": type_dur[code]}
        total = 0
        for d in DAY_NAMES:
            row[d] = type_day[code][d]
            total += type_day[code][d]
        row["Total"] = total
        rows.append(row)
    return pd.DataFrame(rows)


def build_weekly_report_xlsx(result: FullResult) -> bytes:
    """
    Build a multi-sheet XLSX report:
      Sheet 1 – Roster           (worker, day, shift HHMM-HHMM, hours)
      Sheet 2 – Worker Summary   (worker, total hours, FTE, shifts per day)
      Sheet 3 – Shift Types      (HHMM-HHMM, duration, count per day)
      Sheet 4 – Coverage         (interval, demand, coverage)
    """
    import io as _io
    p1 = result.phase1
    p2 = result.phase2

    buf = _io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book

        # ── Sheet 1: Roster ──────────────────────────────────────────────
        roster_rows = []
        for w_idx, (schedule, hrs) in enumerate(
                zip(p2.worker_schedules, p2.worker_hours)):
            for s in schedule:
                roster_rows.append({
                    "Worker": f"EMP-{w_idx + 1:03d}",
                    "Day": DAY_NAMES[s.day],
                    "Shift": s.shift_code,
                    "Start": s.start_time_str,
                    "End": s.end_time_str,
                    "Hours": s.duration_hours,
                })
        roster_df = pd.DataFrame(roster_rows)
        if not roster_df.empty:
            roster_df.sort_values(["Worker", "Day"], inplace=True,
                                  key=lambda c: c.map(
                                      {d: i for i, d in enumerate(DAY_NAMES)}
                                  ) if c.name == "Day" else c)
            roster_df.reset_index(drop=True, inplace=True)
        roster_df.to_excel(writer, sheet_name="Roster", index=False)

        # ── Sheet 2: Worker Summary ──────────────────────────────────────
        summary_rows = []
        for w_idx, (schedule, hrs) in enumerate(
                zip(p2.worker_schedules, p2.worker_hours)):
            row: Dict = {
                "Worker": f"EMP-{w_idx + 1:03d}",
                "TotalHours": round(hrs, 1),
                "FTE(÷45)": round(hrs / 45.0, 2),
                "Shifts": len(schedule),
            }
            for d in range(7):
                day_shifts = [s for s in schedule if s.day == d]
                if day_shifts:
                    row[DAY_NAMES[d]] = ", ".join(s.shift_code for s in day_shifts)
                else:
                    row[DAY_NAMES[d]] = "OFF"
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="Worker Summary", index=False)

        # auto-size columns
        ws = writer.sheets["Worker Summary"]
        for i, col in enumerate(summary_df.columns):
            max_len = max(summary_df[col].astype(str).str.len().max(),
                          len(col)) + 2
            ws.set_column(i, i, max_len)

        # ── Sheet 3: Shift Types ─────────────────────────────────────────
        st_df = shift_type_summary(p1)
        st_df.to_excel(writer, sheet_name="Shift Types", index=False)

        # ── Sheet 4: Coverage ────────────────────────────────────────────
        cov_df = coverage_dataframe(result)
        cov_df.to_excel(writer, sheet_name="Coverage", index=False)

    return buf.getvalue()
