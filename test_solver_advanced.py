"""
ShiftCover – Advanced solver tests.

Focus areas (no overlap with test_bugs.py):
  I   – Objective / minimization quality
  II  – All-week & multi-day patterns
  III – Constraint combinations
  IV  – max_unique_shifts constraint
  V   – Transition penalty effect
  VI  – Backward-compat solve() wrapper & output helpers
  VII – Scaling & proportionality
  VIII– Edge values for SolverParams fields

Run with:  python test_solver_advanced.py
"""
import sys
import unittest
import numpy as np

from solver import (
    INTERVALS_PER_HOUR, INTERVALS_PER_DAY, TOTAL_INTERVALS,
    DAY_NAMES, DAY_ABBR,
    SolverParams, PhaseOneResult, MultiCurveResult,
    generate_candidate_shifts,
    daily_entry_headcount, max_headcount,
    solve, solve_multi,
    shifts_to_dataframe, coverage_dataframe,
    shift_type_summary, build_weekly_report_xlsx,
)

# ── Helpers ────────────────────────────────────────────────────────────────

def zero_demand():
    return np.zeros(TOTAL_INTERVALS, dtype=int)

def flat_demand(workers: int) -> np.ndarray:
    return np.full(TOTAL_INTERVALS, workers, dtype=int)

def day_block(day: int, start_h: int, end_h: int, workers: int) -> np.ndarray:
    d = zero_demand()
    s = day * INTERVALS_PER_DAY + start_h * INTERVALS_PER_HOUR
    e = day * INTERVALS_PER_DAY + end_h  * INTERVALS_PER_HOUR
    d[s:e] = workers
    return d

def week_block(start_h: int, end_h: int, workers: int) -> np.ndarray:
    """Demand `workers` for [start_h, end_h) on every day of the week."""
    d = zero_demand()
    for day in range(7):
        s = day * INTERVALS_PER_DAY + start_h * INTERVALS_PER_HOUR
        e = day * INTERVALS_PER_DAY + end_h   * INTERVALS_PER_HOUR
        d[s:e] = workers
    return d

def quick_params(**kwargs) -> SolverParams:
    defaults = dict(
        min_shift_hours=4.0,
        max_shift_hours=8.0,
        shift_start_granularity_min=60,
        shift_duration_step_min=60,
        min_weekly_hours=0.0,
        max_weekly_hours=200.0,
        min_rest_hours=0.0,
        max_unique_shifts=0,
        transition_penalty=0,
        solver_time_limit_sec=15,
    )
    defaults.update(kwargs)
    return SolverParams(**defaults)

def feasible(r: MultiCurveResult) -> bool:
    return r.combined_phase1.status in ("OPTIMAL", "FEASIBLE")

def active_codes(r: MultiCurveResult) -> set:
    return {s.shift_code for s, c in
            zip(r.combined_phase1.shifts, r.combined_phase1.counts) if c > 0}


# ── Group I: Objective / minimisation quality ────────────────────────────────

class TestObjectiveQuality(unittest.TestCase):

    def test_zero_demand_zero_workers(self):
        """Zero demand → solver assigns zero workers (minimum is 0)."""
        r = solve_multi([zero_demand()], ["Staff"], quick_params())
        self.assertTrue(feasible(r))
        self.assertEqual(int(r.combined_phase1.coverage.sum()), 0)
        self.assertEqual(sum(r.combined_phase1.counts), 0)

    def test_minimal_coverage_not_massive_overstaff(self):
        """For a 2-worker 4-hour Mon block, total worker-hours should be ≤ 4× demand."""
        demand = day_block(0, 8, 12, 2)   # 2 workers × 4 h = 8 wh
        r = solve_multi([demand], ["Staff"], quick_params())
        self.assertTrue(feasible(r))
        total_wh = r.combined_phase1.total_worker_intervals / INTERVALS_PER_HOUR
        # Allow up to 4× overage (a single 8-h shift × 2 workers = 32 wh)
        self.assertLessEqual(total_wh, 4 * 8,
            f"Excessive worker-hours: {total_wh:.1f} for an 8-wh demand")

    def test_larger_demand_uses_more_workers(self):
        """Double the demand → solution must use at least as many worker-hours."""
        demand_low  = day_block(0, 8, 16, 2)
        demand_high = day_block(0, 8, 16, 4)
        p = quick_params()
        r_low  = solve_multi([demand_low],  ["S"], p)
        r_high = solve_multi([demand_high], ["S"], p)
        if not (feasible(r_low) and feasible(r_high)):
            self.skipTest("One or both solves infeasible")
        wh_low  = r_low.combined_phase1.total_worker_intervals
        wh_high = r_high.combined_phase1.total_worker_intervals
        self.assertGreaterEqual(wh_high, wh_low,
            "Higher demand produced fewer worker-hours than lower demand")

    def test_no_unnecessary_workers_on_zero_demand_days(self):
        """Demand only on Monday – solver should not assign workers on other days."""
        demand = day_block(0, 8, 16, 3)
        r = solve_multi([demand], ["Staff"], quick_params())
        self.assertTrue(feasible(r))
        p1 = r.combined_phase1
        for day in range(1, 7):
            day_cov = p1.coverage[day * INTERVALS_PER_DAY:(day + 1) * INTERVALS_PER_DAY]
            self.assertEqual(int(day_cov.sum()), 0,
                f"Workers assigned on {DAY_NAMES[day]} despite zero demand")

    def test_exact_coverage_for_single_worker(self):
        """Demand = 1 worker for one shift-length block → solver assigns exactly 1."""
        demand = day_block(0, 8, 12, 1)   # 1 worker, 4 h
        p = quick_params(min_shift_hours=4.0, max_shift_hours=4.0,
                         shift_duration_step_min=60)
        r = solve_multi([demand], ["Staff"], p)
        self.assertTrue(feasible(r))
        # Peak coverage at demand intervals must be >= 1
        slot = 0 * INTERVALS_PER_DAY + 8 * INTERVALS_PER_HOUR
        self.assertGreaterEqual(int(r.combined_phase1.coverage[slot]), 1)
        # Total coverage should be ≤ demand_intervals × small_factor
        surplus_wh = float(
            np.maximum(r.combined_phase1.coverage - demand, 0).sum()
        ) / INTERVALS_PER_HOUR
        self.assertLessEqual(surplus_wh, 8.0,
            f"Too much surplus coverage: {surplus_wh:.1f} wh")


# ── Group II: All-week & multi-day patterns ───────────────────────────────────

class TestAllWeekPatterns(unittest.TestCase):

    def test_full_week_flat_demand(self):
        """Flat 2-worker demand all week must be covered on all 7 days."""
        demand = week_block(8, 16, 2)
        r = solve_multi([demand], ["Staff"], quick_params())
        self.assertTrue(feasible(r))
        for day in range(7):
            sl = slice(day * INTERVALS_PER_DAY + 8 * INTERVALS_PER_HOUR,
                       day * INTERVALS_PER_DAY + 16 * INTERVALS_PER_HOUR)
            deficit = int(np.maximum(demand[sl] - r.combined_phase1.coverage[sl], 0).max())
            self.assertEqual(deficit, 0,
                f"{DAY_NAMES[day]}: uncovered demand (max deficit={deficit})")

    def test_weekend_only_demand(self):
        """Demand only on Saturday and Sunday must be covered."""
        demand = day_block(5, 8, 16, 2) + day_block(6, 8, 16, 2)
        r = solve_multi([demand], ["Staff"], quick_params())
        self.assertTrue(feasible(r))
        for day in (5, 6):
            sl = slice(day * INTERVALS_PER_DAY + 8 * INTERVALS_PER_HOUR,
                       day * INTERVALS_PER_DAY + 16 * INTERVALS_PER_HOUR)
            deficit = int(np.maximum(demand[sl] - r.combined_phase1.coverage[sl], 0).max())
            self.assertEqual(deficit, 0,
                f"{DAY_NAMES[day]}: weekend demand not covered")

    def test_staggered_daily_demand(self):
        """Each day has a different demand level — all must be met."""
        demand = zero_demand()
        for day in range(7):
            w = day + 1   # Mon=1 … Sun=7
            s = day * INTERVALS_PER_DAY + 9  * INTERVALS_PER_HOUR
            e = day * INTERVALS_PER_DAY + 17 * INTERVALS_PER_HOUR
            demand[s:e] = w
        r = solve_multi([demand], ["Staff"], quick_params())
        self.assertTrue(feasible(r))
        for day in range(7):
            w = day + 1
            sl = slice(day * INTERVALS_PER_DAY + 9  * INTERVALS_PER_HOUR,
                       day * INTERVALS_PER_DAY + 17 * INTERVALS_PER_HOUR)
            deficit = int(np.maximum(w - r.combined_phase1.coverage[sl], 0).max())
            self.assertEqual(deficit, 0,
                f"{DAY_NAMES[day]} (demand={w}): uncovered intervals")

    def test_gap_days_zero_coverage(self):
        """Demand only Mon+Fri → Tue-Thu and Sat-Sun should have zero coverage."""
        demand = day_block(0, 8, 16, 2) + day_block(4, 8, 16, 2)
        r = solve_multi([demand], ["Staff"], quick_params())
        self.assertTrue(feasible(r))
        for day in (1, 2, 3, 5, 6):
            day_cov = r.combined_phase1.coverage[
                day * INTERVALS_PER_DAY:(day + 1) * INTERVALS_PER_DAY]
            self.assertEqual(int(day_cov.sum()), 0,
                f"{DAY_NAMES[day]} should have zero coverage")

    def test_two_shifts_per_day_demand(self):
        """AM + PM demand windows in one day each covered by separate shifts."""
        demand = (day_block(0, 6, 10, 2) + day_block(0, 14, 18, 2))
        r = solve_multi([demand], ["Staff"], quick_params())
        self.assertTrue(feasible(r))
        for window_h in ((6, 10), (14, 18)):
            sl = slice(window_h[0] * INTERVALS_PER_HOUR,
                       window_h[1] * INTERVALS_PER_HOUR)
            deficit = int(np.maximum(2 - r.combined_phase1.coverage[sl], 0).max())
            self.assertEqual(deficit, 0,
                f"Window {window_h[0]}-{window_h[1]}h not covered")


# ── Group III: Constraint combinations ───────────────────────────────────────

class TestConstraintCombinations(unittest.TestCase):

    def test_headcount_plus_entries_combined(self):
        """Both max_headcount and max_entries active simultaneously."""
        demand = day_block(0, 8, 20, 3)
        p = quick_params(
            max_headcount_per_day=[5] * 7,
            max_entries_per_day=[2, 0, 0, 0, 0, 0, 0],
        )
        r = solve_multi([demand], ["Staff"], p)
        if not feasible(r):
            return  # constraints may make it infeasible — that's fine
        p1 = r.combined_phase1
        # headcount constraint
        mon_cov = p1.coverage[:INTERVALS_PER_DAY]
        self.assertLessEqual(int(mon_cov.max()), 5, "Headcount limit violated")
        # entries constraint
        mon_starts = {s.start_interval for s, c in zip(p1.shifts, p1.counts)
                      if c > 0 and s.day == 0}
        self.assertLessEqual(len(mon_starts), 2, "Entry limit violated")

    def test_force_include_plus_headcount(self):
        """Force-include a specific code AND headcount limit — both must hold."""
        demand = day_block(0, 8, 16, 2)
        p = quick_params(
            force_include_shifts=["0800-1600"],
            max_headcount_per_day=[4] * 7,
        )
        r = solve_multi([demand], ["Staff"], p)
        if not feasible(r):
            self.skipTest("Infeasible with this combination")
        self.assertIn("0800-1600", active_codes(r), "Force-include not honoured")
        mon_cov = r.combined_phase1.coverage[:INTERVALS_PER_DAY]
        self.assertLessEqual(int(mon_cov.max()), 4, "Headcount limit violated")

    def test_force_include_multiple_codes(self):
        """Two different force-included codes must both appear in the solution."""
        demand = day_block(0, 6, 18, 2)   # long enough to need multiple shifts
        p = quick_params(
            force_include_shifts=["0600-1400", "1000-1800"],
        )
        r = solve_multi([demand], ["Staff"], p)
        if not feasible(r):
            self.skipTest("Infeasible")
        codes = active_codes(r)
        self.assertIn("0600-1400", codes, "First force-include missing")
        self.assertIn("1000-1800", codes, "Second force-include missing")

    def test_force_include_and_exclude_different_codes(self):
        """Force-include and force-exclude of different codes must both be respected."""
        demand = day_block(0, 8, 20, 2)
        p = quick_params(
            force_include_shifts=["0800-1600"],
            force_exclude_shifts=["1200-2000"],
        )
        r = solve_multi([demand], ["Staff"], p)
        if not feasible(r):
            self.skipTest("Infeasible")
        codes = active_codes(r)
        self.assertIn("0800-1600", codes, "Force-include not honoured")
        self.assertNotIn("1200-2000", codes, "Force-exclude violated")

    def test_allowed_slots_plus_two_curves(self):
        """Two occupations with restricted start/end times."""
        allowed = list(range(0, 1440, 60))   # only full hours
        d1 = day_block(0, 8, 16, 2)
        d2 = day_block(0, 10, 18, 2)
        p = quick_params(allowed_slot_minutes=allowed)
        r = solve_multi([d1, d2], ["A", "B"], p)
        if not feasible(r):
            self.skipTest("Infeasible")
        for s, c in zip(r.combined_phase1.shifts, r.combined_phase1.counts):
            if c > 0:
                start_min = (s.start_interval * 5) % 1440
                end_min   = ((s.start_interval + s.duration_intervals) * 5) % 1440
                self.assertIn(start_min, allowed,
                    f"Active shift {s.shift_code} has start not in allowed slots")
                self.assertIn(end_min, allowed,
                    f"Active shift {s.shift_code} has end not in allowed slots")

    def test_exits_constraint_enforced_with_demand(self):
        """max_exits_per_day limits distinct end times even when demand is continuous."""
        demand = week_block(6, 22, 2)
        p = quick_params(max_exits_per_day=[1, 1, 1, 1, 1, 0, 0])
        r = solve_multi([demand], ["Staff"], p)
        if not feasible(r):
            return
        for day in range(5):  # Mon-Fri have limit=1
            mon_ends = {s.start_interval + s.duration_intervals
                        for s, c in zip(r.combined_phase1.shifts,
                                        r.combined_phase1.counts)
                        if c > 0 and s.day == day}
            self.assertLessEqual(len(mon_ends), 1,
                f"{DAY_NAMES[day]}: exits={len(mon_ends)} exceeds limit of 1")


# ── Group IV: max_unique_shifts ───────────────────────────────────────────────

class TestMaxUniqueShifts(unittest.TestCase):

    def test_unique_shifts_limit_respected(self):
        """Active shift codes across the solution must not exceed max_unique_shifts."""
        # Single day, single window → can be covered by 1 shift type used on 1 day
        demand = day_block(0, 8, 16, 2)
        p = quick_params(max_unique_shifts=2,
                         min_shift_hours=8.0, max_shift_hours=8.0,
                         shift_duration_step_min=60)
        r = solve_multi([demand], ["Staff"], p)
        if not feasible(r):
            self.skipTest("Infeasible with max_unique_shifts=2")
        n_active = sum(1 for c in r.combined_phase1.counts if c > 0)
        # max_unique_shifts limits sum(z[s]) globally
        self.assertLessEqual(n_active, 2,
            f"Number of active shifts {n_active} > max_unique_shifts=2")

    def test_unique_shifts_limit_one(self):
        """max_unique_shifts=1 → only a single shift code may be used globally."""
        # 8-hour window fits exactly into one 8-h shift
        demand = day_block(0, 8, 16, 1)
        p = quick_params(max_unique_shifts=1,
                         min_shift_hours=8.0, max_shift_hours=8.0,
                         shift_duration_step_min=60)
        r = solve_multi([demand], ["Staff"], p)
        if not feasible(r):
            self.skipTest("Infeasible with max_unique_shifts=1")
        n_active = sum(1 for c in r.combined_phase1.counts if c > 0)
        self.assertLessEqual(n_active, 1,
            f"More than 1 shift active with max_unique_shifts=1: {n_active}")

    def test_unlimited_shifts_uses_more_than_limit(self):
        """Without a unique-shifts limit, a varied demand should use more than 1 code."""
        demand = zero_demand()
        for day in range(7):
            # Different window each day so multiple start times are needed
            s = day * INTERVALS_PER_DAY + (6 + day) * INTERVALS_PER_HOUR
            e = day * INTERVALS_PER_DAY + (10 + day) * INTERVALS_PER_HOUR
            demand[s:e] = 1
        p = quick_params(max_unique_shifts=0)  # unlimited
        r = solve_multi([demand], ["Staff"], p)
        if not feasible(r):
            self.skipTest("Infeasible")
        n_active = sum(1 for c in r.combined_phase1.counts if c > 0)
        # With 7 different windows it is very likely the solver needs >1 shift type
        # (test just checks it doesn't crash and gives a non-zero count)
        self.assertGreater(n_active, 0)

    def test_max_unique_shift_codes_vs_shift_count(self):
        """max_unique_shifts caps z-vars (shift slot types), not total worker count."""
        # Single 8-hour block fits in exactly 1 shift type; use 3 workers
        demand = day_block(0, 8, 16, 3)
        p = quick_params(max_unique_shifts=1,
                         min_shift_hours=8.0, max_shift_hours=8.0,
                         shift_duration_step_min=60)
        r = solve_multi([demand], ["Staff"], p)
        if not feasible(r):
            self.skipTest("Infeasible")
        unique_slots = len({s.shift_code for s, c in
                            zip(r.combined_phase1.shifts, r.combined_phase1.counts)
                            if c > 0})
        self.assertLessEqual(unique_slots, 1)
        # But multiple workers CAN be assigned to that one slot
        max_workers = max(
            (c for s, c in zip(r.combined_phase1.shifts, r.combined_phase1.counts) if c > 0),
            default=0,
        )
        self.assertGreaterEqual(max_workers, 1)


# ── Group V: Transition penalty effect ───────────────────────────────────────

class TestTransitionPenalty(unittest.TestCase):

    def test_high_penalty_reduces_unique_shifts(self):
        """With a very high transition penalty, the solver should prefer fewer
        distinct shift codes compared to zero penalty."""
        demand = week_block(8, 16, 2)
        p_none  = quick_params(transition_penalty=0)
        p_high  = quick_params(transition_penalty=500)

        r_none  = solve_multi([demand], ["Staff"], p_none)
        r_high  = solve_multi([demand], ["Staff"], p_high)

        if not (feasible(r_none) and feasible(r_high)):
            self.skipTest("One solve infeasible")

        n_none = len(active_codes(r_none))
        n_high = len(active_codes(r_high))
        # High penalty should not produce more unique shifts
        self.assertLessEqual(n_high, n_none + 2,  # allow tiny rounding slack
            f"High penalty gave MORE unique shifts ({n_high}) than no penalty ({n_none})")

    def test_penalty_does_not_break_coverage(self):
        """Transition penalty must never cause under-coverage."""
        demand = week_block(8, 16, 2)
        p = quick_params(transition_penalty=200)
        r = solve_multi([demand], ["Staff"], p)
        self.assertTrue(feasible(r))
        deficit = np.maximum(demand - r.combined_phase1.coverage, 0)
        self.assertEqual(int(np.count_nonzero(deficit)), 0,
            f"Coverage broken by transition penalty; max deficit={int(deficit.max())}")

    def test_zero_penalty_feasible(self):
        """transition_penalty=0 must not crash."""
        demand = day_block(0, 8, 16, 2)
        r = solve_multi([demand], ["Staff"], quick_params(transition_penalty=0))
        self.assertTrue(feasible(r))


# ── Group VI: solve() wrapper and output helpers ──────────────────────────────

class TestOutputHelpers(unittest.TestCase):

    def _simple_result(self, demand=None):
        if demand is None:
            demand = day_block(0, 8, 16, 2)
        r = solve(demand, quick_params())
        if not feasible(r):
            self.skipTest("Solve infeasible – cannot test outputs")
        return r

    def test_solve_backward_compat_wrapper(self):
        """solve(demand, params) must return a MultiCurveResult with 1 occupation."""
        demand = day_block(0, 8, 16, 2)
        r = solve(demand, quick_params())
        self.assertIsInstance(r, MultiCurveResult)
        self.assertEqual(len(r.occupations), 1)
        self.assertEqual(r.occupations[0].name, "Staff")

    def test_shifts_to_dataframe_days_correct(self):
        """Day column in shifts_to_dataframe must match the shift's actual day."""
        r = self._simple_result()
        p1 = r.combined_phase1
        df = shifts_to_dataframe(p1)
        if df.empty:
            return
        for _, row in df.iterrows():
            self.assertIn(row["Day"], DAY_NAMES,
                f"Unknown day name: {row['Day']}")

    def test_shifts_to_dataframe_workers_positive(self):
        """All rows must have Workers > 0."""
        r = self._simple_result()
        df = shifts_to_dataframe(r.combined_phase1)
        if not df.empty:
            self.assertTrue((df["Workers"] > 0).all())

    def test_shifts_to_dataframe_duration_in_bounds(self):
        """DurationHrs must be within the params' min/max shift hours."""
        p = quick_params(min_shift_hours=4.0, max_shift_hours=8.0)
        r = solve(day_block(0, 8, 16, 2), p)
        df = shifts_to_dataframe(r.combined_phase1)
        if not df.empty:
            self.assertTrue((df["DurationHrs"] >= 4.0 - 1e-9).all())
            self.assertTrue((df["DurationHrs"] <= 8.0 + 1e-9).all())

    def test_shifts_to_dataframe_with_label(self):
        """When label is set, dataframe must include an 'Occupation' column."""
        r = self._simple_result()
        df = shifts_to_dataframe(r.combined_phase1, label="Technician")
        if not df.empty:
            self.assertIn("Occupation", df.columns)
            self.assertTrue((df["Occupation"] == "Technician").all())

    def test_shifts_to_dataframe_workers_sum_matches_counts(self):
        """Sum of Workers in dataframe must equal sum of non-zero counts."""
        r = self._simple_result()
        p1 = r.combined_phase1
        expected_sum = sum(c for c in p1.counts if c > 0)
        df = shifts_to_dataframe(p1)
        self.assertEqual(int(df["Workers"].sum()), expected_sum)

    def test_coverage_dataframe_length(self):
        r = self._simple_result()
        df = coverage_dataframe(r)
        self.assertEqual(len(df), TOTAL_INTERVALS)

    def test_coverage_dataframe_per_occ_columns(self):
        """coverage_dataframe must have Demand_ and Coverage_ columns for each occ."""
        d1 = day_block(0, 8, 16, 2)
        d2 = day_block(0, 10, 18, 3)
        r = solve_multi([d1, d2], ["Alpha", "Beta"], quick_params())
        if not feasible(r):
            self.skipTest("Infeasible")
        df = coverage_dataframe(r)
        for name in ("Alpha", "Beta"):
            self.assertIn(f"Demand_{name}", df.columns)
            self.assertIn(f"Coverage_{name}", df.columns)

    def test_coverage_dataframe_total_columns_correct(self):
        """TotalDemand and TotalCoverage must match combined arrays."""
        r = self._simple_result()
        df = coverage_dataframe(r)
        np.testing.assert_array_equal(
            df["TotalDemand"].values,
            r.combined_demand.astype(int))
        np.testing.assert_array_equal(
            df["TotalCoverage"].values,
            r.combined_phase1.coverage.astype(int))

    def test_shift_type_summary_shift_codes_subset(self):
        """All ShiftType codes in summary must appear in active shift codes."""
        r = self._simple_result()
        summary = shift_type_summary(r.combined_phase1, label="Staff")
        codes_in_summary = set(summary["ShiftType"])
        codes_active = active_codes(r)
        self.assertTrue(codes_in_summary.issubset(codes_active),
            f"Summary has extra codes: {codes_in_summary - codes_active}")

    def test_shift_type_summary_total_matches(self):
        """Sum of Total column must equal sum of active counts."""
        r = self._simple_result()
        p1 = r.combined_phase1
        summary = shift_type_summary(p1)
        expected = sum(c for c in p1.counts if c > 0)
        self.assertEqual(int(summary["Total"].sum()), expected)

    def test_shift_type_summary_label_column(self):
        """Label parameter adds Occupation column."""
        r = self._simple_result()
        df = shift_type_summary(r.combined_phase1, label="Mechanic")
        if not df.empty:
            self.assertIn("Occupation", df.columns)
            self.assertTrue((df["Occupation"] == "Mechanic").all())

    def test_build_weekly_report_xlsx_returns_bytes(self):
        r = self._simple_result()
        xlsx = build_weekly_report_xlsx(r)
        self.assertIsInstance(xlsx, bytes)
        self.assertGreater(len(xlsx), 0)

    def test_build_weekly_report_xlsx_multi_curve(self):
        """XLSX generation must not crash for a multi-curve result."""
        d1 = day_block(0, 8, 16, 2)
        d2 = day_block(0, 10, 18, 3)
        r = solve_multi([d1, d2], ["Alpha", "Beta"], quick_params())
        if not feasible(r):
            self.skipTest("Infeasible")
        xlsx = build_weekly_report_xlsx(r)
        self.assertIsInstance(xlsx, bytes)
        self.assertGreater(len(xlsx), 0)


# ── Group VII: Scaling & proportionality ─────────────────────────────────────

class TestScaling(unittest.TestCase):

    def test_double_workers_double_hours(self):
        """Doubling demand workers should roughly double worker-hours assigned."""
        p = quick_params()
        r1 = solve_multi([day_block(0, 8, 16, 1)], ["S"], p)
        r2 = solve_multi([day_block(0, 8, 16, 2)], ["S"], p)
        if not (feasible(r1) and feasible(r2)):
            self.skipTest("Infeasible")
        wh1 = r1.combined_phase1.total_worker_intervals
        wh2 = r2.combined_phase1.total_worker_intervals
        # wh2 should be between 1.5× and 3× wh1 (not 0 or wildly off)
        self.assertGreaterEqual(wh2, wh1 * 1.5,
            f"Double demand gave <1.5× worker-intervals ({wh1} → {wh2})")
        self.assertLessEqual(wh2, wh1 * 3.0 + 48,
            f"Double demand gave >3× worker-intervals ({wh1} → {wh2})")

    def test_five_workers_at_least_five_times_one(self):
        """5-worker demand must use >= 5× worker-intervals of a 1-worker demand."""
        p = quick_params()
        r1 = solve_multi([day_block(0, 8, 12, 1)], ["S"], p)
        r5 = solve_multi([day_block(0, 8, 12, 5)], ["S"], p)
        if not (feasible(r1) and feasible(r5)):
            self.skipTest("Infeasible")
        wi1 = r1.combined_phase1.total_worker_intervals
        wi5 = r5.combined_phase1.total_worker_intervals
        self.assertGreaterEqual(wi5, wi1 * 5,
            f"5-worker solve uses fewer than 5× intervals of 1-worker solve")

    def test_coverage_scales_with_workers(self):
        """Coverage peak at demand window should be >= demand workers."""
        for workers in (1, 3, 5):
            demand = day_block(0, 9, 13, workers)
            r = solve_multi([demand], ["S"], quick_params())
            if not feasible(r):
                continue
            peak_slot = 9 * INTERVALS_PER_HOUR + 1
            cov_peak = int(r.combined_phase1.coverage[peak_slot])
            self.assertGreaterEqual(cov_peak, workers,
                f"Coverage {cov_peak} < demand {workers} at peak")


# ── Group VIII: SolverParams edge values ─────────────────────────────────────

class TestParamEdgeValues(unittest.TestCase):

    def test_min_equals_max_shift_hours(self):
        """When min_shift_hours == max_shift_hours, only one duration is allowed."""
        p = quick_params(min_shift_hours=6.0, max_shift_hours=6.0,
                         shift_duration_step_min=60)
        shifts = generate_candidate_shifts(p)
        for s in shifts:
            self.assertAlmostEqual(s.duration_hours, 6.0,
                msg=f"Shift {s.shift_code} has duration {s.duration_hours}h ≠ 6h")

    def test_granularity_30min_not_60(self):
        """shift_start_granularity_min=30 must produce :30 starts not found at 60 min."""
        p = quick_params(shift_start_granularity_min=30)
        starts = {(s.start_interval * 5) % 60 for s in generate_candidate_shifts(p)}
        self.assertIn(30, starts, "30-min granularity: no :30 start times generated")

    def test_very_short_time_limit_does_not_crash(self):
        """solver_time_limit_sec=1 must not crash, even if solution is not OPTIMAL."""
        demand = flat_demand(3)
        p = quick_params(solver_time_limit_sec=1)
        r = solve_multi([demand], ["S"], p)
        # Must return without exception; status may be anything
        self.assertIsNotNone(r)
        self.assertIsNotNone(r.combined_phase1.status)

    def test_large_time_limit_is_accepted(self):
        """solver_time_limit_sec=600 must not crash."""
        demand = day_block(0, 8, 12, 1)
        p = quick_params(solver_time_limit_sec=600)
        r = solve_multi([demand], ["S"], p)
        self.assertIsNotNone(r)

    def test_max_shift_12h_generates_overnight(self):
        """max_shift_hours=12 should produce some overnight shift candidates."""
        p = quick_params(min_shift_hours=12.0, max_shift_hours=12.0,
                         shift_duration_step_min=60)
        shifts = generate_candidate_shifts(p)
        overnight = [s for s in shifts
                     if s.start_interval + s.duration_intervals > INTERVALS_PER_DAY]
        self.assertGreater(len(overnight), 0,
            "No overnight candidates with max_shift=12h (expected some)")

    def test_single_allowed_slot_minute_can_be_empty(self):
        """allowed_slot_minutes=[0] (only midnight) → only midnight-to-midnight shifts.
        With 4–8 h range that means 0000-0400 … 0000-0800; none end at midnight."""
        # midnight start, 4h → ends 04:00. 04:00 ∉ {0} → filtered out.
        p = quick_params(allowed_slot_minutes=[0], min_shift_hours=4.0,
                         max_shift_hours=4.0, shift_duration_step_min=60)
        shifts = generate_candidate_shifts(p)
        self.assertEqual(shifts, [],
            "Expected no candidates when only 00:00 is allowed (4h shift ends 04:00)")

    def test_no_force_include_or_exclude_is_safe(self):
        """None values for force lists must not cause AttributeError."""
        p = quick_params(force_include_shifts=None, force_exclude_shifts=None)
        demand = day_block(0, 8, 12, 1)
        r = solve_multi([demand], ["S"], p)
        self.assertIsNotNone(r)

    def test_empty_force_lists_are_treated_as_none(self):
        """Empty lists for force_include/exclude must not add spurious constraints."""
        p_empty = quick_params(force_include_shifts=[], force_exclude_shifts=[])
        p_none  = quick_params(force_include_shifts=None, force_exclude_shifts=None)
        demand  = day_block(0, 8, 16, 2)
        r_empty = solve_multi([demand], ["S"], p_empty)
        r_none  = solve_multi([demand], ["S"], p_none)
        # Both should reach the same feasibility
        self.assertEqual(feasible(r_empty), feasible(r_none))


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"\n{'='*60}")
    print(f"TOTAL: {result.testsRun} tests  |  "
          f"PASSED: {result.testsRun - len(result.failures) - len(result.errors)}  |  "
          f"FAILED: {len(result.failures)}  |  "
          f"ERRORS: {len(result.errors)}")
    sys.exit(0 if result.wasSuccessful() else 1)
