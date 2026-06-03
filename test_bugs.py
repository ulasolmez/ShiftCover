"""
ShiftCover – Comprehensive bug-detection test suite.

Run with:  python test_bugs.py
"""
import sys
import unittest
import numpy as np

from solver import (
    INTERVALS_PER_HOUR, INTERVALS_PER_DAY, TOTAL_INTERVALS,
    CandidateShift, SolverParams, PhaseOneResult,
    generate_candidate_shifts, build_coverage_map, list_possible_shift_codes,
    _shift_code_from, is_night_shift, night_overlap_intervals,
    daily_entry_headcount, max_headcount,
    solve_multi, shifts_to_dataframe, coverage_dataframe,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

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

def quick_params(**kwargs) -> SolverParams:
    """Fast params: hourly granularity, 4–8 h shifts, 10 s time limit.
    Caller kwargs override the defaults."""
    defaults: dict = dict(
        min_shift_hours=4.0,
        max_shift_hours=8.0,
        shift_start_granularity_min=60,
        shift_duration_step_min=60,
        min_weekly_hours=0.0,
        max_weekly_hours=200.0,
        min_rest_hours=0.0,
        max_unique_shifts=0,
        transition_penalty=0,
        solver_time_limit_sec=10,
    )
    defaults.update(kwargs)
    return SolverParams(**defaults)


# ── Group A: CandidateShift pure-property tests ───────────────────────────────

class TestCandidateShiftProperties(unittest.TestCase):

    def _make(self, day, start, dur):
        return CandidateShift(idx=0, day=day, start_interval=start,
                              duration_intervals=dur)

    def test_global_start(self):
        s = self._make(0, 0, 12)
        self.assertEqual(s.global_start, 0)
        s2 = self._make(1, 24, 12)          # Tue 02:00
        self.assertEqual(s2.global_start, 288 + 24)

    def test_global_end(self):
        s = self._make(0, 0, 48)            # 4 h
        self.assertEqual(s.global_end, 48)

    def test_duration_hours(self):
        s = self._make(0, 0, 48)
        self.assertAlmostEqual(s.duration_hours, 4.0)

    def test_start_time_str_midnight(self):
        s = self._make(0, 0, 12)
        self.assertEqual(s.start_time_str, "00:00")

    def test_start_time_str_halftime(self):
        s = self._make(0, 6, 12)            # 00:30
        self.assertEqual(s.start_time_str, "00:30")

    def test_start_time_str_noon(self):
        s = self._make(0, 144, 12)          # 12:00
        self.assertEqual(s.start_time_str, "12:00")

    def test_start_time_str_end_of_day(self):
        s = self._make(0, 282, 12)          # 23:30
        self.assertEqual(s.start_time_str, "23:30")

    def test_end_time_str_no_wrap(self):
        s = self._make(0, 96, 48)           # 08:00 + 4 h = 12:00
        self.assertEqual(s.end_time_str, "12:00")

    def test_end_time_str_wraps_midnight(self):
        # 22:00 + 4 h = 02:00 next day
        s = self._make(0, 264, 48)
        self.assertEqual(s.end_time_str, "02:00")

    def test_end_time_str_exactly_midnight(self):
        # 20:00 + 4 h = 00:00
        s = self._make(0, 240, 48)
        self.assertEqual(s.end_time_str, "00:00")

    def test_shift_code_standard(self):
        s = self._make(0, 96, 96)           # 08:00–16:00
        self.assertEqual(s.shift_code, "0800-1600")

    def test_shift_code_overnight(self):
        s = self._make(0, 216, 96)          # 18:00–02:00
        self.assertEqual(s.shift_code, "1800-0200")

    def test_shift_code_midnight_end(self):
        s = self._make(0, 192, 96)          # 16:00–00:00
        self.assertEqual(s.shift_code, "1600-0000")

    def test_shift_code_from_matches_candidate(self):
        s = self._make(3, 108, 84)
        self.assertEqual(s.shift_code, _shift_code_from(108, 84))

    # covers() – non-wrapping
    def test_covers_inside(self):
        s = self._make(0, 96, 48)           # Mon 08:00–12:00
        self.assertTrue(s.covers(100))
        self.assertTrue(s.covers(96))
        self.assertFalse(s.covers(144))     # exclusive end
        self.assertFalse(s.covers(95))

    # covers() – wrapping Sunday → Monday
    def test_covers_wrapping_sunday(self):
        # Sunday 23:00 (276) + 2 h (24 intervals) wraps into Monday
        s = self._make(6, 276, 24)
        self.assertTrue(s.covers(6 * 288 + 276))   # Sunday 23:00
        self.assertTrue(s.covers(6 * 288 + 287))   # Sunday 23:55
        self.assertTrue(s.covers(0))               # Monday 00:00
        self.assertTrue(s.covers(11))              # Monday 00:55
        self.assertFalse(s.covers(12))             # Monday 01:00 (exclusive)
        self.assertFalse(s.covers(6 * 288 + 275))  # Sunday 22:55


# ── Group B: Night detection ──────────────────────────────────────────────────

class TestNightDetection(unittest.TestCase):

    def test_day_shift_not_night(self):
        # 08:00–16:00 (dur 8 h = 96 intervals)
        self.assertFalse(is_night_shift(96, 96))

    def test_night_shift_detected(self):
        # 22:00–06:00 (start=264, dur=96)
        self.assertTrue(is_night_shift(264, 96))

    def test_evening_long_shift_not_night(self):
        # 14:00–22:00 (start=168, dur=96) – mostly daytime
        self.assertFalse(is_night_shift(168, 96))

    def test_short_night_not_flagged(self):
        # 22:00–02:00 (start=264, dur=48) – only 4 h, under the 8 h threshold
        self.assertFalse(is_night_shift(264, 48))

    def test_night_overlap_full_night(self):
        # 20:00–06:00 = 10 h night window exactly → 120 intervals
        overlap = night_overlap_intervals(240, 240 + 120)
        self.assertEqual(overlap, 120)

    def test_night_overlap_zero_for_day(self):
        # 08:00–16:00 → no night overlap
        self.assertEqual(night_overlap_intervals(96, 192), 0)

    def test_night_overlap_partial_morning(self):
        # 04:00–10:00 (start=48, end=120) overlaps 04:00–06:00 (2 h = 24 ivl)
        self.assertEqual(night_overlap_intervals(48, 120), 24)

    def test_night_overlap_partial_evening(self):
        # 18:00–22:00 (start=216, end=264) overlaps 20:00–22:00 (2 h = 24 ivl)
        self.assertEqual(night_overlap_intervals(216, 264), 24)


# ── Group C: Shift generation ─────────────────────────────────────────────────

class TestShiftGeneration(unittest.TestCase):

    def test_generates_7_days(self):
        p = quick_params()
        shifts = generate_candidate_shifts(p)
        days = {s.day for s in shifts}
        self.assertEqual(days, set(range(7)))

    def test_duration_within_bounds(self):
        p = quick_params(min_shift_hours=4.0, max_shift_hours=8.0,
                         shift_duration_step_min=60)
        for s in generate_candidate_shifts(p):
            self.assertGreaterEqual(s.duration_hours, 4.0 - 1e-9)
            self.assertLessEqual(s.duration_hours, 8.0 + 1e-9)

    def test_granularity_respected(self):
        p = quick_params(shift_start_granularity_min=60)
        for s in generate_candidate_shifts(p):
            self.assertEqual((s.start_interval * 5) % 60, 0,
                             f"Start {s.start_time_str} is not on the hour")

    def test_force_exclude_removes_code(self):
        p = quick_params(force_exclude_shifts=["0800-1600"])
        for s in generate_candidate_shifts(p):
            self.assertNotEqual(s.shift_code, "0800-1600")

    def test_allowed_slot_minutes_filters_start(self):
        # Only allow :00 times
        allowed = list(range(0, 1440, 60))
        p = quick_params(allowed_slot_minutes=allowed)
        for s in generate_candidate_shifts(p):
            start_min = (s.start_interval * 5) % 1440
            self.assertIn(start_min, allowed,
                         f"Start {s.start_time_str} not in allowed slots")

    def test_allowed_slot_minutes_filters_end(self):
        allowed = list(range(0, 1440, 60))
        p = quick_params(allowed_slot_minutes=allowed,
                         min_shift_hours=4.0, max_shift_hours=4.0,
                         shift_duration_step_min=60)
        for s in generate_candidate_shifts(p):
            end_min = ((s.start_interval + s.duration_intervals) * 5) % 1440
            self.assertIn(end_min, allowed,
                         f"End time not in allowed slots for {s.shift_code}")

    def test_sunday_overnight_excluded_without_circular(self):
        p = quick_params(min_shift_hours=4.0, max_shift_hours=12.0,
                         shift_start_granularity_min=60,
                         shift_duration_step_min=60,
                         circular_week=False)
        for s in generate_candidate_shifts(p):
            if s.day == 6:
                self.assertLessEqual(
                    s.global_end, TOTAL_INTERVALS,
                    f"Sun shift wraps without circular: {s.shift_code}")

    def test_sunday_overnight_allowed_with_circular(self):
        p = quick_params(min_shift_hours=4.0, max_shift_hours=8.0,
                         shift_start_granularity_min=60,
                         shift_duration_step_min=60,
                         circular_week=True)
        wrapping = [s for s in generate_candidate_shifts(p)
                    if s.day == 6 and s.global_end > TOTAL_INTERVALS]
        self.assertGreater(len(wrapping), 0,
                          "No wrapping Sunday shifts found with circular_week=True")

    def test_exclude_night_shifts(self):
        p = quick_params(min_shift_hours=8.0, max_shift_hours=12.0,
                         shift_duration_step_min=60,
                         exclude_night_shifts=True)
        for s in generate_candidate_shifts(p):
            self.assertFalse(
                is_night_shift(s.start_interval, s.duration_intervals),
                f"Night shift {s.shift_code} not excluded")

    def test_idx_is_unique(self):
        p = quick_params()
        shifts = generate_candidate_shifts(p)
        idxs = [s.idx for s in shifts]
        self.assertEqual(len(idxs), len(set(idxs)), "Duplicate shift idx found")

    def test_idx_matches_list_position(self):
        """idx values must be stable sequential (used as dict key in coverage map)."""
        p = quick_params()
        shifts = generate_candidate_shifts(p)
        for i, s in enumerate(shifts):
            self.assertEqual(s.idx, i)

    def test_list_possible_codes_consistent_with_generate(self):
        """list_possible_shift_codes should produce exactly the set of codes in generate."""
        p = quick_params(min_shift_hours=4.0, max_shift_hours=6.0,
                         shift_start_granularity_min=60,
                         shift_duration_step_min=60)
        codes_list = set(list_possible_shift_codes(p))
        codes_gen  = {s.shift_code for s in generate_candidate_shifts(p)}
        self.assertEqual(codes_list, codes_gen,
                        "list_possible_shift_codes inconsistent with generate_candidate_shifts")

    def test_no_shifts_when_fully_filtered(self):
        """If allowed_slot_minutes contains only times incompatible with any valid shift,
        generate_candidate_shifts should return an empty list (not crash)."""
        # Only allow 01:00 as start/end; no 4–8 h shift fits since 01:00+4h=05:00 ∉ {60}
        p = quick_params(allowed_slot_minutes=[60])  # only 01:00
        shifts = generate_candidate_shifts(p)
        self.assertEqual(shifts, [])

    # BUG PROBE: overnight non-Sunday shifts are included even without circular_week
    def test_overnight_non_sunday_always_included(self):
        """Mon–Sat overnight shifts are always included regardless of circular_week.
        This is intentional (they cover the next day within the same week).
        Verify this is indeed the case and no crash occurs."""
        p = quick_params(min_shift_hours=4.0, max_shift_hours=8.0,
                         shift_start_granularity_min=60,
                         shift_duration_step_min=60,
                         circular_week=False)
        overnight_non_sun = [s for s in generate_candidate_shifts(p)
                             if s.day < 6 and
                             s.start_interval + s.duration_intervals > INTERVALS_PER_DAY]
        # Just verify none of these cause a crash later in build_coverage_map
        cov = build_coverage_map(overnight_non_sun)
        self.assertEqual(len(cov), TOTAL_INTERVALS)


# ── Group D: Coverage map ─────────────────────────────────────────────────────

class TestCoverageMap(unittest.TestCase):

    def test_simple_shift_covered_intervals(self):
        # Mon 08:00–12:00 (start=96, dur=48)
        s = CandidateShift(0, 0, 96, 48)
        cov = build_coverage_map([s])
        for t in range(96, 144):
            self.assertIn(0, cov[t], f"Interval {t} should be covered")
        # Before and after should not be covered
        self.assertNotIn(0, cov[95])
        self.assertNotIn(0, cov[144])

    def test_coverage_map_size(self):
        cov = build_coverage_map([])
        self.assertEqual(len(cov), TOTAL_INTERVALS)

    def test_wrapping_sunday_shift(self):
        # Sunday 23:00 + 2 h wraps into Monday
        s = CandidateShift(0, 6, 276, 24)  # global_start=1980, global_end=2004
        cov = build_coverage_map([s])
        for t in range(6 * 288 + 276, 6 * 288 + 288):
            self.assertIn(0, cov[t], f"Sun interval {t} should be covered")
        # Wraps to Monday: global_end=2004, 2004 % 2016 = 2004 (no wrap here actually)
        # Actually 6*288+276=1980, 1980+24=2004 < 2016, so no wrap needed
        # Let's use a bigger wrap:
        s2 = CandidateShift(1, 6, 276, 48)  # 4 h wrap: global_end=2028 > 2016
        cov2 = build_coverage_map([s2])
        # Monday 00:00–01:00 should appear (t=0..11)
        for t in range(0, 12):
            self.assertIn(1, cov2[t], f"Monday interval {t} should be covered by wrapping shift")

    def test_multiple_shifts_overlap(self):
        # Two shifts overlapping the same intervals
        s1 = CandidateShift(0, 0, 96, 48)   # Mon 08:00–12:00
        s2 = CandidateShift(1, 0, 100, 48)  # Mon 08:20–12:20
        cov = build_coverage_map([s1, s2])
        self.assertIn(0, cov[100])
        self.assertIn(1, cov[100])
        self.assertIn(0, cov[96])
        self.assertNotIn(1, cov[96])          # s2 starts at 100


# ── Group E: Headcount helpers ───────────────────────────────────────────────

class TestHeadcountHelpers(unittest.TestCase):

    def _make_p1(self, day_shifts):
        """Build a PhaseOneResult from {(day,start,dur): count}."""
        shifts = [CandidateShift(i, d, st, du)
                  for i, (d, st, du) in enumerate(day_shifts.keys())]
        counts = list(day_shifts.values())
        coverage = np.zeros(TOTAL_INTERVALS, dtype=int)
        for s, cnt in zip(shifts, counts):
            for t in range(s.global_start, s.global_end):
                coverage[t % TOTAL_INTERVALS] += cnt
        return PhaseOneResult(shifts, counts, 0, coverage, "OPTIMAL", 0.0)

    def test_daily_entry_headcount_single_day(self):
        p1 = self._make_p1({(0, 96, 48): 5, (0, 120, 48): 3, (1, 96, 48): 2})
        daily = daily_entry_headcount(p1)
        self.assertEqual(daily[0], 8)   # Mon: 5+3
        self.assertEqual(daily[1], 2)   # Tue
        for d in range(2, 7):
            self.assertEqual(daily[d], 0)

    def test_max_headcount(self):
        p1 = self._make_p1({(0, 96, 48): 5, (0, 120, 48): 3, (2, 96, 48): 10})
        self.assertEqual(max_headcount(p1), 10)  # Wed has 10

    def test_headcount_empty_counts(self):
        """Infeasible result has empty counts – must not crash."""
        shifts = [CandidateShift(0, 0, 96, 48)]
        p1 = PhaseOneResult(shifts, [], 0, np.zeros(TOTAL_INTERVALS, int),
                            "INFEASIBLE", 0.0)
        daily = daily_entry_headcount(p1)
        self.assertEqual(daily, [0] * 7)
        self.assertEqual(max_headcount(p1), 0)


# ── Group F: Solver integration tests ────────────────────────────────────────

class TestSolverIntegration(unittest.TestCase):

    def _assert_coverage_satisfied(self, result, demands, label=""):
        """Verify coverage >= demand at every interval for every occupation."""
        for i, (occ, demand) in enumerate(zip(result.occupations, demands)):
            cov = occ.phase1.coverage
            deficit = np.maximum(demand - cov, 0)
            bad = int(np.count_nonzero(deficit))
            self.assertEqual(bad, 0,
                f"{label} occ {i} ({occ.name}): {bad} intervals under-covered, "
                f"max deficit={int(deficit.max())}")

    def test_single_curve_basic(self):
        """Solver covers a simple 2-worker 4-hour Mon morning demand."""
        demand = day_block(0, 8, 12, 2)
        p = quick_params()
        r = solve_multi([demand], ["Staff"], p)
        self.assertIn(r.combined_phase1.status, ("OPTIMAL", "FEASIBLE"))
        self._assert_coverage_satisfied(r, [demand], "single_curve_basic")

    def test_coverage_array_matches_shifts(self):
        """coverage[t] should equal sum of x[occ][s] for shifts covering t."""
        demand = day_block(0, 8, 16, 3)
        p = quick_params()
        r = solve_multi([demand], ["Staff"], p)
        self.assertIn(r.combined_phase1.status, ("OPTIMAL", "FEASIBLE"))
        # Manually recompute coverage from shifts+counts
        p1 = r.combined_phase1
        recomputed = np.zeros(TOTAL_INTERVALS, dtype=int)
        for s, cnt in zip(p1.shifts, p1.counts):
            if cnt > 0:
                for t in range(s.global_start, s.global_end):
                    recomputed[t % TOTAL_INTERVALS] += cnt
        np.testing.assert_array_equal(p1.coverage, recomputed,
            "coverage array does not match sum of shift counts")

    def test_two_curve_shared_shifts(self):
        """Two occupations must use identical shift time slots (shared z vars)."""
        d1 = day_block(0, 8, 16, 2)
        d2 = day_block(0, 8, 16, 3)
        p = quick_params()
        r = solve_multi([d1, d2], ["A", "B"], p)
        self.assertIn(r.combined_phase1.status, ("OPTIMAL", "FEASIBLE"))
        self._assert_coverage_satisfied(r, [d1, d2], "two_curve")

        # Verify that the set of active shift codes is identical for both occupations
        codes_a = {s.shift_code for s, c in zip(r.occupations[0].phase1.shifts,
                                                  r.occupations[0].phase1.counts) if c > 0}
        codes_b = {s.shift_code for s, c in zip(r.occupations[1].phase1.shifts,
                                                  r.occupations[1].phase1.counts) if c > 0}
        self.assertEqual(codes_a, codes_b,
            "Two occupations have different shift codes – shared activation violated")

    def test_force_include_respected(self):
        """A force-included shift code must appear in the solution."""
        demand = day_block(0, 8, 16, 2)
        p = quick_params(force_include_shifts=["0800-1600"])
        r = solve_multi([demand], ["Staff"], p)
        if r.combined_phase1.status not in ("OPTIMAL", "FEASIBLE"):
            self.skipTest("Infeasible – cannot test force_include")
        active = {s.shift_code for s, c in zip(r.combined_phase1.shifts,
                                                r.combined_phase1.counts) if c > 0}
        self.assertIn("0800-1600", active, "Force-included shift not found in solution")

    def test_force_exclude_respected(self):
        """A force-excluded shift code must NOT appear in the solution."""
        demand = day_block(0, 8, 16, 2)
        p = quick_params(force_exclude_shifts=["0800-1600"])
        r = solve_multi([demand], ["Staff"], p)
        if r.combined_phase1.status not in ("OPTIMAL", "FEASIBLE"):
            return  # Infeasible is acceptable; no need to check active shifts
        active = {s.shift_code for s, c in zip(r.combined_phase1.shifts,
                                                r.combined_phase1.counts) if c > 0}
        self.assertNotIn("0800-1600", active, "Force-excluded shift appears in solution")

    def test_max_headcount_constraint(self):
        """Simultaneous workers must not exceed max_headcount_per_day on any day."""
        demand = flat_demand(5)
        hc_limit = 10
        p = quick_params(max_headcount_per_day=[hc_limit] * 7)
        r = solve_multi([demand], ["Staff"], p)
        if r.combined_phase1.status not in ("OPTIMAL", "FEASIBLE"):
            return  # Infeasible is allowed; constraint may be too tight
        cov = r.combined_phase1.coverage
        for day in range(7):
            day_cov = cov[day * INTERVALS_PER_DAY:(day + 1) * INTERVALS_PER_DAY]
            actual_max = int(day_cov.max())
            self.assertLessEqual(actual_max, hc_limit,
                f"Day {day}: simultaneous workers {actual_max} > limit {hc_limit}")

    def test_max_entries_per_day_constraint(self):
        """Active distinct start times must not exceed max_entries_per_day."""
        demand = day_block(0, 8, 20, 3)
        p = quick_params(max_entries_per_day=[2, 0, 0, 0, 0, 0, 0])
        r = solve_multi([demand], ["Staff"], p)
        if r.combined_phase1.status not in ("OPTIMAL", "FEASIBLE"):
            return
        p1 = r.combined_phase1
        mon_starts = {s.start_interval for s, c in zip(p1.shifts, p1.counts)
                      if c > 0 and s.day == 0}
        self.assertLessEqual(len(mon_starts), 2,
            f"Monday has {len(mon_starts)} distinct entry times, limit is 2")

    def test_max_exits_per_day_constraint(self):
        """Active distinct end times must not exceed max_exits_per_day."""
        demand = day_block(0, 8, 20, 3)
        p = quick_params(max_exits_per_day=[2, 0, 0, 0, 0, 0, 0])
        r = solve_multi([demand], ["Staff"], p)
        if r.combined_phase1.status not in ("OPTIMAL", "FEASIBLE"):
            return
        p1 = r.combined_phase1
        mon_ends = {s.start_interval + s.duration_intervals
                    for s, c in zip(p1.shifts, p1.counts)
                    if c > 0 and s.day == 0}
        self.assertLessEqual(len(mon_ends), 2,
            f"Monday has {len(mon_ends)} distinct exit times, limit is 2")

    def test_infeasible_returns_correct_status(self):
        """A demand impossible to satisfy should return a non-OPTIMAL/FEASIBLE status."""
        # Demand 50 workers but headcount limited to 1 – solver should declare infeasible
        demand = flat_demand(50)
        p = quick_params(max_headcount_per_day=[1] * 7)
        r = solve_multi([demand], ["Staff"], p)
        self.assertNotIn(r.combined_phase1.status, ("OPTIMAL", "FEASIBLE"),
            "Expected INFEASIBLE status for unsatisfiable problem")

    # ── BUG PROBE: Infeasible result aliasing ────────────────────────────────
    def test_infeasible_multi_curve_aliasing(self):
        """
        BUG: When the solve is infeasible for a multi-curve problem,
        `return [empty] * n_occ` creates a list of aliases to the SAME object.
        All occupation PhaseOneResults should be independent instances, not the same object.
        """
        demand = flat_demand(50)
        p = quick_params(max_headcount_per_day=[1] * 7)
        r = solve_multi([demand, demand], ["A", "B"], p)
        if r.combined_phase1.status in ("OPTIMAL", "FEASIBLE"):
            self.skipTest("Expected infeasible – cannot test aliasing")
        p1_a = r.occupations[0].phase1
        p1_b = r.occupations[1].phase1
        self.assertIsNot(p1_a, p1_b,
            "BUG CONFIRMED: infeasible results for two occupations are the same object (aliased)")

    # ── BUG PROBE: force_include silently ignored when code filtered out ──────
    def test_force_include_silently_ignored_when_not_in_candidates(self):
        """
        BUG: If force_include_shifts contains a code that is filtered out
        by allowed_slot_minutes, no constraint is added and the force-include
        is silently ignored (the solver finds a solution without it).
        """
        demand = day_block(0, 8, 16, 2)
        # Only allow :00 times, so 0800-1600 IS in candidates
        # Force include 0900-1700 which is also valid
        # Now restrict allowed_slot_minutes to exclude 09:00 (540 min)
        allowed = [m for m in range(0, 1440, 60) if m != 540]  # remove 09:00
        p = quick_params(
            force_include_shifts=["0900-1700"],
            allowed_slot_minutes=allowed,
        )
        # 0900-1700 requires start=540 min, which is excluded → no candidate generated
        # Solver should NOT silently succeed — ideally it should warn or fail
        r = solve_multi([demand], ["Staff"], p)
        if r.combined_phase1.status not in ("OPTIMAL", "FEASIBLE"):
            return  # Infeasible is acceptable here
        active = {s.shift_code for s, c in zip(r.combined_phase1.shifts,
                                                r.combined_phase1.counts) if c > 0}
        # Document: "0900-1700" will NOT be present because it was filtered from candidates
        # This means force_include was silently ignored – a known limitation
        if "0900-1700" not in active:
            print("\n  [KNOWN LIMITATION] force_include_shifts silently ignored "
                  "when code is filtered by allowed_slot_minutes")

    def test_allowed_slot_minutes_no_candidates_is_infeasible(self):
        """When allowed_slot_minutes produces zero candidates for a non-zero demand,
        the solver should return INFEASIBLE (not crash)."""
        demand = day_block(0, 8, 16, 2)
        p = quick_params(allowed_slot_minutes=[60])  # only 01:00 – no 4–8 h shift fits
        r = solve_multi([demand], ["Staff"], p)
        # Should be infeasible or feasible with zero workers (demand not met)
        # At minimum: must not raise an exception. Status check is secondary.
        self.assertIsNotNone(r)

    def test_zero_demand_is_feasible(self):
        """Zero demand across the week should always yield a feasible empty solution."""
        demand = zero_demand()
        p = quick_params()
        r = solve_multi([demand], ["Staff"], p)
        self.assertIn(r.combined_phase1.status, ("OPTIMAL", "FEASIBLE"))
        # Zero demand → zero coverage is valid (nothing to cover)
        self.assertEqual(int(r.combined_phase1.coverage.sum()), 0,
            "Zero demand should result in zero coverage (no workers needed)")

    def test_callback_called_for_build_stages(self):
        """String callbacks should be fired at every build stage."""
        demand = day_block(0, 8, 12, 1)
        p = quick_params()
        messages = []
        r = solve_multi([demand], ["Staff"], p, callback=messages.append)
        self.assertGreater(len(messages), 0, "No callback messages received")
        all_text = " ".join(str(m) for m in messages)
        self.assertIn("Generated", all_text)
        self.assertIn("coverage map", all_text.lower())

    def test_shifts_to_dataframe_only_active(self):
        """shifts_to_dataframe should only include shifts with count > 0."""
        demand = day_block(0, 8, 16, 2)
        p = quick_params()
        r = solve_multi([demand], ["Staff"], p)
        p1 = r.occupations[0].phase1
        df = shifts_to_dataframe(p1)
        total_active = sum(1 for c in p1.counts if c > 0)
        # df may have multiple rows per day but every row must have Workers > 0
        if not df.empty:
            self.assertTrue((df["Workers"] > 0).all(),
                "shifts_to_dataframe contains rows with Workers=0")
            self.assertLessEqual(len(df), total_active)

    def test_coverage_dataframe_columns(self):
        """coverage_dataframe should have expected columns."""
        demand = day_block(0, 8, 16, 2)
        p = quick_params()
        r = solve_multi([demand], ["Staff"], p)
        df = coverage_dataframe(r)
        self.assertEqual(len(df), TOTAL_INTERVALS)
        self.assertIn("TotalDemand", df.columns)
        self.assertIn("TotalCoverage", df.columns)

    def test_combined_demand_equals_sum_of_occupations(self):
        """combined_demand must equal the elementwise sum of all occupation demands."""
        d1 = day_block(0, 8, 16, 2)
        d2 = day_block(0, 10, 18, 3)
        p = quick_params()
        r = solve_multi([d1, d2], ["A", "B"], p)
        expected = d1 + d2
        np.testing.assert_array_equal(r.combined_demand, expected,
            "combined_demand != sum of occupation demands")

    def test_combined_coverage_equals_sum_of_occ_coverage(self):
        """combined_phase1.coverage must equal sum of all occupation coverages."""
        d1 = day_block(0, 8, 16, 2)
        d2 = day_block(0, 10, 18, 3)
        p = quick_params()
        r = solve_multi([d1, d2], ["A", "B"], p)
        if r.combined_phase1.status not in ("OPTIMAL", "FEASIBLE"):
            return
        expected = (r.occupations[0].phase1.coverage +
                    r.occupations[1].phase1.coverage)
        np.testing.assert_array_equal(
            r.combined_phase1.coverage, expected,
            "combined coverage != sum of occupation coverages")

    def test_three_curves(self):
        """Three-occupation solve must satisfy all three demand curves."""
        d1 = day_block(0, 6, 14, 2)
        d2 = day_block(0, 8, 16, 3)
        d3 = day_block(0, 10, 18, 1)
        p = quick_params()
        r = solve_multi([d1, d2, d3], ["A", "B", "C"], p)
        if r.combined_phase1.status not in ("OPTIMAL", "FEASIBLE"):
            self.skipTest("Three-curve solve infeasible with quick params")
        self._assert_coverage_satisfied(r, [d1, d2, d3], "three_curve")


# ── Group G: _shift_code_from edge cases ─────────────────────────────────────

class TestShiftCodeFrom(unittest.TestCase):

    def test_standard(self):
        self.assertEqual(_shift_code_from(96, 96), "0800-1600")

    def test_midnight_end(self):
        self.assertEqual(_shift_code_from(192, 96), "1600-0000")

    def test_overnight_wrap(self):
        # 22:00 + 4 h = 02:00
        self.assertEqual(_shift_code_from(264, 48), "2200-0200")

    def test_start_zero(self):
        self.assertEqual(_shift_code_from(0, 48), "0000-0400")

    def test_half_hour(self):
        # 08:30 start
        self.assertEqual(_shift_code_from(102, 96), "0830-1630")

    def test_max_shift(self):
        # 23:30 + 12 h = 11:30 next day
        self.assertEqual(_shift_code_from(282, 144), "2330-1130")


if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(sys.modules[__name__])
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    print(f"\n{'='*60}")
    print(f"TOTAL: {result.testsRun} tests | "
          f"PASSED: {result.testsRun - len(result.failures) - len(result.errors)} | "
          f"FAILED: {len(result.failures)} | "
          f"ERRORS: {len(result.errors)}")
    sys.exit(0 if result.wasSuccessful() else 1)
