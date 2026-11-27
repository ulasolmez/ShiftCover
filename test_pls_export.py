"""Tests for the PLSHIFTS (.pls) shift-plan import file."""

import datetime as dt

from pls_export import (
    read_pls, write_pls, read_shift_codes, normalise_code, plan_to_pls, PlsFile,
)

SAMPLE = (
    "HEADER;PLSHIFTS;V1.11;20251015;2347;DIY_RAMP;;20260824;20260830;\r\n"
    "17626727;0730-1430;;DIY_R;18;24;20260827;0730;1430;;;;;;;02;DIY_RAMP;DIY_W;00;00;\r\n"
    "17644930;1700-0130;;DIY_R;18;24;20260830;1700;0130;;;;;;;02;DIY_RAMP;DIY_CW;00;00;\r\n"
    "17623824;0600-1130;;DIY_R;18;24;20260824;0600;1130;;;;;;;03;DIY_RAMP;DIY_CW;DIY_W;00;00;\r\n"
    "17642326;1800-2359;;DIY_R;18;24;20260826;1800;2359;;;;;;;02;DIY_RAMP;DIY_EW;00;00;\r\n"
)


def test_round_trip_is_byte_identical():
    raw = ("﻿" + SAMPLE).encode("utf-16-le")
    f = read_pls(raw)
    assert f.resource_type == "DIY_RAMP" and f.period_from == dt.date(2026, 8, 24)
    assert [s.qualifications for s in f.shifts] == [["DIY_W"], ["DIY_CW"], ["DIY_CW", "DIY_W"], ["DIY_EW"]]
    assert write_pls(f) == raw


def test_shift_codes_from_utf16_and_text():
    raw = ("﻿" + SAMPLE).encode("utf-16-le")
    assert read_shift_codes(raw) == ["0600-1130", "0730-1430", "1700-0130", "1800-2359"]
    assert read_shift_codes(b"0130-1030\n0730-1430, 2500-0100") == ["0130-1030", "0730-1430"]
    assert normalise_code("1800-2359") == "1800-0000" and normalise_code("0730-1430") == "0730-1430"


def test_plan_to_pls_lines():
    from task_solver import TaskParams, solve_tasks
    from tasks import Task
    tasks = [Task(0, "Load", "DIY_W", 600, 660, 60, 2, department="DIY_RAMP"),
             Task(1, "Load", "DIY_W", 6 * 1440 + 1380, 6 * 1440 + 1435, 55, 1,
                  department="DIY_RAMP")]
    res = solve_tasks(tasks, TaskParams(solver_time_limit_sec=15, min_shift_hours=3))
    pf = plan_to_pls(res, dt.date(2026, 8, 24), "DIY_RAMP",
                     now=dt.datetime(2026, 11, 27, 9, 30))
    assert len(pf.shifts) == len(res.workers)
    back = read_pls(write_pls(pf))
    assert back.period_to == dt.date(2026, 8, 30) and back.export_time == "0930"
    for s in back.shifts:
        assert s.code == f"{s.begin}-{s.end}"
        assert s.sid.endswith(f"{s.date.day:02d}")
        assert s.resource_type == "DIY_RAMP" and s.qualifications == ["DIY_W"]
    assert len({s.sid for s in back.shifts}) == len(back.shifts)


def test_midnight_and_overnight_ends():
    from types import SimpleNamespace as NS
    from solver import CandidateShift
    shifts = [CandidateShift(0, 6, 18 * 12, 6 * 12),      # Sun 18:00-24:00
              CandidateShift(1, 2, 17 * 12, 8 * 12 + 6)]  # Wed 17:00-01:30
    occ = NS(name="DIY_CW", phase1=NS(shifts=shifts, counts=[2, 1]))
    pf = plan_to_pls(NS(multi=NS(occupations=[occ])), dt.date(2026, 8, 24), "DIY_RAMP",
                     first_number=100)
    got = [(s.sid, s.code, f"{s.date:%Y%m%d}") for s in pf.shifts]
    assert got == [("10026", "1700-0130", "20260826"),
                   ("10130", "1800-2359", "20260830"),
                   ("10230", "1800-2359", "20260830")]
