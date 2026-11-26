"""Tests for the Plan Control .sfp export (synthetic template)."""

import pytest

from sfp_export import (
    SfpTemplate, ShiftRecord, build_sfp, read_template, read_sfp_shifts, _cstr,
)

BASE = b"\x18\x80\x19\x80\x1a\x80"


def _template_bytes():
    head = b"\xff\xff\x23\x00\x0a\x00PCShiftDoc" + _cstr("DIY_RAMP") + b"\x00" * 40
    tpl = SfpTemplate(head=head, list_decl=b"\xff\xff\x03\x00\x0a\x00CShiftList",
                      shift_decl=b"\xff\xff\x2c\x00\x06\x00CShift", shift_tag=b"\x20\x80",
                      task_tag=b"\x16\x80", base_tags=BASE, resource_type="DIY_RAMP",
                      n_shifts=0, shift_id_base=100, task_id_base=500, group_id_base=900,
                      qualifications=[])
    recs = [ShiftRecord("DIY_W", "DIY_RAMP", (0, 1, 1, 1, 1, 1, 0), "0700-1500", 420, 900,
                        100, 1, 500, 900, 30),
            ShiftRecord("DIY_DA", "DIY_RAMP", (1, 1, 1, 1, 1, 1, 1), "2200-0600", 1320, 360,
                        101, 7, 501, 901, 60)]
    return tpl, recs, build_sfp(tpl, recs)


def test_written_file_reads_back_and_serves_as_template():
    tpl, recs, data = _template_bytes()
    back = read_sfp_shifts(data)
    assert [(r["qual"], r["code"], r["start"], r["end"], r["flags"], r["day"]) for r in back] == \
        [(x.qualification, x.code, x.start, x.end, x.valid_days, x.day) for x in recs]
    assert back[0]["window"] == (420, 900)
    assert back[1]["window"] == (6 * 1440 + 1320, 6 * 1440 + 1320 + 480)   # overnight Sunday
    t2 = read_template(data)
    assert t2.checked_shifts == 2 and t2.checked_day_tasks == 2
    assert t2.resource_type == "DIY_RAMP" and t2.shift_id_base == 102
    assert t2.head == tpl.head and t2.shift_tag == b"\x20\x80" and t2.task_tag == b"\x16\x80"


def test_export_of_a_plan_matches_its_workers():
    from task_solver import TaskParams, solve_tasks
    from tasks import Task
    from sfp_export import shift_records
    tpl = read_template(_template_bytes()[2])
    tasks = [Task(i, "Load", "DIY_W", 1440 * d + 600, 1440 * d + 660, 60, 2,
                  f"F{i}", "1", department="DIY_RAMP") for i, d in enumerate([0, 0, 2, 5])]
    res = solve_tasks(tasks, TaskParams(solver_time_limit_sec=15))
    recs = shift_records(res, tpl)
    back = read_sfp_shifts(build_sfp(tpl, recs))
    assert len(back) == len(res.workers)
    assert sorted((r["code"], r["day"]) for r in back) == \
        sorted((w.code, w.shift.day + 1) for w in res.workers)
    assert {r["qual"] for r in back} == {"DIY_W"} and {r["rt"] for r in back} == {"DIY_RAMP"}
    code_days = {}
    for r in back:
        code_days.setdefault(r["code"], set()).add(r["day"] % 7)   # Sunday-first index
    for r in back:
        assert {i for i, f in enumerate(r["flags"]) if f} == code_days[r["code"]]


def test_not_an_sfp():
    with pytest.raises(ValueError):
        read_template(b"hello")
