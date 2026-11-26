"""Tests for the GroundStar engage-rule export and the INFORM flight export."""

import pandas as pd

from groundstar_rules import (
    decode_text, read_engage_rules, read_inform_flights, is_inform_flights,
    leg_week_starts, build_events, engage_tasks, rules_summary,
    default_actype_groups, referenced_groups, parse_flight_number_groups,
)

HDR = "<Version=7.9.73082/><Class Name=PBEngageRule/><Block Name=Engage/><Tab Name={}/>"

# Two tables, different column sets and orders; RAMP comes first here
RULES = "\r\n".join([
    HDR.format("RAMP"),
    "ruleDescription\truleActivation\tResourceType\trulePhase\tchainFlag\tmaxGroundTime\t"
    "minGroundTime\tairline\tbegin\tend\tcount\tmandQual1\tengagegroups\tpriority\t",
    "W turnaround GT<60\t1\tRAMP\t2\tYes\t60\t\t\t0\t0\t2\tW\tGround w\t10\t",
    "W turnaround GT<60 CAI\t1\tRAMP\t2\tYes\t60\t\t CAI\t0\t0\t3\tW\tGround w\t20\t",
    "W arrival GT>60\t1\tRAMP\t1\tYes\t\t61\t\t0\t30\t2\tW\tArr w\t10\t",
    "W departure GT>60\t1\tRAMP\t0\tYes\t\t61\t\t-60\t0\t2\tW\tDep w\t10\t",
    "Switched off\t0\tRAMP\t1\t\t\t\t\t0\t30\t9\tW\tOff\t10\t",
    "Reversed window\t1\tRAMP\t0\t\t\t\t\t30\t-25\t1\tW\tStair\t10\t",
    "",
    HDR.format("TRF"),
    "ruleActivation\tResourceType\tbegin\tcount\tend\tduration\tPax_T_Min\tPax_T_Max\t"
    "handlingType1\trulePhase\truleDescription\tmandQual1\tengagegroups\tpriority",
    "1\tTRF\t-150\t1\t-30\t\t1\t150\tI\t0\tCheck-in small\tTRF\tCheck-in\t5",
    "1\tTRF\t-150\t3\t-30\t\t151\t\tI\t0\tCheck-in big\tTRF\tCheck-in\t5",
    "1\tTRF\t-40\t2\t0\t20\t\t\t\t0\tBoarding\tTRF\tBoarding\t10",
    "",
])


def _legs_text():
    # airline;no;sfx;A/D;linked …;ht1;ht2;ht3;reg;type;stand;;station;time;…;pax(23);…seats(30)
    def line(date, al, no, ad, lal, lno, ldate, lad, ht1, hhmm, pax, seats=189, actype="738"):
        c = [""] * 46
        c[0], c[1], c[2], c[4] = date, al, no, ad
        c[5], c[6], c[8], c[9] = lal, lno, ldate, lad
        c[10], c[11], c[12], c[13], c[14], c[15] = ht1, "S", "P", "TC-ABC", actype, "3"
        c[17], c[18], c[23], c[30] = "IST", hhmm, str(pax), str(seats)
        return ";".join(c)
    return "\r\n".join([
        line("20250811", "PGT", "100", "A", "PGT", "101", "20250811", "D", "D", "0800", 120),
        line("20250811", "PGT", "101", "D", "PGT", "100", "20250811", "A", "D", "0840", 180),
        line("20250811", "CAI", "200", "A", "CAI", "201", "20250811", "D", "I", "1000", 160),
        line("20250811", "CAI", "201", "D", "CAI", "200", "20250811", "A", "I", "1045", 170),
        line("20250812", "SXS", "300", "A", "SXS", "301", "20250812", "D", "I", "0600", 90),
        line("20250812", "SXS", "301", "D", "SXS", "300", "20250812", "A", "I", "0900", 60),
        line("20250818", "SXS", "400", "A", "SXS", "401", "20250818", "D", "D", "0600", 90),
        line("20250818", "SXS", "401", "D", "SXS", "400", "20250818", "A", "D", "0700", 90),
    ]) + "\r\n"


def test_utf16_and_any_table_or_column_order():
    rules, warns = read_engage_rules(RULES.encode("utf-16"))
    assert not warns
    assert set(rules["ResourceType"]) == {"RAMP", "TRF"}
    assert len(rules) == 9
    # same content, tables swapped, UTF-8 with BOM and semicolons
    blocks = RULES.split("\r\n\r\n")
    swapped = "\r\n\r\n".join([blocks[1], blocks[0]]).replace("\t", ";")
    r2, _ = read_engage_rules(b"\xef\xbb\xbf" + swapped.encode("utf-8"))
    assert len(r2) == 9 and set(r2["ResourceType"]) == {"RAMP", "TRF"}
    s = rules_summary(rules).set_index("Department")
    assert s.loc["RAMP", "Rules"] == 6 and s.loc["RAMP", "Active"] == 5


def test_cp1254_text_decodes():
    assert decode_text("Yatı".encode("cp1254")) == "Yatı"


def test_inform_parse_pairs_and_weeks():
    data = _legs_text().encode()
    assert is_inform_flights(data.decode())
    legs, warns = read_inform_flights(data)
    assert not warns and len(legs) == 8
    weeks = leg_week_starts(legs)
    assert [w.date().isoformat() for w in weeks] == ["2025-08-11", "2025-08-18"]
    ev = build_events(legs, weeks[0])
    assert len(ev) == 3 and all(e.chained for e in ev)    # week 2 left out
    gts = sorted(e.ground_time for e in ev)
    assert gts == [40, 45, 180]


def _tasks(deps, **kw):
    legs, _ = read_inform_flights(_legs_text().encode())
    ev = build_events(legs, leg_week_starts(legs)[0])
    rules, _ = read_engage_rules(RULES.encode("utf-16"))
    return engage_tasks(ev, rules, deps, **kw)


def test_phases_ground_time_and_priority_override():
    res = _tasks(["RAMP"])
    by = {}
    for t in res.tasks:
        by.setdefault(t.flight, []).append(t)
    pgt = by["PGT100/PGT101"]                         # GT 40: ground-time rule
    assert [(t.rule, t.staff, t.earliest, t.latest_end) for t in pgt] == \
        [("W turnaround GT<60", 2, 480, 520)]
    cai = by["CAI200/CAI201"]                         # CAI rule wins (priority 20)
    assert [(t.rule, t.staff) for t in cai] == [("W turnaround GT<60 CAI", 3)]
    sxs = by["SXS300/SXS301"]                         # GT 180: arrival + departure rules
    got = sorted((t.rule, t.earliest, t.latest_end) for t in sxs)
    assert got == [("W arrival GT>60", 1440 + 360, 1440 + 390),
                   ("W departure GT>60", 1440 + 480, 1440 + 540)]
    assert all(t.occupation == "W" and t.department == "RAMP" for t in res.tasks)
    assert any("Reversed window" in w for w in res.warnings)
    assert not any(t.rule == "Switched off" for t in res.tasks)


def test_pax_bands_handling_type_and_duration():
    res = _tasks(["TRF"])
    checkin = {t.flight: t.staff for t in res.tasks if t.name == "Check-in"}
    # only international departures: CAI201 (170 pax → big), SXS301 (60 → small)
    assert checkin == {"CAI200/CAI201": 3, "SXS300/SXS301": 1}
    board = [t for t in res.tasks if t.name == "Boarding"]
    assert len(board) == 3
    assert all(t.duration == 20 and t.latest_end - t.earliest == 40 for t in board)


def test_plan_per_department():
    res = _tasks(["RAMP", "TRF"], plan_by="department")
    assert {t.occupation for t in res.tasks} == {"RAMP", "TRF"}


def test_group_helpers():
    legs, _ = read_inform_flights(_legs_text().encode())
    df = default_actype_groups(legs, ["NB", "WB", "BULK", "01_SEATS_0-20"])
    row = df.set_index("AC type").loc["738"]
    assert row["NB"] and not row["WB"] and not row["01_SEATS_0-20"]
    fg = parse_flight_number_groups("THYA = THY2600-2602, THY2805\nX = ")
    assert ("THY", 2601) in fg["THYA"] and ("THY", 2805) in fg["THYA"] and fg["X"] == set()
