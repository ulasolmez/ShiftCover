"""Tests for groundstar_import: GroundStar 'Curve Data' exports + plain curves.

Synthetic workbooks mimic the real export layout:
  metadata rows · series-info rows (type/role/visible/colour) · header ·
  Date | Time | Minutes Since Start of Day | Workload | Sum of Visible | curves…

Set GS_SAMPLE=/path/to/export.xlsx to also check a real export.
"""
import datetime as dt
import io
import os

import numpy as np
import openpyxl

from groundstar_import import read_workload_file, curve_week, combine_by_occupation

_pass = _fail = 0


def check(label, cond, detail=""):
    global _pass, _fail
    if cond:
        _pass += 1; print(f"  [PASS] {label}")
    else:
        _fail += 1; print(f"  [FAIL] {label}  {detail}")


class Named(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data); self.name = name


def gs_book(curves, start, days, step=5, sod=dt.time(0, 0), sheet="Curve Data",
            translations=None, value_fn=None):
    """Build a GroundStar-like workload export. ``curves`` = list of names."""
    wb = openpyxl.Workbook()
    ws = wb.active; ws.title = sheet
    meta = [("GS Planning", "2020 (7.9.73082 X64)"), ("Format", "Workload"),
            ("Version", 2), ("Export Type", "Workload"),
            ("Time Period", start, start + dt.timedelta(days=days - 1)),
            ("Start of Day", sod), ("Resource Type", curves[0] if len(curves) == 1 else "MIX"),
            ("Locations", "all locations"), (None,), ("Curve Building Criteria", "Task Resource Type")]
    for m in meta:
        ws.append(list(m))
    ws.append([None, None, None] + ["Workload"] * (2 + len(curves)))
    ws.append([None, None, None, "TotalSum", "SumOfVisible"] + ["Standard"] * len(curves))
    ws.append([None, None, None, 1, 0] + [1] * len(curves))
    ws.append([None, None, None, "0,255,0", "255,195,107"] + ["0,255,0"] * len(curves))
    ws.append(["Date", "Time", "Minutes Since Start of Day", "Workload", "Sum of Visible"] + curves)
    value_fn = value_fn or (lambda k, ts: (k + 1) * (1 + ts.hour // 6))
    sod_min = sod.hour * 60 + sod.minute
    for d in range(days):
        day = start + dt.timedelta(days=d)
        for m in range(0, 1440, step):
            ts = day + dt.timedelta(minutes=sod_min + m)
            vals = [value_fn(k, ts) for k in range(len(curves))]
            ws.append([day, ts.time(), m, sum(vals), sum(vals)] + vals)
    if translations:
        tr = wb.create_sheet("Translations")
        tr.append(["CurveDataSheet.Name", translations])
    buf = io.BytesIO(); wb.save(buf)
    return buf.getvalue()


MON = dt.datetime(2025, 8, 11)          # a Monday

print("\n=== single curve, 8-day export (like DIY_RAMP) ===")
wf = read_workload_file(Named(gs_book(["DIY_RAMP"], MON, 8), "DIY_RAMP.xlsx"))
check("detected as GroundStar", wf.kind == "groundstar")
check("resource type read", wf.resource_type == "DIY_RAMP")
check("default curve = the Standard column", [c.name for c in wf.default_curves()] == ["DIY_RAMP"])
check("one full Mon–Sun week offered", wf.week_starts() == [MON], wf.week_starts())
d = curve_week(wf.default_curves()[0], MON)
check("2016 slots", len(d) == 2016)
check("Mon 00:00 value", d[0] == 1)
check("Mon 06:00 value", d[72] == 2)
check("Sun 23:55 value", d[-1] == 4)

print("\n=== multi-curve file (classification by task) ===")
wf = read_workload_file(Named(gs_book(["DCL", "CLN", "INF"], MON, 7), "ramp.xlsx"))
names = [c.name for c in wf.default_curves()]
check("three Standard curves offered", names == ["DCL", "CLN", "INF"], names)
check("aggregates not offered by default",
      all(c.role == "Standard" for c in wf.default_curves()))
cln = next(c for c in wf.curves if c.name == "CLN")
check("CLN values (k=1 → ×2)", curve_week(cln, MON)[0] == 2)
tot = next(c for c in wf.curves if c.role == "TotalSum")
check("TotalSum = sum of curves", curve_week(tot, MON)[0] == 1 + 2 + 3)

print("\n=== 15-min export ===")
wf = read_workload_file(Named(gs_book(["X"], MON, 7, step=15,
                                      value_fn=lambda k, ts: ts.minute // 15 + 1), "q.xlsx"))
d = curve_week(wf.curves[-1], MON)
check("warning about conversion", any("15-min" in w for w in wf.warnings), wf.warnings)
check("each 15-min value held for 3 five-min slots",
      list(d[:12]) == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], list(d[:12]))

print("\n=== start of day 03:00 ===")
wf = read_workload_file(Named(gs_book(["X"], MON, 8, sod=dt.time(3, 0),
                                      value_fn=lambda k, ts: ts.hour), "sod.xlsx"))
starts = wf.week_starts()
d = curve_week(wf.curves[-1], starts[0])
check("week starts on a Monday 00:00", starts and starts[0].weekday() == 0 and starts[0].hour == 0, starts)
check("slot at Tue 01:00 holds hour 1 (rows after midnight → next day)",
      d[288 + 12] == 1, d[288 + 12])
check("slot at Tue 05:00 holds hour 5", d[288 + 60] == 5)
check("Mon 00:00–02:55 filled from the export's last night (hour 0-2)",
      list(d[[0, 12, 24, 35]]) == [0, 1, 2, 2], list(d[[0, 12, 24, 35]]))
check("Mon 03:00 = first exported row", d[36] == 3)

print("\n=== export starting on Wednesday ===")
wed = MON + dt.timedelta(days=2)
wf = read_workload_file(Named(gs_book(["X"], wed, 7,
                                      value_fn=lambda k, ts: ts.weekday() + 1), "wed.xlsx"))
check("no full Mon–Sun week", wf.week_starts() == [])
d = curve_week(wf.curves[-1], None)
check("weekday mapping: Mon slot = Monday's data",
      [int(d[i * 288]) for i in range(7)] == [1, 2, 3, 4, 5, 6, 7],
      [int(d[i * 288]) for i in range(7)])

print("\n=== two-week export ===")
wf = read_workload_file(Named(gs_book(["X"], MON, 15,
                                      value_fn=lambda k, ts: 1 if ts < MON + dt.timedelta(days=7) else 9),
                              "two.xlsx"))
check("two Mondays offered", len(wf.week_starts()) == 2, wf.week_starts())
check("second week picked correctly", curve_week(wf.curves[-1], wf.week_starts()[1]).min() == 9)

print("\n=== fractional workload ===")
wf = read_workload_file(Named(gs_book(["X"], MON, 7, value_fn=lambda k, ts: 2.1), "frac.xlsx"))
check("2.1 rounded up to 3", curve_week(wf.curves[-1], MON)[0] == 3)
wf = read_workload_file(Named(gs_book(["X"], MON, 7, value_fn=lambda k, ts: 3.0000000001), "fp.xlsx"))
check("float noise 3.0000000001 stays 3", curve_week(wf.curves[-1], MON)[0] == 3)

print("\n=== localised sheet name via Translations ===")
wf = read_workload_file(Named(gs_book(["X"], MON, 7, sheet="Eğri Verileri",
                                      translations="Eğri Verileri"), "tr.xlsx"))
check("found renamed curve sheet", wf.kind == "groundstar" and len(curve_week(wf.curves[-1], MON)) == 2016)

print("\n=== plain CSV (old format) ===")
csv = "Interval,Day,Time,Workers\n" + "\n".join(f"{i},Monday,00:00,{i % 3}" for i in range(2016))
wf = read_workload_file(Named(csv.encode(), "plain.csv"))
check("plain detected", wf.kind == "plain")
check("'Workers' column used, not Interval", curve_week(wf.curves[0], None).max() == 2)
check("curve named after file", wf.curves[0].name == "plain")
rows = ["Time,Required"] + [f"t{i},{'' if i == 5 else (-1 if i == 6 else 2)}" for i in range(2016)]
wf = read_workload_file(Named("\n".join(rows).encode(), "b.csv"))
d = curve_week(wf.curves[0], None)
check("blank → 0, negative → 0", d[5] == 0 and d[6] == 0 and d[7] == 2)
check("warnings for blank and negative", len(wf.warnings) == 2, wf.warnings)
try:
    read_workload_file(Named(b"Required\n1\n2\n", "short.csv")); check("short file rejected", False)
except ValueError:
    check("short file rejected", True)

print("\n=== combine by occupation ===")
a, b, c = np.ones(2016, int), np.full(2016, 2), np.full(2016, 5)
out = combine_by_occupation([{"occupation": "RAMP", "demand": a},
                             {"occupation": "CLN", "demand": c},
                             {"occupation": " RAMP ", "demand": b},
                             {"occupation": "", "demand": c}])
check("same name summed, order kept, blank skipped",
      list(out) == ["RAMP", "CLN"] and out["RAMP"][0] == 3 and out["CLN"][0] == 5, list(out))

real = os.environ.get("GS_SAMPLE")
if real:
    print(f"\n=== real export: {os.path.basename(real)} ===")
    wf = read_workload_file(real)
    st = wf.week_starts()
    check("real: GroundStar + one full week", wf.kind == "groundstar" and len(st) >= 1)
    d = curve_week(wf.default_curves()[0], st[0])
    check("real: 2016 slots, non-zero", len(d) == 2016 and d.sum() > 0)
    print(f"    {wf.resource_type}: peak {d.max()}, {d.sum() / 12:,.1f} workload-h")

print(f"\nTOTAL: {_pass + _fail} tests | PASSED: {_pass} | FAILED: {_fail}")
