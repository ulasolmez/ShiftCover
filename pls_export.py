"""
GroundStar shift-plan import file (.pls, PLSHIFTS)
==================================================

The text format GroundStar's *shift plan import* reads (Plan Control keeps a
copy as ``Temp/*.sft``).  UTF-16 with BOM, CRLF line ends::

    HEADER;PLSHIFTS;V1.11;<export date>;<export HHMM>;<resource type>;;<from>;<to>;
    <id>;<code>;;<resource type[:5]>;18;24;<date>;<begin>;<end>;;;;;;;<n>;<resource type>;<qualification>[;…];00;00;

* one line per shift instance (one worker, one day)
* ``id`` = a running number followed by the two-digit day of month
* ``code`` = ``HHMM-HHMM``; a shift that ends at midnight ends ``2359``;
  overnight shifts end before they begin (``1700-0130``); ``date`` is the
  day the shift begins
* ``n`` = number of the following items: the resource type and the
  qualification(s), e.g. ``02;DIY_RAMP;DIY_W`` or ``03;DIY_RAMP;DIY_CW;DIY_W``

Checked against six PLSHIFTS files from GroundStar (RAMP, OPS, TRF,
DRIVER): parsing and writing them again gives the same bytes.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

VERSION = "V1.11"
BG, FG = "18", "24"


@dataclass
class PlsShift:
    sid: str                      # full id as written (running number + DD)
    code: str
    date: dt.date
    begin: str                    # HHMM
    end: str                      # HHMM
    resource_type: str
    qualifications: List[str] = field(default_factory=list)
    short: str = ""               # resource short name (default rt[:5])
    bg: str = BG
    fg: str = FG


@dataclass
class PlsFile:
    export_date: str              # YYYYMMDD
    export_time: str              # HHMM
    resource_type: str
    period_from: dt.date
    period_to: dt.date
    shifts: List[PlsShift]


def _d(s: str) -> dt.date:
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def _decode(data: bytes) -> str:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1254")


def read_pls(data: bytes) -> PlsFile:
    lines = [l for l in _decode(data).split("\r\n")]
    if lines and lines[-1] == "":
        lines = lines[:-1]
    h = lines[0].split(";")
    if h[0] != "HEADER" or h[1] != "PLSHIFTS":
        raise ValueError("Not a PLSHIFTS file (first line must start HEADER;PLSHIFTS).")
    shifts = []
    for l in lines[1:]:
        if not l.strip():
            continue
        f = l.split(";")
        n = int(f[15])
        items = f[16:16 + n]
        if f[16 + n:16 + n + 2] != ["00", "00"]:
            raise ValueError(f"Unexpected line end: {l}")
        shifts.append(PlsShift(f[0], f[1], _d(f[6]), f[7], f[8], items[0], items[1:],
                               f[3], f[4], f[5]))
    return PlsFile(h[3], h[4], h[5], _d(h[7]), _d(h[8]), shifts)


def write_pls(p: PlsFile) -> bytes:
    out = [f"HEADER;PLSHIFTS;{VERSION};{p.export_date};{p.export_time};"
           f"{p.resource_type};;{p.period_from:%Y%m%d};{p.period_to:%Y%m%d};"]
    for s in p.shifts:
        items = [s.resource_type] + list(s.qualifications)
        out.append(";".join([s.sid, s.code, "", s.short or s.resource_type[:5], s.bg, s.fg,
                             f"{s.date:%Y%m%d}", s.begin, s.end, "", "", "", "", "", "",
                             f"{len(items):02d}", *items, "00", "00", ""]))
    return ("﻿" + "\r\n".join(out) + "\r\n").encode("utf-16-le")


# ── Shift codes known to GroundStar ─────────────────────────────────────────
_CODE_U16 = re.compile(rb"(?:\d\x00){4}-\x00(?:\d\x00){4}")
_CODE_TXT = re.compile(r"\b(\d{4})-(\d{4})\b")


def normalise_code(code: str) -> str:
    """GroundStar writes a shift ending at midnight as ...-2359; ours end 0000."""
    a, b = code.split("-")
    return f"{a}-0000" if b == "2359" else code


def read_shift_codes(data: bytes) -> List[str]:
    """Shift codes from Shifttype.Shifttype.GRule, a .pls/.sft or a text list."""
    codes = {m.group(0).decode("utf-16-le") for m in _CODE_U16.finditer(data)}
    if not codes:
        txt = _decode(data)
        codes = {f"{a}-{b}" for a, b in _CODE_TXT.findall(txt)}
    ok = set()
    for c in codes:
        a, b = c.split("-")
        if int(a[:2]) < 24 and int(a[2:]) < 60 and int(b[:2]) < 24 and int(b[2:]) < 60:
            ok.add(c)
    return sorted(ok)


# ── Plan → PLSHIFTS ──────────────────────────────────────────────────────────
def plan_to_pls(res, week_start: dt.date, resource_type: str,
                occupations: Optional[Set[str]] = None, first_number: int = 900000,
                now: Optional[dt.datetime] = None) -> PlsFile:
    """One PLSHIFTS line per worker shift of a task_solver.TaskResult
    (or one per counted shift of a curve-based result)."""
    multi = getattr(res, "multi", res)
    now = now or dt.datetime.now()
    rows: List[Tuple] = []
    for occ in multi.occupations:
        if occupations is not None and occ.name not in occupations:
            continue
        for s, c in zip(occ.phase1.shifts, occ.phase1.counts):
            for _ in range(int(c)):
                rows.append((s.day, s.start_interval, s.duration_intervals, occ.name))
    rows.sort()
    shifts = []
    for k, (day, st, dur, occ) in enumerate(rows):
        date = week_start + dt.timedelta(days=day)
        b_min = st * 5
        e_min = (st + dur) * 5
        begin = f"{b_min // 60:02d}{b_min % 60:02d}"
        e = e_min % 1440
        end = "2359" if e == 0 else f"{e // 60:02d}{e % 60:02d}"
        quals = [occ] if occ else [resource_type]
        shifts.append(PlsShift(f"{first_number + k}{date.day:02d}", f"{begin}-{end}", date,
                               begin, end, resource_type, quals))
    return PlsFile(f"{now:%Y%m%d}", f"{now:%H%M}", resource_type, week_start,
                   week_start + dt.timedelta(days=6), shifts)
