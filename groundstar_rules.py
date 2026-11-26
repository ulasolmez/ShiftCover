"""
GroundStar engage rules + INFORM flight export → tasks
=======================================================

Engage-rule export (``Class Name=PBEngageRule``)
    One file holds several tables, one per resource type / department
    (``Tab Name=DIY_RAMP`` …).  Every table has its own header line and its
    own column set and order, and the tables can come in any order.  The
    file may be UTF-16 (GroundStar default), UTF-8 or Windows-1254, tab-,
    semicolon- or comma-separated.  Each table is read by its own header,
    so column and table order do not matter.

INFORM flight export (``.txt``)
    Semicolon-separated, no header, one row per leg; arrival and departure
    of a rotation point at each other.

How a rule becomes a task (GroundStar semantics)
    rulePhase 1 = arrival leg (times relative to STA), 0 = departure leg
    (relative to STD), 2 = ground time (begin relative to STA, end
    relative to STD).  The task window is [ref + begin − setup,
    ref + end + cleanup].  With a ``duration`` the task floats inside the
    window, otherwise it fills it.  ``count`` is the staff.
    Within one flight and one engage group of a resource type only the
    matching rule(s) with the highest priority are used (ties all apply);
    this is how airline-specific rules override the defaults.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from tasks import Task, Flight, WEEK_MIN, _fit_week, _WB_HINTS

# ══════════════════════════════════════════════════════════════════════════════
# Text decoding
# ══════════════════════════════════════════════════════════════════════════════
def decode_text(data: bytes) -> str:
    """UTF-16 (with or without BOM), UTF-8 (BOM optional) or Windows-1254."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    if data[:3] == b"\xef\xbb\xbf":
        return data[3:].decode("utf-8", errors="replace")
    # UTF-16 without BOM: every other byte is 0 for ASCII text
    if len(data) > 4 and data[1:200:2].count(0) > 0.4 * len(data[1:200:2]):
        try:
            return data.decode("utf-16-le")
        except UnicodeDecodeError:
            pass
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1254", errors="replace")


# ══════════════════════════════════════════════════════════════════════════════
# INFORM flight export
# ══════════════════════════════════════════════════════════════════════════════
# Column positions (0-based) of the semicolon INFORM export
INFORM_COLUMNS = {
    0: "Date (of the leg)", 1: "Airline", 2: "Flight number", 3: "Suffix",
    4: "A/D", 5: "Linked airline", 6: "Linked flight number",
    7: "Linked suffix", 8: "Linked date", 9: "Linked A/D",
    10: "Handling type 1 (D/I)", 11: "Handling type 2 (S/C)",
    12: "Handling type 3 (P/C/F…)", 13: "Registration", 14: "Aircraft type",
    15: "Stand", 17: "Station (origin/destination)", 18: "STA/STD (HHMM)",
    23: "Pax total", 30: "Seats", 33: "Cargo kg", 34: "Baggage kg",
    35: "Mail kg",
}


@dataclass
class Leg:
    airline: str
    number: str
    suffix: str
    date: dt.date
    ad: str                         # "A" / "D"
    link: Optional[Tuple]           # key of the linked leg
    ht1: str = ""                   # D / I
    ht2: str = ""                   # S / C
    ht3: str = ""                   # P / C / F / T
    reg: str = ""
    actype: str = ""
    stand: str = ""
    station: str = ""
    hhmm: int = 0                   # minutes after midnight (scheduled)
    pax: Optional[int] = None
    seats: Optional[int] = None
    cargo: Optional[int] = None     # cargo + mail, kg
    t: int = 0                      # minutes from the planning-week Monday

    @property
    def key(self) -> Tuple:
        return (self.airline, self.number, self.suffix, self.date, self.ad)

    @property
    def label(self) -> str:
        return f"{self.airline}{self.number}{self.suffix}"

    @property
    def stamp(self) -> dt.datetime:
        return dt.datetime.combine(self.date, dt.time()) + dt.timedelta(minutes=self.hhmm)


def _int(v) -> Optional[int]:
    try:
        s = str(v).strip()
        return int(float(s)) if s else None
    except ValueError:
        return None


def _date8(s: str) -> Optional[dt.date]:
    s = s.strip()
    if re.fullmatch(r"\d{8}", s):
        try:
            return dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))
        except ValueError:
            return None
    return None


def is_inform_flights(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()][:20]
    if not lines:
        return False
    ok = 0
    for l in lines:
        c = l.split(";")
        if len(c) >= 24 and _date8(c[0]) and c[4].strip() in ("A", "D"):
            ok += 1
    return ok >= max(1, int(0.8 * len(lines)))


def read_inform_flights(data: bytes) -> Tuple[List[Leg], List[str]]:
    text = decode_text(data)
    legs: List[Leg] = []
    bad = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        c = line.split(";")
        if len(c) < 24 or c[4].strip() not in ("A", "D") or not _date8(c[0]):
            bad += 1
            continue
        hhmm = c[18].strip()
        if not re.fullmatch(r"\d{3,4}", hhmm):
            bad += 1
            continue
        hhmm = int(hhmm)
        ld = _date8(c[8]) if len(c) > 8 else None
        link = ((c[5].strip(), c[6].strip(), c[7].strip(), ld, c[9].strip())
                if c[6].strip() and ld and c[9].strip() in ("A", "D") else None)
        cargo = None
        if len(c) > 35:
            cg, ml = _int(c[33]), _int(c[35])
            if cg is not None or ml is not None:
                cargo = (cg or 0) + (ml or 0)
        legs.append(Leg(
            airline=c[1].strip(), number=c[2].strip(), suffix=c[3].strip(),
            date=_date8(c[0]), ad=c[4].strip(), link=link,
            ht1=c[10].strip(), ht2=c[11].strip(), ht3=c[12].strip(),
            reg=c[13].strip(), actype=c[14].strip(), stand=c[15].strip(),
            station=c[17].strip() if len(c) > 17 else "",
            hhmm=(hhmm // 100) * 60 + hhmm % 100,
            pax=_int(c[23]), seats=_int(c[30]) if len(c) > 30 else None,
            cargo=cargo,
        ))
    warns = [f"{bad} line(s) not in INFORM flight format were skipped."] if bad else []
    return legs, warns


def leg_week_starts(legs: List[Leg]) -> List[pd.Timestamp]:
    return sorted({pd.Timestamp(l.date - dt.timedelta(days=l.date.weekday()))
                   for l in legs})


@dataclass
class Event:
    """One flight event: a rotation (arrival + departure) or a single leg."""
    arr: Optional[Leg]
    dep: Optional[Leg]

    @property
    def chained(self) -> bool:
        return self.arr is not None and self.dep is not None

    @property
    def ground_time(self) -> Optional[int]:
        return self.dep.t - self.arr.t if self.chained else None

    @property
    def label(self) -> str:
        if self.chained:
            return f"{self.arr.label}/{self.dep.label}"
        return (self.arr or self.dep).label

    @property
    def stand(self) -> str:
        return (self.arr or self.dep).stand


def build_events(legs: List[Leg], week_start: pd.Timestamp) -> List[Event]:
    """Pair arrivals with their linked departures; set week minutes.

    All legs take part in the pairing so rotations crossing the week edges
    still pair up; events with no leg inside the planning week are left out.
    """
    ws = week_start.to_pydatetime()
    by_key = {}
    for l in legs:
        l.t = int((l.stamp - ws).total_seconds() // 60)
        by_key[l.key] = l
    events: List[Event] = []
    used = set()
    for l in sorted(legs, key=lambda x: x.t):
        if l.key in used:
            continue
        other = by_key.get(l.link) if l.link else None
        if other is not None and other.key not in used and other.ad != l.ad:
            a, d = (l, other) if l.ad == "A" else (other, l)
            if d.t >= a.t:
                events.append(Event(a, d))
                used.update({a.key, d.key})
                continue
        events.append(Event(l if l.ad == "A" else None, l if l.ad == "D" else None))
        used.add(l.key)
    # keep events with at least one leg inside the planning week
    return [e for e in events
            if any(l is not None and 0 <= l.t < WEEK_MIN for l in (e.arr, e.dep))]


def events_dataframe(events: List[Event]) -> pd.DataFrame:
    def _f(l):
        if l is None:
            return ""
        return f"{l.stamp:%a %d %b %H:%M}"
    rows = []
    for e in sorted(events, key=lambda e: (e.arr or e.dep).t):
        l = e.arr or e.dep
        rows.append({
            "Flight": e.label, "Type": "Rotation" if e.chained else
            ("Arrival only" if e.arr else "Departure only"),
            "STA": _f(e.arr), "STD": _f(e.dep),
            "Ground time (min)": e.ground_time if e.chained else None,
            "AC type": l.actype, "Stand": l.stand,
            "Pax in": e.arr.pax if e.arr else None,
            "Pax out": e.dep.pax if e.dep else None,
            "Dom/Int": l.ht1,
        })
    return pd.DataFrame(rows)


def flights_to_events(flights: List[Flight]) -> List[Event]:
    """Plain flight table → events (no pax, handling types or aircraft type:
    rules filtering on those do not match)."""
    out = []
    for f in flights:
        parts = f.flight.split("/")
        def _leg(name, ad, t):
            m = re.match(r"([A-Z]+)?\s*(\d+)(\w*)", name.strip().upper())
            al, num, suf = (m.group(1) or "", m.group(2), m.group(3)) if m else ("", name, "")
            return Leg(al, num, suf, dt.date(2000, 1, 3), ad, None,
                       actype=f.category, stand=f.stand, t=t)
        a = _leg(parts[0], "A", f.sta) if f.sta is not None else None
        d = _leg(parts[-1], "D", f.std) if f.std is not None else None
        out.append(Event(a, d))
    return out


def events_to_flights(events: List[Event]) -> List[Flight]:
    """For the simple task-rule table (STA/STD + NB/WB)."""
    out = []
    for e in events:
        l = e.arr or e.dep
        ref = l.t
        cat = "WB" if any(h in l.actype.upper() for h in _WB_HINTS) else "NB"
        out.append(Flight(e.label, max(0, min(6, ref // 1440)),
                          e.arr.t if e.arr else None, e.dep.t if e.dep else None,
                          cat, l.stand))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Engage-rule export
# ══════════════════════════════════════════════════════════════════════════════
_HDR = re.compile(r"<Class Name=([^/>]*)/?>.*?<Tab Name=([^/>]*)/?>", re.I)


def _split(line: str, sep: str) -> List[str]:
    return next(csv.reader([line], delimiter=sep, quotechar='"'))


def _sep_of(line: str) -> str:
    counts = {s: line.count(s) for s in ("\t", ";", ",")}
    return max(counts, key=counts.get)


def read_engage_rules(data: bytes) -> Tuple[pd.DataFrame, List[str]]:
    """All rule tables of an engage-rule export as one DataFrame.

    Adds ``Tab`` (table name) and ``Row`` (line in the file).  All values are
    strings; empty cells are "".  Table and column order do not matter.
    """
    text = decode_text(data).replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    warns: List[str] = []
    frames = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip("﻿")
        m = _HDR.search(line)
        header_line = None
        tab = ""
        if m:
            tab = m.group(2).strip()
            if m.group(1).strip() and "engagerule" not in m.group(1).lower():
                warns.append(f"Table {tab}: class {m.group(1)} is not an engage rule — skipped.")
                i += 1
                while i < n and lines[i].strip() and not _HDR.search(lines[i]):
                    i += 1
                continue
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            header_line, i = (lines[j] if j < n else None), j + 1
        elif "ruleDescription" in line or "ruleActivation" in line:
            header_line, i = line, i + 1             # header without a tag line
        else:
            i += 1
            continue
        if header_line is None:
            break
        sep = _sep_of(header_line)
        cols = [c.strip() for c in _split(header_line, sep)]
        rows = []
        while i < n and lines[i].strip() and not _HDR.search(lines[i]):
            vals = [v.strip() for v in _split(lines[i], sep)]
            vals += [""] * (len(cols) - len(vals))
            row = {c: v for c, v in zip(cols, vals) if c}
            row["Tab"] = tab
            row["Row"] = i + 1
            rows.append(row)
            i += 1
        if rows:
            frames.append(pd.DataFrame(rows))
        elif tab:
            warns.append(f"Table {tab} has no rules.")
    if not frames:
        raise ValueError("No engage-rule tables found. Expected lines like "
                         "'<Version=…/><Class Name=PBEngageRule/>…<Tab Name=…/>' "
                         "followed by a header line with ruleDescription, "
                         "ResourceType, begin, end, count …")
    df = pd.concat(frames, ignore_index=True).fillna("")
    for c in ("ResourceType", "ruleDescription", "begin", "end", "count"):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing in the rule export.")
    df.loc[df["ResourceType"] == "", "ResourceType"] = df["Tab"]
    return df, warns


# Columns that only describe the task (not filters)
_NON_FILTER = {
    "Tab", "Row", "ruleActivation", "ruleDescription", "ResourceType",
    "begin", "end", "count", "duration", "setup", "cleanup", "splitSetup",
    "splitCleanup", "TaskDisplayColor", "mandQual1", "engagegroups", "priority",
    "flexibleTeam", "rulePhase",
}
_KNOWN_FILTERS = {
    "airline", "notAirline", "chainFlag", "minGroundTime", "maxGroundTime",
    "handlingType1", "handlingType2", "handlingType3", "handlingType1Group",
    "handlingType2Group", "handlingType3Group", "chainHandlingType1",
    "flightNumberGroup", "acTypeGroup", "notAcTypeGroup", "Pax_T_Min",
    "Pax_T_Max", "minTotalCargoWeight", "maxTotalCargoWeight", "leg",
}


def rules_summary(rules: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dep, g in rules.groupby("ResourceType", sort=False):
        act = g[g.get("ruleActivation", pd.Series("1", index=g.index)).astype(str) != "0"]
        quals = sorted({q for q in g.get("mandQual1", pd.Series(dtype=str)) if q})
        rows.append({"Department": dep, "Rules": len(g), "Active": len(act),
                     "Qualifications": ", ".join(quals) or "—",
                     "Table": ", ".join(sorted(set(g["Tab"])))})
    return pd.DataFrame(rows)


def referenced_groups(rules: pd.DataFrame, col_names: Iterable[str]) -> List[str]:
    out: Dict[str, None] = {}
    for c in col_names:
        if c in rules.columns:
            for v in rules[c]:
                for g in str(v).split(","):
                    if g.strip():
                        out.setdefault(g.strip(), None)
    return list(out)


# ── Aircraft-type groups ────────────────────────────────────────────────────
def _guess_group(name: str, actype: str, seats: Optional[int]) -> bool:
    n, t = name.upper(), actype.upper()
    wide = any(h in t for h in _WB_HINTS)
    small = (seats is not None and seats <= 20) or any(
        k in t for k in ("HAWKER", "H25", "C56", "C68", "GLF", "CL6", "FA7", "PC12", "LJ"))
    m = re.search(r"SEATS_(\d+)-(\d+)", n)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return seats is not None and lo <= seats <= hi if seats is not None else small
    if n in ("WB", "WIDE", "WIDEBODY", "HEAVY"):
        return wide and not small
    if n in ("NB", "NARROW", "NARROWBODY"):
        return not wide and not small
    if n == "ULD":
        return wide
    if n == "BULK":
        return not wide and not small
    if n in ("NB_ULD", "ULD_NB"):
        return False
    return False


def default_actype_groups(legs: List[Leg], groups: List[str]) -> pd.DataFrame:
    seats: Dict[str, Optional[int]] = {}
    cnt: Dict[str, int] = defaultdict(int)
    for l in legs:
        cnt[l.actype] += 1
        if l.seats:
            seats[l.actype] = max(seats.get(l.actype) or 0, l.seats)
    rows = []
    for t in sorted(cnt, key=lambda k: -cnt[k]):
        r = {"AC type": t, "Legs": cnt[t], "Seats": seats.get(t)}
        for g in groups:
            r[g] = _guess_group(g, t, seats.get(t))
        rows.append(r)
    return pd.DataFrame(rows)


# Known renamed groups: old flight-number group → today's airline code
DEFAULT_FN_GROUPS = {"THYA": "TKJ"}      # AJet: was THY 2xxx, now its own code TKJ


def parse_flight_number_groups(text: str) -> Dict[str, Set[Tuple[str, Optional[int]]]]:
    """One group per line: ``THYA = TKJ`` (every TKJ flight) or
    ``THYA = THY2600-2699, THY2805`` (flight numbers / ranges)."""
    out: Dict[str, Set[Tuple[str, Optional[int]]]] = {}
    for line in (text or "").splitlines():
        if "=" not in line:
            continue
        name, spec = line.split("=", 1)
        s: Set[Tuple[str, Optional[int]]] = set()
        for part in re.split(r"[,\s]+", spec.strip().upper()):
            if not part:
                continue
            if re.fullmatch(r"[A-Z0-9]{2,3}", part) and not part.isdigit():
                s.add((part, None))                 # whole airline
                continue
            m = re.fullmatch(r"([A-Z][A-Z0-9]{1,2}?)(\d+)(?:-(\d+))?", part)
            if not m:
                continue
            al, a, b = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
            s.update((al, k) for k in range(a, b + 1))
        out[name.strip()] = s
    return out


def default_fn_text(names: List[str]) -> str:
    return "\n".join(f"{n} = {DEFAULT_FN_GROUPS.get(n, '')}".rstrip() + ("" if n in DEFAULT_FN_GROUPS else " ")
                     for n in names)


# ══════════════════════════════════════════════════════════════════════════════
# Rules → tasks
# ══════════════════════════════════════════════════════════════════════════════
def _num(v) -> Optional[float]:
    try:
        s = str(v).strip().replace(",", ".")
        return float(s) if s else None
    except ValueError:
        return None


def _list(v) -> List[str]:
    return [x.strip() for x in str(v).split(",") if x.strip()]


def _code_group(name: str, value: str) -> bool:
    """Handling-type groups like P_F_T or NO_MILITARY."""
    n = name.upper()
    if n.startswith("NO_"):
        return value.upper() not in ("M", "MIL", "MILITARY")
    parts = n.split("_")
    if all(len(p) == 1 for p in parts):
        return value.upper() in parts
    return True


@dataclass
class _Ctx:
    leg: Leg
    other: Optional[Leg]
    chained: bool
    gt: Optional[int]
    pax: Optional[int]


@dataclass
class EngageResult:
    tasks: List[Task]
    warnings: List[str]
    summary: pd.DataFrame
    rule_use: pd.DataFrame


def engage_tasks(events: List[Event], rules: pd.DataFrame, departments: List[str],
                 plan_by: str = "qualification",
                 ac_groups: Optional[Dict[str, Set[str]]] = None,
                 fn_groups: Optional[Dict[str, Set[Tuple[str, int]]]] = None,
                 circular: bool = False) -> EngageResult:
    """Apply the engage rules of the chosen departments to every flight event."""
    ac_groups = ac_groups or {}
    fn_groups = fn_groups or {}
    warns: List[str] = []
    R = rules[rules["ResourceType"].isin(departments)].copy()
    if "ruleActivation" in R.columns:
        R = R[R["ruleActivation"].astype(str).str.strip() != "0"]
    cols = set(R.columns)
    unknown = sorted(c for c in cols - _NON_FILTER - _KNOWN_FILTERS
                     if (R[c].astype(str).str.strip() != "").any())
    if unknown:
        warns.append("Filter column(s) not understood and ignored: " + ", ".join(unknown))
    if "leg" in cols and (R["leg"].astype(str).str.strip() != "").any():
        warns.append("The 'leg' filter is ignored (rules using it apply to every leg).")
    missing_fn = sorted({g for g in referenced_groups(R, ["flightNumberGroup"])
                         if g not in fn_groups})
    if missing_fn:
        warns.append("Flight-number group(s) without a definition — their rules match "
                     "no flight: " + ", ".join(missing_fn))

    recs = R.to_dict("records")
    for r in recs:
        r["_phase"] = int(_num(r.get("rulePhase")) or 0)
        r["_prio"] = _num(r.get("priority")) or 0
        r["_group"] = (r.get("ResourceType", ""), r.get("engagegroups") or
                       f"rule:{r.get('Row')}")

    def g(r, k):
        return str(r.get(k, "") or "").strip()

    def matches(r, c: _Ctx) -> bool:
        l = c.leg
        fg = g(r, "flightNumberGroup")
        al = _list(g(r, "airline"))
        # With a flight-number group the group decides which flights are meant
        # (e.g. THYA = AJet, today airline TKJ), so the airline filter is skipped.
        if al and not fg and l.airline not in al:
            return False
        if l.airline in _list(g(r, "notAirline")):
            return False
        cf = g(r, "chainFlag").lower()
        if cf in ("yes", "1", "true", "y") and not c.chained:
            return False
        if cf in ("no", "0", "false", "n") and c.chained:
            return False
        lo, hi = _num(g(r, "minGroundTime")), _num(g(r, "maxGroundTime"))
        if lo is not None or hi is not None:
            if c.gt is None or (lo is not None and c.gt < lo) or (hi is not None and c.gt > hi):
                return False
        for k, v in (("handlingType1", l.ht1), ("handlingType2", l.ht2),
                     ("handlingType3", l.ht3)):
            want = _list(g(r, k))
            if want and v not in want:
                return False
        for k, v in (("handlingType1Group", l.ht1), ("handlingType2Group", l.ht2),
                     ("handlingType3Group", l.ht3)):
            grp = g(r, k)
            if grp and not _code_group(grp, v):
                return False
        ch = _list(g(r, "chainHandlingType1"))
        if ch and (c.other is None or c.other.ht1 not in ch):
            return False
        if fg:
            num = _int(re.sub(r"\D", "", l.number))
            grp = fn_groups.get(fg, set())
            if (l.airline, num) not in grp and (l.airline, None) not in grp:
                return False
        mine = ac_groups.get(l.actype, set())
        ag = _list(g(r, "acTypeGroup"))
        if ag and not any(x in mine for x in ag):
            return False
        if any(x in mine for x in _list(g(r, "notAcTypeGroup"))):
            return False
        lo, hi = _num(g(r, "Pax_T_Min")), _num(g(r, "Pax_T_Max"))
        if lo is not None or hi is not None:
            if c.pax is None or (lo is not None and c.pax < lo) or (hi is not None and c.pax > hi):
                return False
        lo, hi = _num(g(r, "minTotalCargoWeight")), _num(g(r, "maxTotalCargoWeight"))
        if lo is not None or hi is not None:
            cg = l.cargo
            if cg is None or (lo is not None and cg < lo) or (hi is not None and cg > hi):
                return False
        return True

    tasks: List[Task] = []
    use = defaultdict(int)
    bad_window: Dict[str, int] = defaultdict(int)
    dropped = 0
    for e in events:
        ctx = {}
        if e.arr is not None:
            ctx[1] = _Ctx(e.arr, e.dep, e.chained, e.ground_time, e.arr.pax)
        if e.dep is not None:
            ctx[0] = _Ctx(e.dep, e.arr, e.chained, e.ground_time, e.dep.pax)
        if e.chained:
            pax = max(x for x in (e.arr.pax, e.dep.pax, -1) if x is not None)
            ctx[2] = _Ctx(e.arr, e.dep, True, e.ground_time, pax if pax >= 0 else None)
        hits = [r for r in recs if r["_phase"] in ctx and matches(r, ctx[r["_phase"]])]
        best: Dict[Tuple, float] = {}
        for r in hits:
            best[r["_group"]] = max(best.get(r["_group"], -1e9), r["_prio"])
        for r in hits:
            if r["_prio"] < best[r["_group"]]:
                continue
            ph = r["_phase"]
            ref_b = e.arr.t if ph in (1, 2) else e.dep.t
            ref_e = e.dep.t if ph in (0, 2) else e.arr.t
            setup = int(_num(r.get("setup")) or 0)
            cleanup = int(_num(r.get("cleanup")) or 0)
            a = ref_b + int(_num(r.get("begin")) or 0) - setup
            b = ref_e + int(_num(r.get("end")) or 0) + cleanup
            dur_v = _num(r.get("duration"))
            dur = int(dur_v) + setup + cleanup if dur_v else b - a
            staff = int(_num(r.get("count")) or 0)
            if staff <= 0:
                continue
            if b - a <= 0 or dur <= 0 or dur > b - a:
                bad_window[g(r, "ruleDescription")] += 1
                continue
            fit = _fit_week(a, b, dur, circular)
            if fit is None:
                dropped += 1
                continue
            occ = (g(r, "mandQual1") or r["ResourceType"]) if plan_by == "qualification" \
                else r["ResourceType"]
            name = g(r, "engagegroups") or g(r, "ruleDescription")
            tasks.append(Task(len(tasks), name, occ, fit[0], fit[1], dur, staff,
                              e.label, e.stand, rule=g(r, "ruleDescription"),
                              department=r["ResourceType"]))
            use[(r["ResourceType"], g(r, "ruleDescription"), r["Row"])] += 1
    if bad_window:
        warns.append("Rules with an empty or reversed time window were skipped: " +
                     "; ".join(f"{k} ({v}×)" for k, v in sorted(bad_window.items())))
    if dropped:
        warns.append(f"{dropped} task(s) fall outside the planning week (turn on "
                     "*Circular week* to wrap them).")

    sm = defaultdict(lambda: [0, 0.0, 0])
    for t in tasks:
        k = (t.department, t.occupation)
        sm[k][0] += 1
        sm[k][1] += t.duration * t.staff / 60
        sm[k][2] = max(sm[k][2], t.staff)
    summary = pd.DataFrame([{"Department": d, "Plan group": o, "Tasks": v[0],
                             "Task hours": round(v[1], 1), "Max staff per task": v[2]}
                            for (d, o), v in sm.items()])
    rule_use = pd.DataFrame([{"Department": d, "Rule": desc, "File line": row, "Tasks": n}
                             for (d, desc, row), n in sorted(use.items(), key=lambda x: (x[0][0], x[0][2]))])
    return EngageResult(tasks, warns, summary, rule_use)
