"""
Simplex (task-based) – task data
================================
Tasks are the input of the task-based optimiser.  A task is one piece of
work with a time window, e.g. "Loading TK1234, 3 staff, 25 min, somewhere
between 07:10 and 08:00".

Three ways to get tasks:

* **Flights + task rules** – the GroundStar way.  A flight schedule (STA /
  STD per flight) is combined with task rules relative to STA or STD
  ("Unloading: from STA+0 to STA+35, 20 min, 3 staff on a narrow-body").
* **Task list** – an export with one row per task (earliest start, latest
  end, duration, staff, qualification).
* **Sample** – a generated hub schedule with the default rules.

All times are minutes from Monday 00:00 of the planning week (0 … 10080).
"""

from __future__ import annotations

import datetime as dt
import io
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

WEEK_MIN = 7 * 1440
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]
DAY_ABBR3 = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Aircraft types treated as wide-body when only a type code is given
_WB_HINTS = ("330", "332", "333", "338", "339", "340", "343", "346", "350",
             "359", "35K", "380", "388", "747", "744", "748", "767", "763",
             "764", "777", "772", "773", "77W", "77L", "787", "788", "789",
             "78X", "WB", "WIDE", "HEAVY")


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class Task:
    tid: int
    name: str
    occupation: str
    earliest: int          # earliest start, minutes from Monday 00:00
    latest_end: int        # latest end, minutes from Monday 00:00
    duration: int          # minutes
    staff: int = 1
    flight: str = ""
    location: str = ""
    rule: str = ""               # source rule (engage-rule import)
    department: str = ""         # resource type (engage-rule import)

    @property
    def slack(self) -> int:
        return max(0, self.latest_end - self.earliest - self.duration)

    @property
    def work_minutes(self) -> int:
        return self.duration * self.staff


@dataclass
class Flight:
    flight: str
    day: int                     # 0 = Monday (day of STA, or STD if no STA)
    sta: Optional[int] = None    # minutes from Monday 00:00
    std: Optional[int] = None
    category: str = "NB"         # NB / WB
    stand: str = ""


RULE_COLUMNS = ["Use", "Occupation", "Task", "Ref", "From (min)",
                "To (min)", "Duration (min)", "Staff NB", "Staff WB"]


def default_rules() -> pd.DataFrame:
    """GroundStar-style demand rules relative to STA / STD.

    From / To are offsets of the task window from the reference time:
    the task may start at Ref+From at the earliest and must be finished
    by Ref+To.  When To - From equals the duration the task is fixed.
    """
    rows = [
        # Use   Occupation   Task               Ref   From   To  Dur NB WB
        (True,  "Ramp",      "Unloading",       "STA",   0,  40, 20, 3, 5),
        (True,  "Ramp",      "Loading",         "STD", -55, -10, 25, 3, 5),
        (True,  "Ramp",      "Pushback",        "STD", -10,  15, 15, 1, 2),
        (True,  "Cleaning",  "Cabin cleaning",  "STA",   5,  55, 20, 2, 4),
        (False, "Pax",       "Check-in",        "STD", -150, -45, 105, 2, 4),
        (False, "Pax",       "Boarding",        "STD", -45,  -5, 40, 2, 3),
        (False, "Technician", "Turnaround check", "STA", 0,  45, 25, 1, 2),
    ]
    return pd.DataFrame(rows, columns=RULE_COLUMNS)


# ── Small parsing helpers ────────────────────────────────────────────────────
def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _find_col(df: pd.DataFrame, *aliases: str) -> Optional[str]:
    """First column whose normalised name equals (or starts with) an alias."""
    cols = {_norm(c): c for c in df.columns}
    for a in aliases:
        if _norm(a) in cols:
            return cols[_norm(a)]
    for a in aliases:
        na = _norm(a)
        for nc, c in cols.items():
            if nc.startswith(na):
                return c
    return None


def _day_index(v) -> Optional[int]:
    """Weekday from 'Mon', 'Monday', 1-7 (Mon=1), a date or a datetime."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (dt.date, dt.datetime, pd.Timestamp)):
        return pd.Timestamp(v).weekday()
    s = str(v).strip()
    if not s:
        return None
    if s.isdigit():
        n = int(s)
        return (n - 1) % 7 if 1 <= n <= 7 else None
    s3 = s[:3].lower()
    for i, d in enumerate(DAY_ABBR3):
        if d.lower() == s3:
            return i
    try:
        return pd.Timestamp(s).weekday()
    except Exception:
        return None


def _clock_minutes(v) -> Optional[int]:
    """Minutes after midnight from '07:35', '0735', a time or a datetime."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (dt.datetime, pd.Timestamp)):
        return v.hour * 60 + v.minute
    if isinstance(v, dt.time):
        return v.hour * 60 + v.minute
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if 0 <= v < 1:                       # Excel fraction of a day
            return int(round(v * 1440))
        n = int(v)
        if n <= 2400 and n % 100 < 60:       # 735 → 07:35
            return (n // 100) * 60 + n % 100
        return None
    s = str(v).strip()
    if not s:
        return None
    m = re.match(r"^(\d{1,2})[:.h](\d{2})", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    if re.match(r"^\d{3,4}$", s):
        n = int(s)
        return (n // 100) * 60 + n % 100
    try:
        t = pd.Timestamp(s)
        return t.hour * 60 + t.minute
    except Exception:
        return None


def _has_date(v) -> bool:
    if isinstance(v, (dt.datetime, pd.Timestamp)):
        return True
    if isinstance(v, str) and re.search(r"\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}[./]\d{2,4}", v):
        return True
    return False


def _as_timestamp(v) -> Optional[pd.Timestamp]:
    try:
        t = pd.Timestamp(v)
        return None if pd.isna(t) else t
    except Exception:
        return None


def _category(v) -> str:
    s = str(v or "").upper()
    if s in ("WB", "W", "WIDE", "WIDEBODY", "WIDE-BODY", "H", "HEAVY"):
        return "WB"
    if s in ("NB", "N", "NARROW", "NARROWBODY", "NARROW-BODY", "M"):
        return "NB"
    return "WB" if any(h in s for h in _WB_HINTS) else "NB"


def _read_table(data: bytes, file_name: str) -> pd.DataFrame:
    if file_name.lower().endswith(".csv"):
        txt = data.decode("utf-8-sig", errors="replace")
        sep = ";" if txt.count(";") > txt.count(",") else ","
        return pd.read_csv(io.StringIO(txt), sep=sep)
    xls = pd.ExcelFile(io.BytesIO(data))
    # Pick the first sheet that has a header row with more than two columns
    for sh in xls.sheet_names:
        df = xls.parse(sh)
        if df.shape[1] >= 3 and len(df) > 0:
            return df
    return xls.parse(xls.sheet_names[0])


def week_starts_of(values) -> List[pd.Timestamp]:
    """Mondays of all weeks touched by the dated values."""
    mondays = set()
    for v in values:
        t = _as_timestamp(v) if _has_date(v) else None
        if t is not None:
            mondays.add((t.normalize() - pd.Timedelta(days=t.weekday())))
    return sorted(mondays)


def _week_minute(v, day_hint, week_start: Optional[pd.Timestamp]) -> Optional[int]:
    """Minute of the planning week for a time value.

    Dated values are placed relative to ``week_start`` (values outside the
    week return None).  Undated clock times use ``day_hint``.
    """
    if _has_date(v):
        t = _as_timestamp(v)
        if t is None:
            return None
        if week_start is None:
            return t.weekday() * 1440 + t.hour * 60 + t.minute
        delta = (t - week_start).total_seconds() / 60
        return int(round(delta))
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 2400:
        return int(v)                            # already week minutes
    cm = _clock_minutes(v)
    if cm is None or day_hint is None:
        return None
    return day_hint * 1440 + cm


# ── Flights ──────────────────────────────────────────────────────────────────
def read_flights(data: bytes, file_name: str,
                 week_start: Optional[pd.Timestamp] = None
                 ) -> Tuple[List[Flight], List[str], List[pd.Timestamp]]:
    """Read a flight schedule (CSV/XLSX).

    Recognised columns (case-insensitive, first match wins):
      Flight / Flight no / Arrival flight / Departure flight
      Day / Date / Weekday
      STA / Arrival / Scheduled arrival        (time or date-time)
      STD / Departure / Scheduled departure    (time or date-time)
      Category / Body / AC type / Aircraft     (NB / WB or type code)
      Stand / Position / Gate                  (optional)

    Returns (flights, warnings, week_starts_found).
    """
    df = _read_table(data, file_name)
    warns: List[str] = []
    c_fl = _find_col(df, "Flight", "Flight no", "Flight number", "Flightno",
                     "Arrival flight", "Arr flight", "Departure flight", "Dep flight",
                     "Callsign")
    c_fl2 = _find_col(df, "Departure flight", "Dep flight")
    c_day = _find_col(df, "Day", "Weekday", "Date", "Day of week", "Ops date")
    c_sta = _find_col(df, "STA", "Scheduled arrival", "Arrival time", "Arrival",
                      "Arr", "Onblock", "On block")
    c_std = _find_col(df, "STD", "Scheduled departure", "Departure time",
                      "Departure", "Dep", "Offblock", "Off block")
    c_cat = _find_col(df, "Category", "Body", "Size", "AC category",
                      "Aircraft category", "AC type", "Aircraft type",
                      "Aircraft", "Equipment", "Type")
    c_st = _find_col(df, "Stand", "Position", "Parking position", "Gate")
    if c_sta is None and c_std is None:
        raise ValueError("No STA or STD column found. Expected columns like "
                         "'Flight', 'Day', 'STA', 'STD', 'Category', 'Stand'.")

    time_vals = []
    for c in (c_sta, c_std):
        if c is not None:
            time_vals.extend(df[c].tolist())
    starts = week_starts_of(time_vals)
    if week_start is None and starts:
        week_start = starts[0]

    flights: List[Flight] = []
    skipped = 0
    for i, r in df.iterrows():
        day_hint = _day_index(r[c_day]) if c_day is not None else None
        sta = _week_minute(r[c_sta], day_hint, week_start) if c_sta else None
        std = _week_minute(r[c_std], day_hint, week_start) if c_std else None
        if sta is not None and std is not None and std < sta \
                and not _has_date(r[c_std]):
            std += 1440                          # departs after midnight
        if sta is None and std is None:
            skipped += 1
            continue
        ref = sta if sta is not None else std
        if ref < -1440 or ref >= WEEK_MIN + 1440:
            skipped += 1
            continue
        parts = []
        for c in (c_fl, c_fl2):
            if c is not None:
                v = str(r[c]).strip()
                if v and v.lower() not in ("nan", "none") and v not in parts:
                    parts.append(v)
        name = "/".join(parts) or f"F{i + 1}"
        flights.append(Flight(
            flight=name,
            day=max(0, min(6, ref // 1440)),
            sta=sta, std=std,
            category=_category(r[c_cat]) if c_cat is not None else "NB",
            stand=str(r[c_st]).strip() if c_st is not None and
            str(r[c_st]).strip().lower() != "nan" else "",
        ))
    if skipped:
        warns.append(f"{skipped} row(s) skipped (no usable STA/STD in the week).")
    if c_cat is None:
        warns.append("No aircraft category column — all flights treated as "
                      "narrow-body (NB).")
    return flights, warns, starts


def flights_to_dataframe(flights: List[Flight]) -> pd.DataFrame:
    def _fmt(m):
        if m is None:
            return ""
        d, r = divmod(m, 1440)
        return f"{DAY_ABBR3[d % 7]} {r // 60:02d}:{r % 60:02d}"
    return pd.DataFrame([{
        "Flight": f.flight, "Day": DAY_NAMES[f.day],
        "STA": _fmt(f.sta), "STD": _fmt(f.std),
        "Category": f.category, "Stand": f.stand,
    } for f in flights])


def tasks_from_flights(flights: List[Flight], rules: pd.DataFrame,
                       circular: bool = False) -> Tuple[List[Task], List[str]]:
    """Apply task rules to a flight schedule."""
    warns: List[str] = []
    tasks: List[Task] = []
    use = rules[rules["Use"].astype(bool)] if "Use" in rules else rules
    dropped = 0
    for f in flights:
        for _, r in use.iterrows():
            ref = f.sta if str(r["Ref"]).upper() == "STA" else f.std
            if ref is None:
                continue
            staff = int(r["Staff WB"] if f.category == "WB" else r["Staff NB"])
            dur = int(r["Duration (min)"])
            if staff <= 0 or dur <= 0:
                continue
            e = ref + int(r["From (min)"])
            le = ref + int(r["To (min)"])
            if le - e < dur:
                le = e + dur
            t = _fit_week(e, le, dur, circular)
            if t is None:
                dropped += 1
                continue
            e, le = t
            tasks.append(Task(len(tasks), str(r["Task"]), str(r["Occupation"]).strip(),
                              e, le, dur, staff, f.flight, f.stand))
    if dropped:
        warns.append(f"{dropped} task(s) fall outside the planning week and were "
                     "left out (turn on *Circular week* to wrap them).")
    return tasks, warns


def _fit_week(e: int, le: int, dur: int, circular: bool) -> Optional[Tuple[int, int]]:
    """Clip a task window to the week (or wrap it when circular)."""
    if circular:
        shift = (e // WEEK_MIN) * WEEK_MIN
        return e - shift, le - shift
    e2, le2 = max(0, e), min(WEEK_MIN, le)
    if le2 - e2 < dur:
        return None
    return e2, le2


# ── Task list import ────────────────────────────────────────────────────────
def read_task_list(data: bytes, file_name: str,
                   week_start: Optional[pd.Timestamp] = None,
                   circular: bool = False
                   ) -> Tuple[List[Task], List[str], List[pd.Timestamp]]:
    """Read a task list (CSV/XLSX), one row per task.

    Recognised columns:
      Task / Name / Activity
      Occupation / Qualification / Skill / Resource / Department
      Day                              (if times are clock times)
      Earliest start / Start / Begin   (time or date-time)
      Latest end / End / Finish        (time or date-time)
      Duration (min)                   (optional — default End - Start)
      Staff / Quantity / Persons       (optional — default 1)
      Flight, Location / Stand         (optional)
    """
    df = _read_table(data, file_name)
    warns: List[str] = []
    c_name = _find_col(df, "Task", "Task name", "Name", "Activity", "Service")
    c_occ = _find_col(df, "Occupation", "Qualification", "Skill", "Resource",
                      "Resource type", "Department", "Group")
    c_day = _find_col(df, "Day", "Weekday", "Date")
    c_es = _find_col(df, "Earliest start", "Earliest", "Start", "Begin",
                     "From", "Window start")
    c_le = _find_col(df, "Latest end", "Latest", "End", "Finish", "To",
                     "Window end")
    c_dur = _find_col(df, "Duration (min)", "Duration", "Dur", "Length")
    c_st = _find_col(df, "Staff", "Quantity", "Qty", "Persons", "Number",
                     "Headcount", "Resources")
    c_fl = _find_col(df, "Flight", "Flight no")
    c_loc = _find_col(df, "Location", "Stand", "Position", "Gate")
    if c_es is None or c_le is None:
        raise ValueError("A task list needs at least an earliest-start and a "
                         "latest-end column (e.g. 'Earliest start', 'Latest end').")
    starts = week_starts_of(df[c_es].tolist() + df[c_le].tolist())
    if week_start is None and starts:
        week_start = starts[0]
    tasks: List[Task] = []
    skipped = 0
    for i, r in df.iterrows():
        day_hint = _day_index(r[c_day]) if c_day is not None else None
        e = _week_minute(r[c_es], day_hint, week_start)
        le = _week_minute(r[c_le], day_hint, week_start)
        if e is None or le is None:
            skipped += 1
            continue
        if le <= e and not _has_date(r[c_le]):
            le += 1440
        dur = le - e
        if c_dur is not None:
            try:
                dv = float(r[c_dur])
                if not np.isnan(dv) and dv > 0:
                    dur = int(round(dv))
            except (TypeError, ValueError):
                cm = _clock_minutes(r[c_dur])       # "00:25"
                if cm:
                    dur = cm
        staff = 1
        if c_st is not None:
            try:
                staff = max(1, int(float(r[c_st])))
            except (TypeError, ValueError):
                pass
        if le - e < dur:
            le = e + dur
        t = _fit_week(e, le, dur, circular)
        if t is None or dur <= 0:
            skipped += 1
            continue
        occ = str(r[c_occ]).strip() if c_occ is not None else "Staff"
        tasks.append(Task(
            len(tasks),
            str(r[c_name]).strip() if c_name is not None else f"Task {i + 1}",
            occ if occ and occ.lower() != "nan" else "Staff",
            t[0], t[1], dur, staff,
            str(r[c_fl]).strip() if c_fl is not None and str(r[c_fl]) != "nan" else "",
            str(r[c_loc]).strip() if c_loc is not None and str(r[c_loc]) != "nan" else "",
        ))
    if skipped:
        warns.append(f"{skipped} row(s) skipped (missing times or outside the week).")
    if c_occ is None:
        warns.append("No qualification/occupation column — all tasks planned "
                     "as one group 'Staff'.")
    return tasks, warns, starts


def tasks_to_dataframe(tasks: List[Task]) -> pd.DataFrame:
    def _fmt(m):
        d, r = divmod(int(m), 1440)
        return f"{DAY_ABBR3[d % 7]} {r // 60:02d}:{r % 60:02d}"
    df = pd.DataFrame([{
        "Task": t.name, "Occupation": t.occupation, "Flight": t.flight,
        "Location": t.location, "Earliest start": _fmt(t.earliest),
        "Latest end": _fmt(t.latest_end), "Duration (min)": t.duration,
        "Staff": t.staff, "Slack (min)": t.slack,
        "Department": t.department, "Rule": t.rule,
    } for t in tasks])
    if not df.empty and not (df["Rule"] != "").any():
        df = df.drop(columns=["Department", "Rule"])
    return df


def task_list_csv(tasks: List[Task]) -> bytes:
    """Task list in the import format (Day + clock times)."""
    rows = []
    for t in tasks:
        d, s = divmod(t.earliest, 1440)
        e = t.latest_end - d * 1440
        rows.append({
            "Task": t.name, "Occupation": t.occupation, "Flight": t.flight,
            "Location": t.location, "Day": DAY_ABBR3[d % 7],
            "Earliest start": f"{s // 60:02d}:{s % 60:02d}",
            "Latest end": f"{(e // 60) % 24:02d}:{e % 60:02d}",
            "Duration (min)": t.duration, "Staff": t.staff,
        })
    return pd.DataFrame(rows).to_csv(index=False).encode()


def flights_csv(flights: List[Flight]) -> bytes:
    rows = []
    for f in flights:
        def _c(m):
            if m is None:
                return ""
            r = m % 1440
            return f"{r // 60:02d}:{r % 60:02d}"
        rows.append({"Flight": f.flight, "Day": DAY_ABBR3[f.day],
                     "STA": _c(f.sta), "STD": _c(f.std),
                     "Category": f.category, "Stand": f.stand})
    return pd.DataFrame(rows).to_csv(index=False).encode()


def occupations_of(tasks: List[Task]) -> List[str]:
    seen: Dict[str, None] = {}
    for t in tasks:
        seen.setdefault(t.occupation, None)
    return list(seen)


def earliest_load(tasks: List[Task], occupation: Optional[str] = None,
                  step: int = 5) -> np.ndarray:
    """Staff needed every ``step`` minutes if every task starts as early as it can."""
    n = WEEK_MIN // step
    out = np.zeros(n, dtype=int)
    for t in tasks:
        if occupation is not None and t.occupation != occupation:
            continue
        a = t.earliest // step
        b = -(-(t.earliest + t.duration) // step)
        for k in range(a, b):
            out[k % n] += t.staff
    return out


# ── Sample schedule ──────────────────────────────────────────────────────────
def generate_sample_flights(flights_per_day: int = 30, wb_share: float = 0.15,
                            seed: int = 42) -> List[Flight]:
    """Hub-style week: departures and arrivals in banks, lighter weekends."""
    rng = np.random.default_rng(seed)
    banks = [(6 * 60 + 30, 0.22), (9 * 60 + 45, 0.24), (13 * 60 + 15, 0.18),
             (16 * 60 + 45, 0.22), (20 * 60 + 30, 0.14)]
    stands = [f"{a}{n}" for a in "ABC" for n in range(1, 9)]
    flights: List[Flight] = []
    for day in range(7):
        n = int(round(flights_per_day * (0.75 if day >= 5 else 1.0)
                      * rng.uniform(0.9, 1.1)))
        centres = [b for b, _ in banks]
        probs = np.array([p for _, p in banks])
        probs = probs / probs.sum()
        for k in range(n):
            wb = rng.random() < wb_share
            ground = int(rng.integers(90, 135)) if wb else int(rng.integers(40, 65))
            c = centres[rng.choice(len(centres), p=probs)]
            sta = day * 1440 + c + int(rng.normal(0, 25))
            sta = int(round(sta / 5) * 5)
            kind = rng.random()
            if kind < 0.07:                        # originating (first wave)
                std = day * 1440 + 6 * 60 + int(rng.integers(0, 12)) * 5
                fl = Flight(f"S{day + 1}{k:02d}", day, None, std,
                            "WB" if wb else "NB", stands[k % len(stands)])
            elif kind < 0.13:                      # night stop
                fl = Flight(f"S{day + 1}{k:02d}", day, max(day * 1440, min(
                    sta + 6 * 60, day * 1440 + 23 * 60 + 30)), None,
                    "WB" if wb else "NB", stands[k % len(stands)])
            else:
                fl = Flight(f"S{day + 1}{k:02d}", day, sta, sta + ground,
                            "WB" if wb else "NB", stands[k % len(stands)])
            flights.append(fl)
    flights.sort(key=lambda f: (f.sta if f.sta is not None else f.std))
    return flights
