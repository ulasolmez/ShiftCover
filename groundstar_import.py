"""Import workload curves straight from GroundStar exports.

GroundStar's "Workload" Excel export has a sheet called *Curve Data*:

    rows 0..~19   metadata  (Resource Type, Time Period, Start of Day, ...)
    a few rows    series info: type / role (TotalSum, SumOfVisible,
                  Standard) / visible flag / colours
    header row    Date | Time | Minutes Since Start of Day | Workload |
                  Sum of Visible | <curve 1> | <curve 2> | ...
    data rows     one per time step (normally 5 min)

The *Standard* columns are the individual curves (one per resource type,
qualification, task ... depending on "Curve Building Criteria").  TotalSum /
SumOfVisible are aggregates and are only used when a file has no Standard
curve.

Plain CSV / XLSX files with a single numeric column (the old format) are
still accepted and become one curve each.
"""
from __future__ import annotations

import datetime as dt
import io
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import openpyxl

INTERVAL_MIN = 5
INTERVALS_PER_WEEK = 7 * 24 * 60 // INTERVAL_MIN      # 2016

_AGG_ROLES = {"totalsum", "sumofvisible"}
_INDEX_HEADERS = {"date", "time", "minutes since start of day"}


@dataclass
class Curve:
    """One workload curve inside an imported file."""
    name: str                      # column header, e.g. "DIY_RAMP" or "CLN"
    role: str                      # "Standard", "TotalSum", "SumOfVisible", "Plain"
    series: pd.Series              # values indexed by absolute timestamp
    step_min: int = INTERVAL_MIN


@dataclass
class WorkloadFile:
    file_name: str
    kind: str                      # "groundstar" or "plain"
    meta: Dict[str, object] = field(default_factory=dict)
    curves: List[Curve] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def resource_type(self) -> str:
        return str(self.meta.get("Resource Type") or "").strip()

    def default_curves(self) -> List[Curve]:
        """Curves to offer by default: Standard ones, else the total."""
        std = [c for c in self.curves if c.role in ("Standard", "Plain")]
        if std:
            return std
        tot = [c for c in self.curves if c.role.lower() == "totalsum"]
        return tot or self.curves[:1]

    def week_starts(self) -> List[dt.datetime]:
        """Mondays 00:00 whose full Mon-Sun week lies inside the data."""
        if not self.curves or self.kind != "groundstar":
            return []
        idx = self.curves[0].series.index
        first, last = idx.min(), idx.max()
        span = last - first + pd.Timedelta(minutes=self.curves[0].step_min)
        if span < pd.Timedelta(days=7):
            return []
        # A Monday qualifies if a full 7 x 24 h of data starts on or after it
        # and before the next day boundary.  With "Start of Day" 03:00 the
        # export runs Mon 03:00 → Mon 02:55; the Mon 00:00–02:55 slots are
        # then filled from the same clock time one week later (the week's
        # night tail), exactly like the solver's circular week.
        d = first.normalize() - pd.Timedelta(days=first.weekday())
        out = []
        while d <= last:
            win_start = max(d, first)
            if (win_start - d < pd.Timedelta(days=1)
                    and win_start + pd.Timedelta(days=7) - pd.Timedelta(minutes=INTERVAL_MIN)
                    <= last + pd.Timedelta(minutes=self.curves[0].step_min - INTERVAL_MIN)):
                out.append(d.to_pydatetime())
            d += pd.Timedelta(days=7)
        return out


# ── Helpers ──────────────────────────────────────────────────────────────────
def _name(f) -> str:
    return getattr(f, "name", None) or os.path.basename(str(f))


def _read_bytes(f) -> bytes:
    if isinstance(f, (str, os.PathLike)):
        with open(f, "rb") as fh:
            return fh.read()
    if hasattr(f, "getvalue"):
        return f.getvalue()
    pos = f.tell() if hasattr(f, "tell") else None
    data = f.read()
    if pos is not None and hasattr(f, "seek"):
        f.seek(pos)
    return data


def _to_time(v) -> Optional[dt.time]:
    if isinstance(v, dt.time):
        return v
    if isinstance(v, dt.datetime):
        return v.time()
    if isinstance(v, (int, float)) and 0 <= v < 1:          # Excel day fraction
        secs = int(round(v * 86400))
        return dt.time(secs // 3600 % 24, secs // 60 % 60)
    if isinstance(v, str) and ":" in v:
        h, m = v.strip().split(":")[:2]
        return dt.time(int(h), int(m))
    return None


def _to_date(v) -> Optional[dt.datetime]:
    if isinstance(v, dt.datetime):
        return v.replace(hour=0, minute=0, second=0, microsecond=0)
    if isinstance(v, dt.date):
        return dt.datetime(v.year, v.month, v.day)
    if isinstance(v, str):
        try:
            return pd.to_datetime(v, dayfirst=True).to_pydatetime().replace(
                hour=0, minute=0, second=0, microsecond=0)
        except (ValueError, TypeError):
            return None
    return None


def _find_curve_sheet(wb) -> Optional[str]:
    names = wb.sheetnames
    # 1) the export's own translation table knows the localised sheet name
    if "Translations" in names:
        for row in wb["Translations"].iter_rows(values_only=True):
            if row and row[0] == "CurveDataSheet.Name" and row[1] in names:
                return row[1]
    # 2) English default
    for n in names:
        if n.strip().lower() == "curve data":
            return n
    # 3) any sheet with a Date | Time | Minutes... header
    for n in names:
        for i, row in enumerate(wb[n].iter_rows(max_row=60, values_only=True)):
            cells = [str(c).strip().lower() for c in row[:3] if c is not None]
            if len(cells) == 3 and cells[0] == "date" and cells[1] == "time":
                return n
    return None


# ── GroundStar reader ────────────────────────────────────────────────────────
def _read_groundstar(data: bytes, file_name: str, sheet: str, wb) -> WorkloadFile:
    rows = list(wb[sheet].iter_rows(values_only=True))
    wf = WorkloadFile(file_name, "groundstar")

    # Header row: first row whose first cells are Date | Time (| Minutes ...)
    hdr_i = None
    for i, r in enumerate(rows):
        c0 = str(r[0]).strip().lower() if r and r[0] is not None else ""
        c1 = str(r[1]).strip().lower() if len(r) > 1 and r[1] is not None else ""
        if c0 == "date" and c1 == "time":
            hdr_i = i
            break
    if hdr_i is None:
        raise ValueError(f"{file_name}: no 'Date | Time' header in sheet '{sheet}'")

    # Metadata: key/value rows above the series block
    for r in rows[:hdr_i]:
        if r and isinstance(r[0], str) and r[0].strip():
            vals = [v for v in r[1:] if v is not None]
            wf.meta[r[0].strip()] = vals[0] if len(vals) == 1 else (vals or None)

    header = [str(c).strip() if c is not None else "" for c in rows[hdr_i]]
    has_minutes = len(header) > 2 and header[2].lower() == "minutes since start of day"
    first_val_col = 3 if has_minutes else 2

    # Series roles: row above the header that mentions TotalSum / Standard
    roles: Dict[int, str] = {}
    for r in rows[max(0, hdr_i - 8):hdr_i]:
        vals = [str(c).strip() if c is not None else "" for c in r]
        if any(v.lower() in _AGG_ROLES | {"standard"} for v in vals):
            for j, v in enumerate(vals):
                if v:
                    roles[j] = v
            break

    sod = _to_time(wf.meta.get("Start of Day")) or dt.time(0, 0)
    sod_min = sod.hour * 60 + sod.minute

    stamps, data_rows = [], []
    for r in rows[hdr_i + 1:]:
        if not r or r[0] is None:
            continue
        day = _to_date(r[0])
        if day is None:
            continue
        if has_minutes and isinstance(r[2], (int, float)):
            ts = day + dt.timedelta(minutes=sod_min + int(r[2]))
        else:
            t = _to_time(r[1])
            if t is None:
                continue
            m = t.hour * 60 + t.minute
            if m < sod_min:                      # belongs to the next calendar day
                m += 1440
            ts = day + dt.timedelta(minutes=m)
        stamps.append(ts)
        data_rows.append(r)
    if not stamps:
        raise ValueError(f"{file_name}: '{sheet}' has no data rows")

    idx = pd.DatetimeIndex(stamps)
    diffs = (np.diff(idx.values).astype("timedelta64[s]").astype(np.int64) // 60
             if len(idx) > 1 else np.array([INTERVAL_MIN]))
    diffs = diffs[diffs > 0]
    step = int(pd.Series(diffs).mode().iloc[0]) if len(diffs) else INTERVAL_MIN
    if step != INTERVAL_MIN:
        wf.warnings.append(f"{file_name}: data is in {step}-min steps — "
                           f"converted to 5-min steps.")

    for j in range(first_val_col, len(header)):
        name = header[j]
        if not name:
            continue
        vals = pd.to_numeric(pd.Series([r[j] if j < len(r) else None for r in data_rows]),
                             errors="coerce")
        if vals.isna().all():
            continue
        role = roles.get(j, "Standard")
        if name.lower() in ("workload", "sum of visible") and j < first_val_col + 2 \
                and role == "Standard":
            role = "TotalSum" if name.lower() == "workload" else "SumOfVisible"
        s = pd.Series(vals.to_numpy(dtype=float), index=idx)
        s = s[~s.index.duplicated(keep="first")].sort_index()
        wf.curves.append(Curve(name, role, s, step))

    if not wf.curves:
        raise ValueError(f"{file_name}: no numeric workload columns in '{sheet}'")
    return wf


# ── Plain file reader (old single-column format) ─────────────────────────────
_KNOWN = ("required", "demand", "workers", "staff", "agents",
          "headcount", "count", "value", "workload")
_SKIP = ("interval", "index", "id", "row", "day", "daynum",
         "minutes since start of day")


def _read_plain(data: bytes, file_name: str) -> WorkloadFile:
    wf = WorkloadFile(file_name, "plain")
    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data), skip_blank_lines=False)
    else:
        df = pd.read_excel(io.BytesIO(data))
    lower = {str(c).strip().lower(): c for c in df.columns}
    col = next((lower[k] for k in _KNOWN if k in lower), None)
    if col is None:
        num = [c for c in df.select_dtypes(include="number").columns
               if str(c).strip().lower() not in _SKIP]
        if not num:
            raise ValueError(f"{file_name}: no numeric demand column found")
        col = num[-1]
    ser = pd.to_numeric(df[col], errors="coerce")
    if len(ser) < INTERVALS_PER_WEEK:
        raise ValueError(f"{file_name}: need {INTERVALS_PER_WEEK} rows, got {len(ser)}")
    if len(ser) > INTERVALS_PER_WEEK:
        wf.warnings.append(f"{file_name}: {len(ser)} rows — only the first "
                           f"{INTERVALS_PER_WEEK} (Mon 00:00 → Sun 23:55) are used.")
    ser = ser.iloc[:INTERVALS_PER_WEEK].reset_index(drop=True)
    n_bad = int(ser.isna().sum())
    if n_bad:
        wf.warnings.append(f"{file_name}: {n_bad} blank/non-numeric value(s) treated as 0.")
    n_neg = int((ser < 0).sum())
    if n_neg:
        wf.warnings.append(f"{file_name}: {n_neg} negative value(s) clipped to 0.")
    stem = os.path.splitext(os.path.basename(file_name))[0]
    wf.curves.append(Curve(stem, "Plain", ser.fillna(0).clip(lower=0)))
    return wf


# ── Public API ───────────────────────────────────────────────────────────────
def read_workload_file(f) -> WorkloadFile:
    """Read a GroundStar workload export or a plain CSV/XLSX curve."""
    file_name = _name(f)
    data = _read_bytes(f)
    if file_name.lower().endswith((".xlsx", ".xlsm")):
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        try:
            sheet = _find_curve_sheet(wb)
            if sheet is not None:
                return _read_groundstar(data, file_name, sheet, wb)
        finally:
            wb.close()
    return _read_plain(data, file_name)


def curve_week(curve: Curve, week_start: Optional[dt.datetime]) -> np.ndarray:
    """Return the curve as 2016 five-minute values, Mon 00:00 → Sun 23:55.

    * Plain curves are already in that layout.
    * GroundStar curves are cut to the calendar week starting ``week_start``
      (a Monday).  If ``week_start`` is None, each weekday is taken from its
      first occurrence in the file (for exports that don't start on Monday).
    * Coarser steps are held for their whole step; finer steps take the max.
    * Fractional workload is rounded UP (2.1 people needed → 3 on shift).
    """
    if curve.role == "Plain":
        vals = curve.series.to_numpy(dtype=float)
    else:
        s = curve.series.fillna(0)
        if curve.step_min < INTERVAL_MIN:
            s = s.resample(f"{INTERVAL_MIN}min").max()
        if week_start is not None:
            grid = pd.date_range(pd.Timestamp(week_start),
                                 periods=INTERVALS_PER_WEEK, freq=f"{INTERVAL_MIN}min")
            # slots before the export's first row (Start of Day > 00:00) take
            # the value from one week later
            early = grid[grid < s.index.min()]
            if len(early):
                s = pd.concat([pd.Series(s.reindex(early + pd.Timedelta(days=7)).to_numpy(),
                                         index=early), s]).sort_index()
        else:
            first = s.index.min().normalize()
            # Monday on/after which each weekday's first occurrence starts
            grid = []
            for wd in range(7):
                day0 = first + pd.Timedelta(days=(wd - first.weekday()) % 7)
                grid.extend(pd.date_range(day0, periods=288, freq=f"{INTERVAL_MIN}min"))
            grid = pd.DatetimeIndex(grid)
        # hold each sample for its whole step (15/30/60-min exports);
        # 5-min data is used as is, with missing slots = 0
        hold = curve.step_min // INTERVAL_MIN - 1
        full = s.reindex(s.index.union(grid))
        if hold > 0:
            full = full.ffill(limit=hold)
        vals = full.reindex(grid).fillna(0).to_numpy()
    vals = np.clip(vals, 0, None)
    return np.ceil(vals - 1e-9).astype(int)


def combine_by_occupation(selections: List[dict]) -> Dict[str, np.ndarray]:
    """Sum the selected curves per occupation name, keeping first-seen order.

    ``selections``: dicts with keys ``occupation`` and ``demand`` (np.ndarray).
    """
    out: Dict[str, np.ndarray] = {}
    for sel in selections:
        name = str(sel["occupation"]).strip()
        if not name:
            continue
        out[name] = out.get(name, 0) + np.asarray(sel["demand"], dtype=int)
    return out
