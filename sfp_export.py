"""
GroundStar Plan Control shift-plan export (.sfp)
================================================

An ``.sfp`` file is the MFC archive Plan Control saves (``PCShiftDoc``).  We
write our shift plan into a template ``.sfp`` exported from GroundStar:

* everything before the shift list (curve display, head-count rules, shift
  restrictions, task chains, planning parameters …) is copied unchanged;
* the shift list (``CShiftList``) is replaced by our shifts.

Layout of one shift (``CShift``), as found in Plan Control 7.9 files
    base-class tags · id −1 · qualification · resource type ·
    7 weekday flags (Sunday first: the days the shift code is valid) ·
    code "HHMM-HHMM" · start / end (minutes after midnight) · colours ·
    shift id · task list
A shift is one instance on one day.  Its day is carried by a day task
(task type 11) in its task list: day number (1 = first day of the plan
period, a Monday) and the shift's window in minutes from the period start.
Every exported shift gets exactly that one day task; flight tasks are not
exported – Plan Control assigns them again from its own flights and rules.

The writer is checked against the template: every shift header and day
task of the template is rebuilt byte for byte before it is used.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

DAY_TASK_TYPE = 11
_COLOURS = (18, 24, 55, 99)


# ── MFC archive primitives ───────────────────────────────────────────────────
def _cstr(s: str) -> bytes:
    """MFC CString, Unicode: FF FE FF + length + UTF-16LE."""
    raw = s.encode("utf-16-le")
    n = len(s)
    if n < 0xFF:
        return b"\xff\xfe\xff" + bytes([n]) + raw
    if n < 0xFFFF:
        return b"\xff\xfe\xff\xff" + struct.pack("<H", n) + raw
    return b"\xff\xfe\xff\xff\xff\xff" + struct.pack("<I", n) + raw


def _i32(*v: int) -> bytes:
    return struct.pack("<" + "i" * len(v), *v)


def _count(n: int) -> bytes:
    return struct.pack("<H", n) if n < 0xFFFF else b"\xff\xff" + struct.pack("<I", n)


class _Reader:
    def __init__(self, b: bytes, p: int = 0):
        self.b, self.p = b, p

    def u8(self):
        v = self.b[self.p]; self.p += 1; return v

    def u16(self):
        v = struct.unpack_from("<H", self.b, self.p)[0]; self.p += 2; return v

    def i32(self):
        v = struct.unpack_from("<i", self.b, self.p)[0]; self.p += 4; return v

    def raw(self, n):
        v = self.b[self.p:self.p + n]; self.p += n; return v

    def cstr(self) -> str:
        if self.b[self.p:self.p + 3] != b"\xff\xfe\xff":
            raise ValueError(f"string expected at byte {self.p}")
        self.p += 3
        n = self.u8()
        if n == 0xFF:
            n = self.u16()
            if n == 0xFFFF:
                n = struct.unpack_from("<I", self.b, self.p)[0]; self.p += 4
        v = self.b[self.p:self.p + 2 * n].decode("utf-16-le"); self.p += 2 * n
        return v


# ── Records ──────────────────────────────────────────────────────────────────
@dataclass
class ShiftRecord:
    qualification: str
    resource_type: str
    valid_days: Tuple[int, ...]    # 7 flags, Sunday first
    code: str
    start: int                     # minutes after midnight
    end: int
    shift_id: int
    day: int                       # 1 = first day of the period (Monday)
    task_id: int
    group_id: int
    break_min: int = 0

    @property
    def length(self) -> int:
        return (self.end - self.start) % 1440 or 1440


def shift_header(r: ShiftRecord, base_tags: bytes, n_tasks: int) -> bytes:
    """CShift body after its class tag, up to and including the task count."""
    return b"".join([
        base_tags,
        _i32(-1), b"\x00" * 9, _cstr(""), _i32(1), b"\x00\x00",
        _cstr(r.qualification), b"\x00",
        _cstr(""), _cstr(""), _cstr(""), _cstr(r.resource_type),
        _i32(0, 0, 0, 0, *r.valid_days, 0, 0, 0),
        _cstr(r.code), _i32(r.start, r.end, 1),
        _cstr(""), _cstr(""), _cstr(""), b"\x00" * 28,
        _cstr(""), _i32(*_COLOURS),
        _cstr(""), _cstr(""), _cstr(""),
        _i32(0, r.shift_id), _count(n_tasks),
    ])


def day_task(r: ShiftRecord, task_tag: bytes, base_tags: bytes) -> bytes:
    """The day task (type 11) that places the shift on its day."""
    w0 = (r.day - 1) * 1440 + r.start
    w1 = w0 + r.length
    return b"".join([
        task_tag, base_tags,
        _cstr(""), b"\xff" * 8, b"\x00" * 24, b"\xff" * 4,
        _cstr(""), b"\xff" * 4, b"\x00" * 12, b"\xff" * 4, b"\x00" * 8,
        _i32(w1, w1), b"\x00" * 16,
        _cstr(""), b"\xff" * 4, b"\x00" * 12,
        _cstr(""), _cstr(""), b"\x00" * 8, b"\xff" * 4, b"\x00" * 8,
        _cstr(""), _cstr(""), _cstr(""),
        _i32(r.task_id, r.group_id, 1, 2, 0), b"\x00\x00",
        _i32(DAY_TASK_TYPE), b"\x00" * 4, _i32(r.break_min, w1),
        b"\xff" * 12, _i32(1, 2), b"\x00\x00",
        _cstr(r.resource_type), _i32(r.day), b"\x00" * 6, _i32(w0, w1), b"\x00" * 8,
    ])


# ── Template ─────────────────────────────────────────────────────────────────
@dataclass
class SfpTemplate:
    head: bytes                 # everything before the CShiftList declaration
    list_decl: bytes            # FF FF schema len "CShiftList"
    shift_decl: bytes           # FF FF schema len "CShift" (first shift)
    shift_tag: bytes            # class tag of CShift for later shifts
    task_tag: bytes             # class tag of CTask
    base_tags: bytes            # tags of BTGCOb, BTGCModel, BTModel
    resource_type: str
    n_shifts: int
    shift_id_base: int
    task_id_base: int
    group_id_base: int
    qualifications: List[str]
    checked_shifts: int = 0
    checked_day_tasks: int = 0


def _decl(b: bytes, name: str) -> Optional[re.Match]:
    return re.search(rb"\xff\xff(..)" + struct.pack("<H", len(name)) + name.encode(), b, re.S)


def _parse_shift(b: bytes, p: int):
    """Read a CShift body (after its class tag); returns fields and end of the task count."""
    r = _Reader(b, p)
    base = r.raw(6)
    r.i32(); r.raw(9); r.cstr(); r.i32(); r.u16()
    qual = r.cstr(); r.u8()
    r.cstr(); r.cstr(); r.cstr()
    rt = r.cstr()
    mid = struct.unpack_from("<14i", b, r.p); r.p += 56
    code = r.cstr(); start = r.i32(); end = r.i32(); r.i32()
    r.cstr(); r.cstr(); r.cstr(); r.raw(28); r.cstr(); r.raw(16)
    r.cstr(); r.cstr(); r.cstr()
    r.i32(); sid = r.i32(); n = r.u16()
    return dict(base=base, qual=qual, rt=rt, flags=tuple(mid[4:11]), code=code,
                start=start, end=end, sid=sid, n_tasks=n, end_pos=r.p, start_pos=p)


def read_template(data: bytes) -> SfpTemplate:
    """Read a Plan Control .sfp and check that our writer reproduces it."""
    if b"PCShiftDoc" not in data[:64]:
        raise ValueError("Not a GroundStar Plan Control shift plan (.sfp): "
                         "PCShiftDoc header missing.")
    m_list = _decl(data, "CShiftList")
    if not m_list:
        raise ValueError("No shift list (CShiftList) in this .sfp.")
    p = m_list.end()
    n_shifts = struct.unpack_from("<H", data, p)[0]
    m_shift = _decl(data[p:p + 64], "CShift")
    if n_shifts < 2 or not m_shift:
        raise ValueError("The template needs at least two shifts.")
    first = p + m_shift.end()
    base_tags = data[first:first + 6]
    shift_starts = [first]
    s0 = _parse_shift(data, first)
    # later shifts: <CShift tag> + base tags; find the tag from the second shift
    tag_re = re.compile(rb"(?=(..)" + re.escape(base_tags) + rb"\xff\xff\xff\xff\x00{9}\xff\xfe\xff\x00)", re.S)
    later = [m for m in tag_re.finditer(data, s0["end_pos"]) if m.group(1)[1] & 0x80]
    if not later:
        raise ValueError("Could not find the second shift in the template.")
    shift_tag = later[0].group(1)
    shift_starts += [m.start() + 2 for m in later if m.group(1) == shift_tag]
    task_tag = data[s0["end_pos"]:s0["end_pos"] + 2]
    if data[s0["end_pos"] + 2:s0["end_pos"] + 8] != base_tags:
        raise ValueError("Unexpected task layout in the template.")

    tpl = SfpTemplate(
        head=data[:m_list.start()], list_decl=data[m_list.start():m_list.end()],
        shift_decl=data[p + m_shift.start():p + m_shift.end()], shift_tag=shift_tag,
        task_tag=task_tag, base_tags=base_tags, resource_type=s0["rt"],
        n_shifts=n_shifts, shift_id_base=0, task_id_base=0, group_id_base=0,
        qualifications=[])
    if len(shift_starts) != n_shifts:
        raise ValueError(f"Template lists {n_shifts} shifts but {len(shift_starts)} were found.")

    # Check: rebuild every shift header and every day task byte for byte
    task_hdr = task_tag + base_tags + _cstr("")
    quals, sids, tids, gids = {}, [], [], []
    checked_days = 0
    bounds = shift_starts[1:] + [len(data) + 2]
    for k, (sp, nxt) in enumerate(zip(shift_starts, bounds)):
        f = _parse_shift(data, sp)
        rec = ShiftRecord(f["qual"], f["rt"], f["flags"], f["code"], f["start"], f["end"],
                          f["sid"], 1, 0, 0)
        if data[sp:f["end_pos"]] != shift_header(rec, base_tags, f["n_tasks"]):
            raise ValueError(f"Shift {k + 1} ({f['code']}) has a layout this exporter does "
                             "not know — the .sfp version may differ.")
        quals.setdefault(f["qual"], None)
        sids.append(f["sid"])
        body = data[f["end_pos"]:nxt - 2]
        q = 0
        while True:
            q = body.find(task_hdr, q)
            if q < 0:
                break
            seg = body[q:q + 400]
            try:
                r = _Reader(seg, 0)
                day = struct.unpack_from("<i", seg, 244 + 2 * (len(f["rt"]) - 8))[0]
                tid = struct.unpack_from("<i", seg, 164)[0]
                gid = struct.unpack_from("<i", seg, 168)[0]
                bmin = struct.unpack_from("<i", seg, 194)[0]
                cand = ShiftRecord(f["qual"], f["rt"], f["flags"], f["code"], f["start"],
                                   f["end"], f["sid"], day, tid, gid, bmin)
                built = day_task(cand, task_tag, base_tags)
                if seg[:len(built)] == built:
                    checked_days += 1
                    tids.append(tid)
                    gids.append(gid)
            except (struct.error, ValueError):
                pass
            q += 1
    if checked_days == 0:
        raise ValueError("No day task in the template matches the known layout.")
    tpl.qualifications = list(quals)
    tpl.shift_id_base = max(sids) + 1
    tpl.task_id_base = max(tids) + 1
    tpl.group_id_base = max(gids) + 1
    tpl.checked_shifts = n_shifts
    tpl.checked_day_tasks = checked_days
    return tpl


# ── Writer ───────────────────────────────────────────────────────────────────
def build_sfp(tpl: SfpTemplate, shifts: List[ShiftRecord]) -> bytes:
    if not shifts:
        raise ValueError("No shifts to export.")
    out = [tpl.head, tpl.list_decl, _count(len(shifts))]
    for k, r in enumerate(shifts):
        out.append(tpl.shift_decl if k == 0 else tpl.shift_tag)
        out.append(shift_header(r, tpl.base_tags, 1))
        out.append(day_task(r, tpl.task_tag, tpl.base_tags))
    return b"".join(out)


def read_sfp_shifts(data: bytes) -> List[Dict]:
    """Shifts of an .sfp written by build_sfp (one day task per shift)."""
    tpl_list = _decl(data, "CShiftList")
    p = tpl_list.end()
    n = struct.unpack_from("<H", data, p)[0]
    p += 2
    m = _decl(data[p:p + 64], "CShift")
    p += m.end()
    out = []
    for k in range(n):
        if k:
            p += 2
        f = _parse_shift(data, p)
        rt_extra = 2 * (len(f["rt"]) - 8)
        q = f["end_pos"]
        day = struct.unpack_from("<i", data, q + 244 + rt_extra)[0]
        w0, w1 = struct.unpack_from("<2i", data, q + 254 + rt_extra)
        f.update(day=day, window=(w0, w1))
        out.append(f)
        p = q + 270 + rt_extra
    if p != len(data):
        raise ValueError(f"{len(data) - p} unread bytes at the end")
    return out


def shift_records(res, tpl: SfpTemplate, resource_type: Optional[str] = None,
                  qualification_from_plan: bool = True,
                  occupations: Optional[set] = None) -> List[ShiftRecord]:
    """One record per worker shift of a task_solver.TaskResult (or curve result).

    Valid-day flags of a code are the days the plan uses it.  The plan week
    starts on Monday = day 1.
    """
    from solver import break_hours
    multi = getattr(res, "multi", res)
    rt_default = resource_type or tpl.resource_type
    dept_of = {}
    for t in getattr(res, "tasks", []) or []:
        if getattr(t, "department", ""):
            dept_of.setdefault(t.occupation, t.department)
    days_of_code: Dict[str, set] = {}
    occs = [o for o in multi.occupations if occupations is None or o.name in occupations]
    for occ in occs:
        for s, c in zip(occ.phase1.shifts, occ.phase1.counts):
            if c:
                days_of_code.setdefault(s.shift_code, set()).add(s.day)
    recs: List[ShiftRecord] = []
    sid, tid, gid = tpl.shift_id_base, tpl.task_id_base, tpl.group_id_base
    for occ in occs:
        rt = resource_type or dept_of.get(occ.name) or rt_default
        qual = occ.name if qualification_from_plan and occ.name != rt else ""
        for s, c in sorted(zip(occ.phase1.shifts, occ.phase1.counts),
                           key=lambda x: (x[0].day, x[0].start_interval, x[0].duration_intervals)):
            if not c:
                continue
            flags = tuple(int(((d - 1) % 7) in days_of_code[s.shift_code]) for d in range(7))
            start = s.start_interval * 5
            end = (s.start_interval + s.duration_intervals) * 5 % 1440
            bmin = int(round(break_hours(s.duration_hours) * 60))
            for _ in range(c):
                recs.append(ShiftRecord(qual, rt, flags, s.shift_code, start, end,
                                        sid, s.day + 1, tid, gid, bmin))
                sid += 1
                tid += 1
                gid += 1
    return recs
