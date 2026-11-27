# Simplex – Task-based Shift Optimiser

Branch `simplex-tasks`. Same app structure and shift rules as the curve-based
Simplex (`hcfinal-fixes`), but planned from **tasks** instead of a workload
curve, the way task-based demand works in GroundStar / Plan Control.

| | Curve-based Simplex | Task-based Simplex (this branch) |
|---|---|---|
| Input | Workload curve (5-min staff need) | Tasks with a time window, duration, staff, qualification |
| Decides | Shift counts | Shift counts **and** each task's start time |
| Checks | Curve covered every 5 min | Every task done whole by one worker, with travel time, briefing, close-out and breaks |
| Output | Shift types, coverage, ERP Shifttype | Same, plus task assignment, a worker Gantt and a GroundStar shift plan import file (.pls) |

## Input

* **Flights + task rules** – upload a flight schedule: the **INFORM flight
  export (.txt)** or a table (Flight, Day, STA, STD, Category/AC type, Stand).
  Tasks come from either
  * a **GroundStar engage-rule export** (`PBEngageRule`, one file with a table
    per department such as DIY_RAMP, DIY_TRF … — tables and columns in any
    order, UTF-16/UTF-8/Windows-1254). Rule phases (arrival / departure /
    ground time), ground-time, pax, cargo, airline, domestic/international,
    aircraft-group and flight-number-group filters are applied (`THYA = TKJ` is preset: AJet's old THY group is today's TKJ), and within an
    engage group only the highest-priority matching rule is used. You pick the
    departments and whether to plan per qualification or per department; or
  * the simple task-rule table in the app (relative to STA/STD, NB/WB staff).
* **Task list** – upload one row per task (Earliest start, Latest end,
  Duration, Staff, Qualification, Flight, Location).
* **Generate sample** – a hub-style week with arrival/departure banks.

## How it solves

1. **Shifts + task timing (MIP, SCIP bundled with OR-Tools).** Every task gets
   one start time in its window. The chosen starts form the task load per
   occupation every 5 minutes (including the walk to the stand). Shifts must
   cover that load with staff who are *available for tasks* (on shift, after
   briefing, before close-out, not on break). Extra window constraints make
   sure tasks can be done whole by one worker. Stage 1 minimises paid hours;
   stage 2 adds the transition penalty (same objective as the curve solver).
2. **Assignment check.** Tasks are given to individual workers (greedy, then
   CP-SAT per independent group of workers where needed). Breaks may move
   within the middle half of the shift.
3. **Repair.** A task that still cannot be staffed whole gets new window
   constraints and the plan is solved again. Anything left is fixed by
   lengthening one worker's shift or, if cheaper, adding a worker.

All shift rules of the curve app are supported: min/max shift length, allowed
start/end times, night-shift exclusion, circular week, force include/exclude,
max unique shift types, entry/exit limits, headcount limits (combined, per
occupation, per shift code) and the transition penalty.

## GroundStar shift plan import (.pls)

After a run, download the plan as `<from>_<to>_<RESOURCE>_SIMPLEX.pls` and load
it in GroundStar with **Shift plan import** (the dialog asks for a .pls or .nam
file). The file is GroundStar's `PLSHIFTS` text format (UTF-16, one line per
shift per day: id, shift code, date, begin, end, resource type,
qualification). Pick the planning week (it defaults to the INFORM export's
week) and the resource type. The writer reproduces the six PLSHIFTS files from
the user's scenario folders byte for byte.

**Plan with GroundStar's shift types only.** In the sidebar, upload
`Shifttype.Shifttype.GRule` (or any .pls/.sft) from a GroundStar scenario. The
solver then only uses shift codes that exist there, so every imported shift
matches a GroundStar shift type. Tasks that no such shift can reach (for
example, a task that starts right after a shift start and leaves no room for
the briefing) are listed and left out.

The older Plan Control document export (.sfp) is still in the app under
*advanced*. It needs a .sfp template and is for opening in Plan Control, not
for the shift plan import.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Community Cloud: deploy this repository with branch
`simplex-tasks` and main file `app.py`. No extra packages are needed – SCIP is
part of the `ortools` wheel.

## Files

| File | |
|---|---|
| `app.py` | Streamlit front-end (task-based) |
| `tasks.py` | Tasks, flights, task rules, imports, sample schedule |
| `groundstar_rules.py` | INFORM flight export + GroundStar engage-rule export → tasks |
| `pls_export.py` | GroundStar shift plan import (.pls / PLSHIFTS) reader and writer, shift-type reader |
| `sfp_export.py` | Plan Control shift plan (.sfp) writer |
| `task_solver.py` | Model, assignment check, repair, reports |
| `solver.py` | Shared shift machinery from the curve-based Simplex |
| `test_task_solver.py`, `test_groundstar_rules.py`, `test_sfp_export.py`, `test_pls_export.py` | Tests for the task-based parts |
