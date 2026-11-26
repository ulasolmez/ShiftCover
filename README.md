# Simplex – Task-based Shift Optimiser

Branch `simplex-tasks`. Same app structure and shift rules as the curve-based
Simplex (`hcfinal-fixes`), but planned from **tasks** instead of a workload
curve, the way task-based demand works in GroundStar / Plan Control.

| | Curve-based Simplex | Task-based Simplex (this branch) |
|---|---|---|
| Input | Workload curve (5-min staff need) | Tasks with a time window, duration, staff, qualification |
| Decides | Shift counts | Shift counts **and** each task's start time |
| Checks | Curve covered every 5 min | Every task done whole by one worker, with travel time, briefing, close-out and breaks |
| Output | Shift types, coverage, ERP Shifttype | Same, plus task assignment and a worker Gantt |

## Input

* **Flights + task rules** – upload a flight schedule (Flight, Day, STA, STD,
  Category/AC type, Stand). Task rules relative to STA/STD turn each flight
  into tasks, e.g. *Unloading: STA+0 … STA+40, 20 min, 3 staff (NB) / 5 (WB)*.
  The rules are editable in the app.
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
| `task_solver.py` | Model, assignment check, repair, reports |
| `solver.py` | Shared shift machinery from the curve-based Simplex |
| `test_task_solver.py` | Tests for the task-based parts |
