"""
Solve endpoints – run the shift-covering solver and stream logs via SSE.
"""

import asyncio
import threading
import queue
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from backend.schemas import (
    SolveRequest, SolveResponse, PhaseOneResultSchema,
    OccupationResultSchema, ShiftInfo,
)
from solver import (
    SolverParams, solve_multi, TOTAL_INTERVALS,
    max_headcount, daily_entry_headcount,
)
from backend.sample_data import generate_sample_demand

router = APIRouter()


def _params_to_solver(params_schema) -> SolverParams:
    """Convert Pydantic schema to SolverParams dataclass."""
    return SolverParams(
        min_shift_hours=params_schema.min_shift_hours,
        max_shift_hours=params_schema.max_shift_hours,
        shift_start_granularity_min=params_schema.shift_start_granularity_min,
        shift_duration_step_min=params_schema.shift_duration_step_min,
        max_unique_shifts=params_schema.max_unique_shifts,
        transition_penalty=params_schema.transition_penalty,
        solver_time_limit_sec=params_schema.solver_time_limit_sec,
        max_entries_per_day=params_schema.max_entries_per_day,
        max_exits_per_day=params_schema.max_exits_per_day,
        max_headcount_per_day=params_schema.max_headcount_per_day,
        min_headcount_per_day=params_schema.min_headcount_per_day,
        occ_max_headcount_per_day=params_schema.occ_max_headcount_per_day,
        occ_min_headcount_per_day=params_schema.occ_min_headcount_per_day,
        occ_headcount_per_shift_code=params_schema.occ_headcount_per_shift_code,
        exclude_night_shifts=params_schema.exclude_night_shifts,
        circular_week=params_schema.circular_week,
        force_include_shifts=params_schema.force_include_shifts,
        force_exclude_shifts=params_schema.force_exclude_shifts,
        allowed_slot_minutes=params_schema.allowed_slot_minutes,
    )


def _phase1_to_schema(p1) -> PhaseOneResultSchema:
    """Convert PhaseOneResult to Pydantic schema."""
    shifts_info = []
    for s, cnt in zip(p1.shifts, p1.counts):
        if cnt > 0:
            shifts_info.append(ShiftInfo(
                day=s.day,
                start_time_str=s.start_time_str,
                end_time_str=s.end_time_str,
                duration_hours=s.duration_hours,
                shift_code=s.shift_code,
                workers=cnt,
            ))
    return PhaseOneResultSchema(
        status=p1.status,
        elapsed_sec=p1.elapsed_sec,
        total_worker_hours=p1.total_worker_intervals / 12.0,
        shifts=shifts_info,
        max_headcount_day=max_headcount(p1),
        peak_simultaneous=int(p1.coverage.max()),
        coverage=p1.coverage.astype(int).tolist(),
        daily_entry_headcount=daily_entry_headcount(p1),
    )


@router.post("/solve", response_model=SolveResponse)
async def solve_endpoint(req: SolveRequest):
    """Solve the shift-covering problem and return results."""
    # Validate inputs
    if not req.demands:
        raise HTTPException(status_code=400, detail="No demands provided")
    if len(req.demands) != len(req.occ_names):
        raise HTTPException(status_code=400,
                           detail="Number of demands must match number of occ_names")
    for i, d in enumerate(req.demands):
        if len(d) != TOTAL_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Demand[{i}] must have {TOTAL_INTERVALS} elements, got {len(d)}"
            )

    demands = [np.array(d, dtype=int) for d in req.demands]
    params = _params_to_solver(req.params)

    try:
        result = solve_multi(demands, req.occ_names, params)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Solver error: {exc}")

    combined = _phase1_to_schema(result.combined_phase1)
    occupations = [
        OccupationResultSchema(
            name=occ.name,
            phase1=_phase1_to_schema(occ.phase1),
        )
        for occ in result.occupations
    ]
    return SolveResponse(
        combined=combined,
        occupations=occupations,
        combined_demand=result.combined_demand.astype(int).tolist(),
    )


@router.post("/solve/stream")
async def solve_stream_endpoint(req: SolveRequest):
    """Solve the shift-covering problem with SSE streaming of log messages."""

    # Validate inputs (same as sync endpoint)
    if not req.demands:
        raise HTTPException(status_code=400, detail="No demands provided")
    if len(req.demands) != len(req.occ_names):
        raise HTTPException(status_code=400,
                           detail="Number of demands must match number of occ_names")
    for i, d in enumerate(req.demands):
        if len(d) != TOTAL_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Demand[{i}] must have {TOTAL_INTERVALS} elements, got {len(d)}"
            )

    demands = [np.array(d, dtype=int) for d in req.demands]
    params = _params_to_solver(req.params)

    log_queue = queue.Queue()
    result_holder = {}

    def _run_solver():
        try:
            def _cb(msg: str):
                log_queue.put({"type": "log", "message": msg})

            result = solve_multi(demands, req.occ_names, params, callback=_cb)
            result_holder["result"] = result
            log_queue.put({"type": "done"})
        except Exception as exc:
            log_queue.put({"type": "error", "message": str(exc)})

    thread = threading.Thread(target=_run_solver, daemon=True)
    thread.start()

    async def _event_generator():
        while True:
            try:
                item = log_queue.get(timeout=0.5)
                if item["type"] == "log":
                    yield {"event": "log", "data": item["message"]}
                elif item["type"] == "done":
                    result = result_holder.get("result")
                    if result:
                        combined = _phase1_to_schema(result.combined_phase1)
                        occupations = [
                            OccupationResultSchema(
                                name=occ.name,
                                phase1=_phase1_to_schema(occ.phase1),
                            )
                            for occ in result.occupations
                        ]
                        resp = SolveResponse(
                            combined=combined,
                            occupations=occupations,
                            combined_demand=result.combined_demand.astype(int).tolist(),
                        )
                        yield {"event": "result",
                               "data": resp.model_dump_json()}
                    yield {"event": "close", "data": ""}
                    return
                elif item["type"] == "error":
                    yield {"event": "error", "data": item["message"]}
                    yield {"event": "close", "data": ""}
                    return
            except queue.Empty:
                if not thread.is_alive():
                    yield {"event": "close", "data": ""}
                    return
                # Send heartbeat to keep connection alive
                yield {"event": "heartbeat", "data": ""}
            await asyncio.sleep(0.1)

    return EventSourceResponse(_event_generator())


@router.post("/sample-demand")
async def sample_demand_endpoint(
    peak_agents: int = 25,
    base_agents: int = 3,
    seed: int = 42,
):
    """Generate a single sample demand curve."""
    demand = generate_sample_demand(
        peak_agents=peak_agents,
        base_agents=base_agents,
        seed=seed,
    )
    return {"demand": demand.astype(int).tolist()}