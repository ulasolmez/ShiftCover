"""Pydantic models for the Simplex API."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class SolverParamsSchema(BaseModel):
    min_shift_hours: float = 3.0
    max_shift_hours: float = 12.0
    shift_start_granularity_min: int = 30
    shift_duration_step_min: int = 30
    max_unique_shifts: int = 0
    transition_penalty: int = 50
    solver_time_limit_sec: int = 120
    max_entries_per_day: Optional[List[int]] = None
    max_exits_per_day: Optional[List[int]] = None
    max_headcount_per_day: Optional[List[int]] = None
    min_headcount_per_day: Optional[List[int]] = None
    occ_max_headcount_per_day: Optional[List[Optional[List[int]]]] = None
    occ_min_headcount_per_day: Optional[List[Optional[List[int]]]] = None
    occ_headcount_per_shift_code: Optional[List[Optional[Dict]]] = None
    exclude_night_shifts: bool = False
    circular_week: bool = False
    force_include_shifts: Optional[List[str]] = None
    force_exclude_shifts: Optional[List[str]] = None
    allowed_slot_minutes: Optional[List[int]] = None


class SolveRequest(BaseModel):
    demands: List[List[int]] = Field(..., description="List of demand arrays, each length 2016")
    occ_names: List[str] = Field(..., description="Occupation names matching demands length")
    params: SolverParamsSchema = Field(default_factory=SolverParamsSchema)


class ShiftInfo(BaseModel):
    day: int
    start_time_str: str
    end_time_str: str
    duration_hours: float
    shift_code: str
    workers: int


class PhaseOneResultSchema(BaseModel):
    status: str
    elapsed_sec: float
    total_worker_hours: float
    shifts: List[ShiftInfo]
    max_headcount_day: int
    peak_simultaneous: int
    coverage: List[int]
    daily_entry_headcount: List[int]


class OccupationResultSchema(BaseModel):
    name: str
    phase1: PhaseOneResultSchema


class SolveResponse(BaseModel):
    combined: PhaseOneResultSchema
    occupations: List[OccupationResultSchema]
    combined_demand: List[int]


class ShiftCodePreviewRequest(BaseModel):
    params: SolverParamsSchema = Field(default_factory=SolverParamsSchema)


class ShiftCodePreviewResponse(BaseModel):
    codes: List[str]
    count: int


class SampleDemandRequest(BaseModel):
    peak_agents: int = 25
    base_agents: int = 3
    seed: int = 42


class SampleDemandResponse(BaseModel):
    demand: List[int]