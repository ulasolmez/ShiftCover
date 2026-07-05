// ── Solver parameter types ────────────────────────────────────────────────
export interface SolverParams {
  min_shift_hours: number;
  max_shift_hours: number;
  shift_start_granularity_min: number;
  shift_duration_step_min: number;
  max_unique_shifts: number;
  transition_penalty: number;
  solver_time_limit_sec: number;
  max_entries_per_day: number[] | null;
  max_exits_per_day: number[] | null;
  max_headcount_per_day: number[] | null;
  min_headcount_per_day: number[] | null;
  occ_max_headcount_per_day: (number[] | null)[] | null;
  occ_min_headcount_per_day: (number[] | null)[] | null;
  occ_headcount_per_shift_code: (Record<string, any> | null)[] | null;
  exclude_night_shifts: boolean;
  circular_week: boolean;
  force_include_shifts: string[] | null;
  force_exclude_shifts: string[] | null;
  allowed_slot_minutes: number[] | null;
}

// ── Shift info ─────────────────────────────────────────────────────────────
export interface ShiftInfo {
  day: number;
  start_time_str: string;
  end_time_str: string;
  duration_hours: number;
  shift_code: string;
  workers: number;
}

// ── Phase 1 result ─────────────────────────────────────────────────────────
export interface PhaseOneResult {
  status: string;
  elapsed_sec: number;
  total_worker_hours: number;
  shifts: ShiftInfo[];
  max_headcount_day: number;
  peak_simultaneous: number;
  coverage: number[];
  daily_entry_headcount: number[];
}

// ── Occupation result ──────────────────────────────────────────────────────
export interface OccupationResult {
  name: string;
  phase1: PhaseOneResult;
}

// ── Solve response ─────────────────────────────────────────────────────────
export interface SolveResponse {
  combined: PhaseOneResult;
  occupations: OccupationResult[];
  combined_demand: number[];
}

// ── Solve request ──────────────────────────────────────────────────────────
export interface SolveRequest {
  demands: number[][];
  occ_names: string[];
  params: SolverParams;
}

// ── SSE event types ────────────────────────────────────────────────────────
export type SseEvent =
  | { type: 'log'; message: string }
  | { type: 'result'; data: SolveResponse }
  | { type: 'error'; message: string }
  | { type: 'close' }
  | { type: 'heartbeat' };

// ── Day names ──────────────────────────────────────────────────────────────
export const DAY_NAMES = [
  'Monday', 'Tuesday', 'Wednesday', 'Thursday',
  'Friday', 'Saturday', 'Sunday',
] as const;
export const DAY_SHORT = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] as const;

// ── Occupation colors ──────────────────────────────────────────────────────
export const OCC_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];

// ── Default solver params ──────────────────────────────────────────────────
export const DEFAULT_PARAMS: SolverParams = {
  min_shift_hours: 3.0,
  max_shift_hours: 12.0,
  shift_start_granularity_min: 30,
  shift_duration_step_min: 30,
  max_unique_shifts: 0,
  transition_penalty: 50,
  solver_time_limit_sec: 120,
  max_entries_per_day: null,
  max_exits_per_day: null,
  max_headcount_per_day: null,
  min_headcount_per_day: null,
  occ_max_headcount_per_day: null,
  occ_min_headcount_per_day: null,
  occ_headcount_per_shift_code: null,
  exclude_night_shifts: false,
  circular_week: false,
  force_include_shifts: null,
  force_exclude_shifts: null,
  allowed_slot_minutes: null,
};

export const TOTAL_INTERVALS = 2016;
export const INTERVALS_PER_DAY = 288;