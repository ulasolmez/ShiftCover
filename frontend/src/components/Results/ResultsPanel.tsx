import type { SolveResponse } from '../../types';
import { DAY_NAMES, OCC_COLORS, INTERVALS_PER_DAY } from '../../types';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend,
} from 'recharts';
import { AlertTriangle, CheckCircle } from 'lucide-react';

interface Props {
  result: SolveResponse;
  occNames: string[];
  prevResult: SolveResponse | null;
}

export default function ResultsPanel({ result, occNames, prevResult }: Props) {
  const { combined, occupations, combined_demand } = result;
  const isFeasible = combined.status === 'OPTIMAL' || combined.status === 'FEASIBLE';

  if (!isFeasible) {
    return (
      <div className="bg-red-900/30 border border-red-700 rounded-lg p-6">
        <div className="flex items-center gap-2 text-red-400 mb-2">
          <AlertTriangle size={20} />
          <h3 className="text-lg font-bold">Solver status: {combined.status}</h3>
        </div>
        <p className="text-sm text-gray-300">
          The model has no feasible solution with the current settings.
          Try relaxing constraints like wider shift ranges or removing headcount limits.
        </p>
      </div>
    );
  }

  const totalDemand = combined_demand.reduce((a, b) => a + b, 0);
  const totalCoverage = combined.coverage.reduce((a, b) => a + b, 0);

  return (
    <div className="space-y-6">
      {/* Metrics cards */}
      <div className="grid grid-cols-7 gap-3">
        <MetricCard label="Status" value={combined.status} />
        <MetricCard label="Active shifts" value={combined.shifts.length} />
        <MetricCard label="Total worker-h" value={combined.total_worker_hours.toFixed(0)} />
        <MetricCard label="Headcount" value={combined.max_headcount_day} />
        <MetricCard label="Peak sim." value={combined.peak_simultaneous} />
        <MetricCard label="FTE (÷45)" value={(combined.total_worker_hours / 45).toFixed(1)} />
        <MetricCard label="Solve time" value={`${combined.elapsed_sec.toFixed(1)}s`} />
      </div>

      {/* Coverage quality */}
      <div className="grid grid-cols-4 gap-3">
        <MetricCard label="Coverage" value={
          totalDemand > 0
            ? `${((1 - combined_demand.filter((d, i) => combined.coverage[i] < d).length / combined_demand.filter(Boolean).length) * 100).toFixed(1)}%`
            : '100%'
        } />
        <MetricCard label="Over-coverage" value={
          `${combined.coverage.reduce((a, c, i) => a + Math.max(0, c - combined_demand[i]), 0) / 12}`} />
      </div>

      {/* Per-occupation headcount */}
      {occupations.length > 1 && (
        <div className="grid grid-cols-5 gap-3">
          {occupations.map((occ, i) => (
            <div key={occ.name} className="bg-gray-800 rounded p-3">
              <p className="text-xs" style={{ color: OCC_COLORS[i] }}>{occ.name}</p>
              <p className="text-lg font-bold">{occ.phase1.max_headcount_day}</p>
              <p className="text-xs text-gray-400">{occ.phase1.total_worker_hours.toFixed(0)} wh</p>
            </div>
          ))}
        </div>
      )}

      {/* Weekly coverage chart */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
        <h3 className="text-sm font-semibold mb-2">Weekly coverage</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={buildWeeklyChartData(combined_demand, combined.coverage, occupations)}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="label" tick={{ fontSize: 10 }} interval={287} />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
            <Legend />
            <Line type="monotone" dataKey="demand" stroke="#ef4444" dot={false} name="Total demand" strokeWidth={2} />
            <Line type="monotone" dataKey="coverage" stroke="#22c55e" dot={false} name="Total coverage" strokeWidth={2} />
            {occupations.map((occ, i) => (
              <Line key={occ.name} type="monotone" dataKey={`occ_${i}`} stroke={OCC_COLORS[i]}
                dot={false} name={`${occ.name} coverage`} strokeWidth={1} strokeDasharray="4 4" />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Shift types table */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
        <h3 className="text-sm font-semibold mb-2">Shift types</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-gray-400 border-b border-gray-700">
                <th className="p-2">Shift Code</th>
                <th className="p-2">Day</th>
                <th className="p-2">Start</th>
                <th className="p-2">End</th>
                <th className="p-2">Duration</th>
                <th className="p-2">Workers</th>
                {occupations.length > 1 && <th className="p-2">Occupation</th>}
              </tr>
            </thead>
            <tbody>
              {combined.shifts.map((s, idx) => (
                <tr key={idx} className="border-b border-gray-800 hover:bg-gray-800/50">
                  <td className="p-2 font-mono">{s.shift_code}</td>
                  <td className="p-2">{DAY_NAMES[s.day]}</td>
                  <td className="p-2">{s.start_time_str}</td>
                  <td className="p-2">{s.end_time_str}</td>
                  <td className="p-2">{s.duration_hours.toFixed(1)}h</td>
                  <td className="p-2">{s.workers}</td>
                  {occupations.length > 1 && <td className="p-2 text-gray-400">—</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Previous comparison */}
      {prevResult && (
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
          <h3 className="text-sm font-semibold mb-2">🔄 Change vs previous solve</h3>
          <div className="grid grid-cols-5 gap-3">
            <MetricCard label="Active shifts" value={combined.shifts.length}
              delta={combined.shifts.length - prevResult.combined.shifts.length} />
            <MetricCard label="Worker-hours" value={combined.total_worker_hours.toFixed(0)}
              delta={+(combined.total_worker_hours - prevResult.combined.total_worker_hours).toFixed(0)} />
            <MetricCard label="Headcount" value={combined.max_headcount_day}
              delta={combined.max_headcount_day - prevResult.combined.max_headcount_day} invert />
            <MetricCard label="Peak sim." value={combined.peak_simultaneous}
              delta={combined.peak_simultaneous - prevResult.combined.peak_simultaneous} invert />
            <MetricCard label="Solve time" value={`${combined.elapsed_sec.toFixed(1)}s`}
              delta={+(combined.elapsed_sec - prevResult.combined.elapsed_sec).toFixed(1)} />
          </div>
        </div>
      )}
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function MetricCard({ label, value, delta, invert }: {
  label: string; value: string | number;
  delta?: number | string; invert?: boolean;
}) {
  const deltaNum = typeof delta === 'string' ? parseFloat(delta) : delta;
  const isPositive = deltaNum !== undefined && deltaNum > 0;
  const isNegative = deltaNum !== undefined && deltaNum < 0;
  const deltaColor = invert
    ? (isPositive ? 'text-red-400' : isNegative ? 'text-green-400' : 'text-gray-400')
    : (isPositive ? 'text-green-400' : isNegative ? 'text-red-400' : 'text-gray-400');

  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <p className="text-xs text-gray-400">{label}</p>
      <p className="text-xl font-bold">{value}</p>
      {delta !== undefined && delta !== 0 && (
        <p className={`text-xs ${deltaColor}`}>
          {deltaNum! > 0 ? '+' : ''}{delta}
        </p>
      )}
    </div>
  );
}

function buildWeeklyChartData(demand: number[], coverage: number[], occupations: any[]) {
  const data = [];
  for (let i = 0; i < demand.length; i += 12) {
    const entry: any = {
      label: `T${i}`,
      demand: demand[i],
      coverage: coverage[i],
    };
    occupations.forEach((occ, j) => {
      entry[`occ_${j}`] = occ.phase1.coverage[i];
    });
    data.push(entry);
  }
  return data;
}