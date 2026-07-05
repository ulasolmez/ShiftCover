import { useState } from 'react';
import type { SolverParams } from '../../types';
import { DAY_SHORT, DEFAULT_PARAMS } from '../../types';
import { ChevronDown, ChevronRight, Settings, Calendar, Users, User, Clock } from 'lucide-react';

interface Props {
  params: SolverParams;
  onChange: (patch: Partial<SolverParams>) => void;
  nCurves: number;
  occNames: string[];
  onNCurvesChange: (n: number) => void;
  onOccNamesChange: (names: string[]) => void;
}

type Section = 'shift' | 'entry' | 'headcount' | 'occ-hc' | 'solver' | null;

export default function Sidebar({
  params, onChange, nCurves, occNames, onNCurvesChange, onOccNamesChange,
}: Props) {
  const [open, setOpen] = useState<Section>(null);

  const toggle = (s: Section) => setOpen((prev) => (prev === s ? null : s));

  return (
    <aside className="w-80 bg-gray-900 border-r border-gray-700 overflow-y-auto p-4 space-y-4 flex-shrink-0">
      <h2 className="text-lg font-bold">Parameters</h2>

      {/* Number of curves */}
      <label className="block text-sm">
        Number of occupation curves
        <input
          type="number" min={1} max={5} value={nCurves}
          onChange={(e) => {
            const n = Math.max(1, Math.min(5, Number(e.target.value)));
            onNCurvesChange(n);
            if (occNames.length < n) {
              const defaults = ['Technician', 'Labourer', 'Helper', 'Supervisor', 'Assistant'];
              onOccNamesChange([...occNames, ...defaults.slice(occNames.length, n)]);
            } else {
              onOccNamesChange(occNames.slice(0, n));
            }
          }}
          className="w-full mt-1 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm"
        />
      </label>

      {/* Occupation names */}
      {Array.from({ length: nCurves }, (_, i) => (
        <input
          key={i}
          type="text"
          value={occNames[i] || ''}
          onChange={(e) => {
            const next = [...occNames];
            next[i] = e.target.value;
            onOccNamesChange(next);
          }}
          placeholder={`Occupation ${i + 1}`}
          className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm"
        />
      ))}

      {/* Shift settings */}
      <SectionHeader icon={<Settings size={16} />} title="Shift settings" open={open === 'shift'} onToggle={() => toggle('shift')} />
      {open === 'shift' && (
        <div className="space-y-2 pl-2">
          <Slider label="Min shift (h)" min={3} max={8} step={0.5} value={params.min_shift_hours}
            onChange={(v) => onChange({ min_shift_hours: v })} />
          <Slider label="Max shift (h)" min={6} max={12} step={0.5} value={params.max_shift_hours}
            onChange={(v) => onChange({ max_shift_hours: v })} />
          <NumInput label="Max unique shifts (0=unlimited)" min={0} max={200} value={params.max_unique_shifts}
            onChange={(v) => onChange({ max_unique_shifts: v })} />
          <Checkbox label="Exclude night shifts (>8.5h, ≥50% night)" checked={params.exclude_night_shifts}
            onChange={(v) => onChange({ exclude_night_shifts: v })} />
          <Checkbox label="Circular week (Sun→Mon)" checked={params.circular_week}
            onChange={(v) => onChange({ circular_week: v })} />
        </div>
      )}

      {/* Entry / Exit limits */}
      <SectionHeader icon={<Calendar size={16} />} title="Entry / Exit limits per day" open={open === 'entry'} onToggle={() => toggle('entry')} />
      {open === 'entry' && (
        <div className="space-y-1 pl-2">
          {DAY_SHORT.map((day, d) => (
            <div key={day} className="flex gap-2 items-center">
              <span className="w-8 text-xs text-gray-400">{day}</span>
              <NumInput label="Entries" min={0} max={48} value={params.max_entries_per_day?.[d] ?? 0}
                onChange={(v) => {
                  const arr = [...(params.max_entries_per_day ?? [0,0,0,0,0,0,0])];
                  arr[d] = v;
                  onChange({ max_entries_per_day: arr });
                }} />
              <NumInput label="Exits" min={0} max={48} value={params.max_exits_per_day?.[d] ?? 0}
                onChange={(v) => {
                  const arr = [...(params.max_exits_per_day ?? [0,0,0,0,0,0,0])];
                  arr[d] = v;
                  onChange({ max_exits_per_day: arr });
                }} />
            </div>
          ))}
        </div>
      )}

      {/* Headcount limits */}
      <SectionHeader icon={<Users size={16} />} title="Headcount limits per day" open={open === 'headcount'} onToggle={() => toggle('headcount')} />
      {open === 'headcount' && (
        <div className="space-y-1 pl-2">
          {DAY_SHORT.map((day, d) => (
            <div key={day} className="flex gap-2 items-center">
              <span className="w-8 text-xs text-gray-400">{day}</span>
              <NumInput label="Max" min={0} max={500} value={params.max_headcount_per_day?.[d] ?? 0}
                onChange={(v) => {
                  const arr = [...(params.max_headcount_per_day ?? [0,0,0,0,0,0,0])];
                  arr[d] = v;
                  onChange({ max_headcount_per_day: arr });
                }} />
              <NumInput label="Min" min={0} max={500} value={params.min_headcount_per_day?.[d] ?? 0}
                onChange={(v) => {
                  const arr = [...(params.min_headcount_per_day ?? [0,0,0,0,0,0,0])];
                  arr[d] = v;
                  onChange({ min_headcount_per_day: arr });
                }} />
            </div>
          ))}
        </div>
      )}

      {/* Per-occupation headcount */}
      <SectionHeader icon={<User size={16} />} title="Per-occupation headcount" open={open === 'occ-hc'} onToggle={() => toggle('occ-hc')} />
      {open === 'occ-hc' && (
        <div className="space-y-3 pl-2">
          {Array.from({ length: nCurves }, (_, i) => (
            <div key={i}>
              <p className="text-xs font-semibold mb-1">{occNames[i]}</p>
              {DAY_SHORT.map((day, d) => (
                <NumInput key={day} label={day} min={0} max={500}
                  value={params.occ_max_headcount_per_day?.[i]?.[d] ?? 0}
                  onChange={(v) => {
                    const arr = [...(params.occ_max_headcount_per_day ?? Array(nCurves).fill(null))];
                    if (!arr[i]) arr[i] = [0,0,0,0,0,0,0];
                    arr[i]![d] = v;
                    onChange({ occ_max_headcount_per_day: arr });
                  }} />
              ))}
            </div>
          ))}
        </div>
      )}

      {/* Solver */}
      <SectionHeader icon={<Clock size={16} />} title="Solver" open={open === 'solver'} onToggle={() => toggle('solver')} />
      {open === 'solver' && (
        <div className="space-y-2 pl-2">
          <NumInput label="Time limit (s)" min={10} max={600} value={params.solver_time_limit_sec}
            onChange={(v) => onChange({ solver_time_limit_sec: v })} />
          <NumInput label="Transition penalty" min={0} max={500} value={params.transition_penalty}
            onChange={(v) => onChange({ transition_penalty: v })} />
        </div>
      )}
    </aside>
  );
}

// ── Reusable form controls ──────────────────────────────────────────────────

function SectionHeader({ icon, title, open, onToggle }: {
  icon: React.ReactNode; title: string; open: boolean; onToggle: () => void;
}) {
  return (
    <button onClick={onToggle} className="flex items-center gap-2 w-full text-left text-sm font-semibold hover:text-blue-400 py-1">
      {icon}
      <span className="flex-1">{title}</span>
      {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
    </button>
  );
}

function Slider({ label, min, max, step, value, onChange }: {
  label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void;
}) {
  return (
    <label className="block text-xs">
      {label}: <span className="font-mono">{value.toFixed(1)}</span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 bg-gray-700 rounded appearance-none cursor-pointer accent-blue-500"
      />
    </label>
  );
}

function NumInput({ label, min, max, value, onChange }: {
  label: string; min: number; max: number; value: number; onChange: (v: number) => void;
}) {
  return (
    <input
      type="number" min={min} max={max} value={value}
      onChange={(e) => onChange(Math.max(min, Math.min(max, Number(e.target.value) || 0)))}
      className="w-16 bg-gray-800 border border-gray-600 rounded px-1 py-0.5 text-xs text-center"
      title={label}
    />
  );
}

function Checkbox({ label, checked, onChange }: {
  label: string; checked: boolean; onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-xs cursor-pointer">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)}
        className="accent-blue-500" />
      {label}
    </label>
  );
}