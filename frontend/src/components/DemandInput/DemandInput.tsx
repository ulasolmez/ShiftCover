import { useState, useRef } from 'react';
import { Upload, RefreshCw } from 'lucide-react';
import { fetchSampleDemand } from '../../api/client';
import { TOTAL_INTERVALS, OCC_COLORS } from '../../types';

interface Props {
  nCurves: number;
  occNames: string[];
  demands: number[][] | null;
  onDemandsChange: (d: number[][]) => void;
}

export default function DemandInput({ nCurves, occNames, demands, onDemandsChange }: Props) {
  const [tab, setTab] = useState<'upload' | 'sample'>('upload');
  const [peaks, setPeaks] = useState<number[]>(Array(nCurves).fill(25));
  const [bases, setBases] = useState<number[]>(Array(nCurves).fill(3));
  const [seed, setSeed] = useState(42);
  const [loading, setLoading] = useState(false);
  const fileInputs = useRef<(HTMLInputElement | null)[]>([]);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const dems = await Promise.all(
        Array.from({ length: nCurves }, (_, i) =>
          fetchSampleDemand(peaks[i] || 25, bases[i] || 3, seed + i * 7)
        )
      );
      onDemandsChange(dems);
    } catch (err) {
      console.error('Failed to generate sample:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (i: number, file: File) => {
    const text = await file.text();
    const lines = text.split('\n').filter(l => l.trim());
    // Skip header if first line is non-numeric
    const values = lines.map(l => {
      const parts = l.split(',');
      const n = Number(parts[parts.length - 1]);
      return isNaN(n) ? 0 : Math.round(n);
    });
    if (values.length < TOTAL_INTERVALS) {
      alert(`File needs ${TOTAL_INTERVALS} rows, got ${values.length}`);
      return;
    }
    const next = demands ? [...demands] : Array(nCurves).fill([]);
    next[i] = values.slice(0, TOTAL_INTERVALS);
    onDemandsChange(next);
  };

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-3">Demand input</h2>

      <div className="flex gap-2 mb-4">
        <button onClick={() => setTab('upload')}
          className={`px-3 py-1 rounded text-sm ${tab === 'upload' ? 'bg-blue-600 text-white' : 'bg-gray-800'}`}>
          ⬆ Upload files
        </button>
        <button onClick={() => setTab('sample')}
          className={`px-3 py-1 rounded text-sm ${tab === 'sample' ? 'bg-blue-600 text-white' : 'bg-gray-800'}`}>
          🎲 Generate sample
        </button>
      </div>

      {tab === 'upload' && (
        <div className="space-y-3">
          <p className="text-xs text-gray-400">
            Upload <strong>{nCurves}</strong> file(s) — one per occupation. Each must have {TOTAL_INTERVALS} rows
            with a numeric column.
          </p>
          {Array.from({ length: nCurves }, (_, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="text-sm w-24" style={{ color: OCC_COLORS[i] }}>{occNames[i]}</span>
              <input
                type="file"
                accept=".csv,.xlsx"
                ref={el => { fileInputs.current[i] = el; }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleFileUpload(i, f);
                }}
                className="text-sm file:mr-3 file:px-3 file:py-1 file:bg-gray-700 file:text-gray-200 file:rounded file:border-0"
              />
              <Upload size={16} className="text-gray-500" />
            </div>
          ))}
        </div>
      )}

      {tab === 'sample' && (
        <div className="space-y-3">
          {Array.from({ length: nCurves }, (_, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="text-sm w-24" style={{ color: OCC_COLORS[i] }}>{occNames[i]}</span>
              <label className="text-xs">
                Peak
                <input type="number" min={5} max={60} value={peaks[i] ?? 25}
                  onChange={e => {
                    const next = [...peaks];
                    next[i] = Number(e.target.value);
                    setPeaks(next);
                  }}
                  className="w-16 ml-1 bg-gray-800 border border-gray-600 rounded px-1 text-xs" />
              </label>
              <label className="text-xs">
                Base
                <input type="number" min={0} max={10} value={bases[i] ?? 3}
                  onChange={e => {
                    const next = [...bases];
                    next[i] = Number(e.target.value);
                    setBases(next);
                  }}
                  className="w-16 ml-1 bg-gray-800 border border-gray-600 rounded px-1 text-xs" />
              </label>
            </div>
          ))}
          <label className="text-xs block">
            Random seed
            <input type="number" value={seed}
              onChange={e => setSeed(Number(e.target.value))}
              className="w-20 ml-2 bg-gray-800 border border-gray-600 rounded px-1" />
          </label>
          <button onClick={handleGenerate} disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white rounded text-sm">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            Generate sample
          </button>
        </div>
      )}

      {demands && demands.length > 0 && demands.some(d => d.length > 0) && (
        <p className="text-xs text-green-400 mt-3">
          {demands.length} curve(s) loaded ✓
        </p>
      )}
    </div>
  );
}