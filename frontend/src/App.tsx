import { useState, useCallback, useRef } from 'react';
import type { SolverParams, SolveResponse, OccupationResult } from './types';
import { DEFAULT_PARAMS /*, DAY_NAMES */ } from './types';
import { healthCheck } from './api/client';
import Sidebar from './components/Sidebar/Sidebar';
import DemandInput from './components/DemandInput/DemandInput';
import ResultsPanel from './components/Results/ResultsPanel';
import SolveButton from './components/SolveButton';

export default function App() {
  // ── State ─────────────────────────────────────────────────────────────
  const [params, setParams] = useState<SolverParams>(() => {
    // Restore from sessionStorage if available
    const saved = sessionStorage.getItem('simplex_params');
    return saved ? { ...DEFAULT_PARAMS, ...JSON.parse(saved) } : { ...DEFAULT_PARAMS };
  });
  const [nCurves, setNCurves] = useState(1);
  const [occNames, setOccNames] = useState<string[]>(['Technician']);
  const [demands, setDemands] = useState<number[][] | null>(null);
  const [result, setResult] = useState<SolveResponse | null>(null);
  const [prevResult, setPrevResult] = useState<SolveResponse | null>(null);
  const [prevParams, setPrevParams] = useState<SolverParams | null>(null);
  const [isSolving, setIsSolving] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const cancelRef = useRef<(() => void) | null>(null);

  // ── Persist params ────────────────────────────────────────────────────
  const updateParams = useCallback((patch: Partial<SolverParams>) => {
    setParams((p) => {
      const next = { ...p, ...patch };
      sessionStorage.setItem('simplex_params', JSON.stringify(next));
      return next;
    });
  }, []);

  // ── Solve handler ─────────────────────────────────────────────────────
  const handleSolve = useCallback(async () => {
    if (!demands) return;
    setIsSolving(true);
    setLogs([]);
    setResult(null);

    // Dynamic import for streamSolve to keep bundle split
    const { streamSolve } = await import('./api/client');

    const cancel = streamSolve(
      { demands, occ_names: occNames, params },
      (event) => {
        switch (event.type) {
          case 'log':
            setLogs((prev) => [...prev, event.message]);
            break;
          case 'result':
            setPrevResult(result);
            setPrevParams(params);
            setResult(event.data);
            setIsSolving(false);
            break;
          case 'error':
            setLogs((prev) => [...prev, `ERROR: ${event.message}`]);
            setIsSolving(false);
            break;
          case 'close':
            setIsSolving(false);
            break;
        }
      }
    );
    cancelRef.current = cancel;
  }, [demands, occNames, params, result]);

  const handleCancel = useCallback(() => {
    cancelRef.current?.();
    setIsSolving(false);
  }, []);

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <Sidebar
        params={params}
        onChange={updateParams}
        nCurves={nCurves}
        occNames={occNames}
        onNCurvesChange={setNCurves}
        onOccNamesChange={setOccNames}
      />

      {/* Main area */}
      <main className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">🕐 Simplex – Weekly Shift Optimiser</h1>
          <StatusDot />
        </div>

        {/* Demand input */}
        {!result && (
          <DemandInput
            nCurves={nCurves}
            occNames={occNames}
            demands={demands}
            onDemandsChange={setDemands}
          />
        )}

        {/* Solve */}
        {demands && (
          <SolveButton
            onSolve={handleSolve}
            onCancel={handleCancel}
            isSolving={isSolving}
            hasDemand={demands.length > 0}
          />
        )}

        {/* Logs */}
        {logs.length > 0 && (
          <pre className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-xs text-green-400 max-h-40 overflow-y-auto font-mono">
            {logs.join('\n')}
          </pre>
        )}

        {/* Results */}
        {result && (
          <ResultsPanel
            result={result}
            occNames={occNames}
            prevResult={prevResult}
          />
        )}
      </main>
    </div>
  );
}

// ── API health dot ──────────────────────────────────────────────────────────
function StatusDot() {
  const [ok, setOk] = useState<boolean | null>(null);
  useState(() => {
    healthCheck().then(setOk);
  });
  return (
    <span
      className={`inline-block w-3 h-3 rounded-full ${
        ok === null ? 'bg-gray-500' : ok ? 'bg-green-500' : 'bg-red-500'
      }`}
      title={ok ? 'API connected' : ok === false ? 'API offline' : 'Checking...'}
    />
  );
}