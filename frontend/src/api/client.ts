import axios from 'axios';
import type { SolveRequest, SolveResponse, SolverParams, SseEvent } from '../types';

const api = axios.create({
  baseURL: '/api',
  timeout: 300000, // 5 min for long solves
});

export async function postSolve(req: SolveRequest): Promise<SolveResponse> {
  const { data } = await api.post<SolveResponse>('/solve', req);
  return data;
}

export function streamSolve(req: SolveRequest, onEvent: (e: SseEvent) => void): () => void {
  const controller = new AbortController();
  let lastEvent = '';

  fetch('/api/solve/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal: controller.signal,
  }).then(async (res) => {
    if (!res.ok || !res.body) {
      onEvent({ type: 'error', message: `HTTP ${res.status}` });
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          lastEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          const payload = line.slice(6);
          if (!payload) continue;
          switch (lastEvent) {
            case 'log':
              onEvent({ type: 'log', message: payload });
              break;
            case 'result':
              try {
                const data = JSON.parse(payload) as SolveResponse;
                onEvent({ type: 'result', data });
              } catch {
                onEvent({ type: 'error', message: 'Failed to parse result' });
              }
              break;
            case 'error':
              onEvent({ type: 'error', message: payload });
              break;
            case 'close':
              onEvent({ type: 'close' });
              break;
            case 'heartbeat':
              onEvent({ type: 'heartbeat' });
              break;
          }
          lastEvent = '';
        }
      }
    }
  }).catch(err => {
    if (err.name !== 'AbortError') {
      onEvent({ type: 'error', message: String(err) });
    }
  });

  return () => controller.abort();
}

export async function fetchShiftCodes(params: SolverParams): Promise<string[]> {
  const { data } = await api.post<{ codes: string[]; count: number }>('/shift-codes', params);
  return data.codes;
}

export async function fetchSampleDemand(
  peakAgents: number, baseAgents: number, seed: number,
): Promise<number[]> {
  const { data } = await api.post<{ demand: number[] }>('/sample-demand', null, {
    params: { peak_agents: peakAgents, base_agents: baseAgents, seed },
  });
  return data.demand;
}

export async function healthCheck(): Promise<boolean> {
  try {
    await api.get('/health');
    return true;
  } catch {
    return false;
  }
}