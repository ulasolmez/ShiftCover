import { Loader2 } from 'lucide-react';

interface Props {
  onSolve: () => void;
  onCancel: () => void;
  isSolving: boolean;
  hasDemand: boolean;
}

export default function SolveButton({ onSolve, onCancel, isSolving, hasDemand }: Props) {
  if (isSolving) {
    return (
      <button
        onClick={onCancel}
        className="px-8 py-3 bg-red-600 hover:bg-red-700 text-white font-bold rounded-lg flex items-center gap-2"
      >
        <Loader2 className="animate-spin w-5 h-5" />
        Cancel
      </button>
    );
  }

  return (
    <button
      onClick={onSolve}
      disabled={!hasDemand}
      className="sticky top-0 z-10 px-8 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-400 text-white font-bold rounded-lg"
    >
      ▶ Run Optimiser
    </button>
  );
}