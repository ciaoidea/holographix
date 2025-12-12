import React from 'react';
import { Chunk } from '../types';
import { Activity, XCircle, CheckCircle2 } from 'lucide-react';

interface ChunkGridProps {
  chunks: Chunk[];
  onToggleChunk: (id: number) => void;
  isSimulating: boolean;
}

export const ChunkGrid: React.FC<ChunkGridProps> = ({ chunks, onToggleChunk, isSimulating }) => {
  const activeCount = chunks.filter(c => c.status === 'active').length;
  const lostCount = chunks.filter(c => c.status === 'lost').length;

  return (
    <div className="bg-space-800 border border-slate-700 rounded-lg p-4 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4 border-b border-slate-700 pb-2">
        <h3 className="text-xs font-mono font-bold text-holo-400 uppercase tracking-widest flex items-center gap-2">
          <Activity className="w-4 h-4" />
          Packet Stream
        </h3>
        <div className="flex gap-3 text-[10px] font-mono">
          <span className="text-green-400 flex items-center gap-1">
             <span className="w-2 h-2 bg-green-500/20 border border-green-500 rounded-sm"></span>
             RX: {activeCount}
          </span>
          <span className="text-red-400 flex items-center gap-1">
             <span className="w-2 h-2 bg-red-500/20 border border-red-500 rounded-sm"></span>
             LOST: {lostCount}
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto pr-1">
        <div className="grid grid-cols-8 sm:grid-cols-10 md:grid-cols-12 lg:grid-cols-8 xl:grid-cols-10 gap-1.5">
          {chunks.map((chunk) => (
            <button
              key={chunk.id}
              onClick={() => onToggleChunk(chunk.id)}
              disabled={!isSimulating}
              className={`
                aspect-square rounded-[1px] text-[8px] font-mono flex items-center justify-center transition-all duration-200
                hover:scale-110 focus:outline-none relative group
                ${chunk.status === 'active' 
                  ? 'bg-holo-500/20 border border-holo-500 text-holo-300 shadow-[0_0_5px_rgba(20,184,166,0.2)]' 
                  : chunk.status === 'lost'
                  ? 'bg-red-900/20 border border-red-900 text-red-800 opacity-50'
                  : 'bg-slate-800 border border-slate-700 text-slate-600'
                }
              `}
              title={`Chunk ID: ${chunk.id} [${chunk.status.toUpperCase()}]`}
            >
              {chunk.id}
              {chunk.status === 'active' && (
                <div className="absolute inset-0 bg-holo-400/10 opacity-0 group-hover:opacity-100 transition-opacity" />
              )}
            </button>
          ))}
        </div>
      </div>
      
      {!isSimulating && (
        <div className="mt-4 text-center p-2 bg-slate-900/50 border border-slate-700/50 rounded text-xs text-slate-500 font-mono">
          AWAITING TRANSMISSION
        </div>
      )}
    </div>
  );
};