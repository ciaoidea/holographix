import React from 'react';
import { Chunk, FileData } from '../types';

interface HoloVisualizerProps {
  file: FileData | null;
  chunks: Chunk[];
  totalChunks: number;
}

export const HoloVisualizer: React.FC<HoloVisualizerProps> = ({ file, chunks, totalChunks }) => {
  const activeChunks = chunks.filter(c => c.status === 'active').length;
  // Integrity is k/N (fraction of chunks present)
  const integrity = totalChunks > 0 ? activeChunks / totalChunks : 0;

  // Theoretical PSNR calculation for the overlay
  // Matches the graph: 23.5 (base) + 15.5 * x^3.2
  const currentPsnr = activeChunks > 0
    ? (23.5 + 15.5 * Math.pow(integrity, 3.2))
    : 0;

  // Simulation Logic:
  // The Holo.Codec v2 algorithm reconstructs the image as:
  // Recon = Coarse + (Original - Coarse) * Mask
  // Where Mask is 1 for pixels covered by received chunks, 0 otherwise.
  //
  // Visually, this means we have a base "Coarse" layer (always present)
  // and a "Detail" layer (Original - Coarse) that is added sparsely.
  //
  // To simulate this in CSS without per-pixel masking:
  // 1. Base Layer: The Coarse image. Simulated by blurring the original.
  //    Coarse size is ~64px. Display is ~800px. Scale factor ~12.5.
  //    Blur radius ~6px is physically accurate for this upscale.
  // 2. Detail Layer: The Sharp image.
  //    We fade this in using `opacity = integrity`.
  //    This mathematically represents the average energy of the reconstructed signal.
  // 3. Noise Layer:
  //    We add noise to simulate the high-frequency variance of the "missing pixels" (Golden Permutation).
  //    Variance is highest at 50% integrity.
  //    Opacity = 4 * integrity * (1 - integrity) * Scale.

  const baseBlurPx = 6;
  const noiseOpacity = Math.max(0, 4 * integrity * (1 - integrity) * 0.35); // Peak opacity at 50% integrity

  if (!file) {
    return (
      <div className="w-full h-full min-h-[400px] flex items-center justify-center border border-slate-700 bg-space-800/50 rounded-lg relative overflow-hidden group">
        <div className="absolute inset-0 grid grid-cols-[repeat(20,minmax(0,1fr))] grid-rows-[repeat(20,minmax(0,1fr))] opacity-10 pointer-events-none">
          {Array.from({ length: 400 }).map((_, i) => (
            <div key={i} className="border border-holo-500/20" />
          ))}
        </div>
        <div className="text-center p-8 z-10">
          <div className="w-16 h-16 border-2 border-slate-600 border-dashed rounded-full mx-auto mb-4 animate-spin-slow flex items-center justify-center group-hover:border-holo-500/50 transition-colors">
            <div className="w-2 h-2 bg-slate-600 rounded-full group-hover:bg-holo-500 transition-colors" />
          </div>
          <p className="text-slate-400 font-mono text-sm uppercase tracking-widest">No Signal Source</p>
          <p className="text-slate-600 text-xs mt-2">Upload media to initialize holographic simulation</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full aspect-video bg-black overflow-hidden border border-holo-900/50 rounded-lg shadow-2xl group">

      {/* 1. The Coarse Layer (Background) - Always visible, constant blur (Thumbnail) */}
      <div
        className="absolute inset-0 bg-center bg-cover transition-transform duration-700 ease-out"
        style={{
          backgroundImage: `url(${file.url})`,
          filter: `blur(${baseBlurPx}px) brightness(0.9) contrast(0.9)`,
          transform: 'scale(1.02)' // Scale slightly to hide blur edges
        }}
      />

      {/* 2. The Detail Layer (Foreground) - Fades in linearly with integrity (Signal Energy) */}
      <div
        className="absolute inset-0 bg-center bg-cover transition-opacity duration-200 ease-linear"
        style={{
          backgroundImage: `url(${file.url})`,
          opacity: integrity,
          // Apply a tiny blur only at extremely low integrity to smooth the initial transition
          filter: integrity < 0.1 ? `blur(${baseBlurPx * (1 - integrity * 10)}px)` : 'none'
        }}
      />

      {/* 3. The Digital Noise/Grain Layer - Simulates the 'Golden Permutation' pixel scattering */}
      <div
        className="absolute inset-0 pointer-events-none mix-blend-overlay"
        style={{
            opacity: noiseOpacity,
            backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='1.5' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`,
            backgroundSize: '128px 128px'
        }}
      />

      {/* 4. CRT/Scanline Overlay (Cosmetic Interface Effect) */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(18,255,247,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(18,255,247,0.02)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none z-10" />
      <div className="absolute inset-0 bg-gradient-to-t from-black/40 via-transparent to-black/40 pointer-events-none z-10" />

      {/* Status Overlay */}
      <div className="absolute top-4 left-4 font-mono text-xs z-20">
         <div className="bg-space-900/90 backdrop-blur-sm border border-holo-500/30 p-2 text-holo-400 rounded-sm shadow-lg min-w-[160px]">
           <div className="flex items-center gap-2 mb-2 border-b border-holo-500/20 pb-1">
             <span className={`w-2 h-2 rounded-full ${integrity > 0.8 ? 'bg-green-500' : integrity > 0.4 ? 'bg-amber-500' : 'bg-red-500'} animate-pulse`} />
             <span className="font-bold tracking-wider">HOLO.VISUALIZER</span>
           </div>
           <div className="space-y-1 text-[10px] text-slate-300">
             <div className="flex justify-between gap-4"><span>INTEGRITY</span> <span className="text-white font-bold">{(integrity * 100).toFixed(1)}%</span></div>
             <div className="flex justify-between gap-4"><span>CHUNKS</span> <span className="font-mono">{activeChunks}/{totalChunks}</span></div>
             <div className="flex justify-between gap-4"><span>ALGO</span> <span className="text-holo-300">GOLDEN_V2</span></div>
           </div>
           <div className="mt-2 pt-1 border-t border-white/10 text-[10px] text-slate-400 flex justify-between items-center">
             <span>EST. PSNR</span>
             <span className="text-white font-bold bg-holo-900/50 px-1.5 py-0.5 rounded border border-holo-500/20">{currentPsnr.toFixed(1)} dB</span>
           </div>
         </div>
      </div>

      {integrity < 0.05 && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/60 z-30 backdrop-blur-[2px]">
           <div className="bg-red-900/20 border border-red-500/50 p-6 rounded text-center shadow-2xl">
             <p className="text-red-500 font-mono font-bold animate-pulse text-xl tracking-widest">SIGNAL CRITICAL</p>
             <p className="text-red-400/70 text-xs mt-2 uppercase">Insufficient residual data for reconstruction</p>
           </div>
        </div>
      )}
    </div>
  );
};
