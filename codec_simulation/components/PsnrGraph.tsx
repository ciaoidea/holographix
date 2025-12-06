import React, { useMemo } from 'react';

interface PsnrGraphProps {
  totalChunks: number;
  activeChunks: number;
}

export const PsnrGraph: React.FC<PsnrGraphProps> = ({ totalChunks, activeChunks }) => {
  // Generate data points for the curve based on the "Resilience curve" logic
  // Typically convex: starts low, ramps up as coherence converges
  const data = useMemo(() => {
    const points = [];
    for (let k = 1; k <= totalChunks; k++) {
      const x = k / totalChunks;
      // Simulation: Base ~24dB (coarse), Max ~39dB (lossless-ish)
      // Formula approximates the provided python plot: 24 + 15 * x^3.5
      const psnr = 23.5 + 15.5 * Math.pow(x, 3.2);
      points.push({ k, psnr });
    }
    return points;
  }, [totalChunks]);

  const maxPsnr = 42;
  const minPsnr = 20;
  const width = 100; // viewBox units
  const height = 50; // viewBox units
  const paddingX = 5;
  const paddingY = 5;

  // Scales
  const xScale = (k: number) => paddingX + ((k - 1) / (totalChunks - 1)) * (width - 2 * paddingX);
  const yScale = (psnr: number) => height - paddingY - ((psnr - minPsnr) / (maxPsnr - minPsnr)) * (height - 2 * paddingY);

  // Path generation
  const pathD = `M ${data.map(p => `${xScale(p.k)},${yScale(p.psnr)}`).join(' L ')}`;

  // Current status point calculation
  const currentPsnr = activeChunks > 0 
    ? (23.5 + 15.5 * Math.pow(activeChunks / totalChunks, 3.2)) 
    : 0;
    
  const cx = activeChunks > 0 ? xScale(activeChunks) : paddingX;
  const cy = activeChunks > 0 ? yScale(currentPsnr) : height - paddingY;

  return (
    <div className="bg-space-800 border border-slate-700 rounded-lg p-4 flex flex-col gap-3 shadow-lg relative overflow-hidden">
        {/* Background Grid Pattern */}
        <div className="absolute inset-0 opacity-5 pointer-events-none" 
            style={{ backgroundImage: 'radial-gradient(#14b8a6 1px, transparent 1px)', backgroundSize: '10px 10px' }} 
        />

       <div className="flex justify-between items-end z-10">
         <div>
            <h3 className="text-xs font-mono font-bold text-amber-400 uppercase tracking-widest flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
                Resilience Curve
            </h3>
            <p className="text-[9px] text-slate-500 font-mono mt-0.5">MEAN PSNR [dB] vs CHUNKS (k)</p>
         </div>
         <div className="text-right bg-slate-900/80 p-1.5 rounded border border-slate-700">
            <div className="text-[9px] text-slate-500 font-mono uppercase">Current Quality</div>
            <div className={`text-lg font-bold font-mono leading-none ${activeChunks === 0 ? 'text-slate-600' : 'text-holo-300'}`}>
                {activeChunks > 0 ? currentPsnr.toFixed(2) : '--'} <span className="text-xs text-slate-500">dB</span>
            </div>
         </div>
       </div>
       
       <div className="relative w-full aspect-[2.5/1] bg-space-900/50 rounded border border-slate-700/50 z-10">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible" preserveAspectRatio="none">
             {/* Grid Lines Y */}
             {Array.from({ length: 5 }).map((_, i) => {
                const val = minPsnr + i * (maxPsnr - minPsnr) / 4;
                const y = yScale(val);
                return (
                    <g key={i}>
                        <line x1={paddingX} y1={y} x2={width - paddingX} y2={y} stroke="#334155" strokeWidth="0.2" strokeDasharray="2,2" />
                        <text x={paddingX - 1} y={y + 1} fontSize="2.5" fill="#475569" textAnchor="end" fontFamily="monospace">{val.toFixed(0)}</text>
                    </g>
                );
             })}

             {/* Curve Shadow/Glow */}
             <path d={pathD} fill="none" stroke="#fbbf24" strokeWidth="1.5" strokeOpacity="0.3" filter="blur(1px)" />

             {/* The Golden Curve */}
             <path d={pathD} fill="none" stroke="#fbbf24" strokeWidth="0.8" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
             
             {/* Gradient Fill */}
             <defs>
               <linearGradient id="curveGradient" x1="0" x2="0" y1="0" y2="1">
                 <stop offset="0%" stopColor="#fbbf24" stopOpacity="0.15" />
                 <stop offset="100%" stopColor="#fbbf24" stopOpacity="0" />
               </linearGradient>
             </defs>
             <path d={`${pathD} L ${width-paddingX},${height-paddingY} L ${paddingX},${height-paddingY} Z`} fill="url(#curveGradient)" stroke="none" />

             {/* Current Position Marker */}
             {activeChunks > 0 && (
                <g className="transition-all duration-300 ease-out" style={{ transform: `translate(${cx}px, ${cy}px)` }}>
                    {/* Crosshair lines */}
                    <line x1={-100} y1={0} x2={100} y2={0} stroke="#14b8a6" strokeWidth="0.1" strokeDasharray="1,1" opacity="0.5" />
                    <line x1={0} y1={100} x2={0} y2={-100} stroke="#14b8a6" strokeWidth="0.1" strokeDasharray="1,1" opacity="0.5" />
                    
                    {/* Dot */}
                    <circle r="1.5" fill="#14b8a6" className="animate-ping opacity-75" vectorEffect="non-scaling-stroke" />
                    <circle r="1" fill="#0f172a" stroke="#14b8a6" strokeWidth="0.5" vectorEffect="non-scaling-stroke" />
                </g>
             )}
          </svg>
       </div>
    </div>
  );
};