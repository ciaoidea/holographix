import React, { useState, useEffect, useRef } from 'react';
import { Chunk, FileData, SignalStatus } from './types';
import { HoloVisualizer } from './components/HoloVisualizer';
import { ChunkGrid } from './components/ChunkGrid';
import { PsnrGraph } from './components/PsnrGraph';
import { Button } from './components/Button';
import { 
  Cpu, 
  Upload, 
  Radio, 
  Settings, 
  RefreshCw, 
  Zap, 
  Wifi,
  WifiOff,
  Github,
  Database,
  Grid,
  Activity,
  Waves
} from 'lucide-react';

export default function App() {
  const [file, setFile] = useState<FileData | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [isSimulating, setIsSimulating] = useState(false);
  const [interference, setInterference] = useState(0);
  const [isUnstable, setIsUnstable] = useState(false);
  
  // Configuration State
  const [totalChunks, setTotalChunks] = useState(64);
  const [chunkSize, setChunkSize] = useState(20);

  // Initialize simulation
  const startSimulation = (selectedFile: File) => {
    const url = URL.createObjectURL(selectedFile);
    setFile({
      name: selectedFile.name,
      size: selectedFile.size,
      type: selectedFile.type.startsWith('image') ? 'image' : 'binary',
      url
    });

    const newChunks: Chunk[] = Array.from({ length: totalChunks }, (_, i) => ({
      id: i,
      status: 'active',
      sizeKb: chunkSize
    }));
    setChunks(newChunks);
    setIsSimulating(true);
    setInterference(0);
    setIsUnstable(false);
  };

  const resetSimulation = () => {
    setFile(null);
    setChunks([]);
    setIsSimulating(false);
    setIsUnstable(false);
  };

  // Toggle individual chunk status
  const toggleChunk = (id: number) => {
    setChunks(prev => prev.map(c => {
      if (c.id === id) {
        return { ...c, status: c.status === 'active' ? 'lost' : 'active' };
      }
      return c;
    }));
  };

  // Mass interference simulation
  const applyInterference = (amount: number) => {
    setInterference(amount);
    setChunks(prev => prev.map(c => ({
      ...c,
      status: Math.random() > (amount / 100) ? 'active' : 'lost'
    })));
  };

  // Continuous Unstable Link Logic
  useEffect(() => {
    let intervalId: ReturnType<typeof setInterval>;

    if (isSimulating && isUnstable) {
      intervalId = setInterval(() => {
        // Randomly fluctuate interference between 10% and 90%
        const randomFlux = Math.floor(Math.random() * 80) + 10;
        
        // Apply logic directly to avoid dependency cycle on 'interference'
        setInterference(randomFlux);
        setChunks(prev => prev.map(c => ({
          ...c,
          status: Math.random() > (randomFlux / 100) ? 'active' : 'lost'
        })));
        
      }, 600); // Updates every 600ms
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isSimulating, isUnstable]);

  const getSignalStatus = (): SignalStatus => {
    if (!isSimulating) return SignalStatus.OFFLINE;
    const active = chunks.filter(c => c.status === 'active').length;
    const pct = active / totalChunks;
    if (pct > 0.8) return SignalStatus.EXCELLENT;
    if (pct > 0.4) return SignalStatus.DEGRADED;
    return SignalStatus.CRITICAL;
  };

  const activeChunksCount = chunks.filter(c => c.status === 'active').length;

  const statusColor = {
    [SignalStatus.EXCELLENT]: 'text-green-400',
    [SignalStatus.DEGRADED]: 'text-yellow-400',
    [SignalStatus.CRITICAL]: 'text-red-500',
    [SignalStatus.OFFLINE]: 'text-slate-500'
  };

  return (
    <div className="min-h-screen flex flex-col font-sans selection:bg-holo-500/30">
      {/* Header */}
      <header className="h-16 border-b border-slate-800 bg-space-900/80 backdrop-blur-md flex items-center justify-between px-6 sticky top-0 z-40">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-holo-600 rounded flex items-center justify-center shadow-[0_0_10px_rgba(13,148,136,0.5)]">
            <Cpu className="text-white w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
              HOLO.CODEC <span className="text-[10px] bg-slate-800 px-1.5 py-0.5 rounded text-holo-400 font-mono">v2.1</span>
            </h1>
            <p className="text-[10px] text-slate-400 font-mono tracking-wider uppercase">Holographic Media Protocol</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
           {isUnstable && isSimulating && (
             <div className="flex items-center gap-2 px-3 py-1 bg-red-900/20 border border-red-500/30 rounded text-red-400 animate-pulse">
               <Activity size={14} />
               <span className="text-[10px] font-mono font-bold uppercase">Unstable Connection</span>
             </div>
           )}
           <div className="text-right hidden sm:block">
             <div className="text-[10px] text-slate-500 uppercase font-mono">System Status</div>
             <div className={`text-xs font-bold font-mono flex items-center justify-end gap-2 ${statusColor[getSignalStatus()]}`}>
               {getSignalStatus()}
               {getSignalStatus() === SignalStatus.OFFLINE ? <WifiOff size={12}/> : <Wifi size={12}/>}
             </div>
           </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 p-4 md:p-6 overflow-y-auto relative">
        <div className="max-w-7xl mx-auto h-full">
          
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-full">
              
              {/* Left Column: Visualizer & Controls */}
              <div className="lg:col-span-8 flex flex-col gap-6">
                
                {/* Visualizer Area */}
                <div className="relative group">
                  <div className="absolute -inset-0.5 bg-gradient-to-r from-holo-600 to-blue-600 rounded-lg blur opacity-20 group-hover:opacity-40 transition duration-1000"></div>
                  <div className="relative bg-space-900 rounded-lg p-1">
                     <HoloVisualizer file={file} chunks={chunks} totalChunks={totalChunks} />
                  </div>
                </div>

                {/* Simulation Controls */}
                <div className="bg-space-800 border border-slate-700 rounded-lg p-6 shadow-lg">
                  <div className="flex items-center justify-between mb-6 border-b border-slate-700 pb-4">
                    <h3 className="text-sm font-bold text-slate-200 uppercase flex items-center gap-2">
                      <Settings className="w-4 h-4 text-holo-400" />
                      Codec Configuration & Simulation
                    </h3>
                  </div>
                  
                  {/* Codec Parameters */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                     <div className="space-y-3">
                        <label className="text-[10px] font-mono font-bold text-slate-400 uppercase flex items-center gap-2">
                           <Grid size={12} />
                           Total Holographic Blocks (N)
                        </label>
                        <div className="flex items-center gap-4">
                             <input
                                type="range"
                                min="16"
                                max="256"
                                step="16"
                                disabled={isSimulating}
                                value={totalChunks}
                                onChange={(e) => setTotalChunks(Number(e.target.value))}
                                className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-holo-500 disabled:opacity-50 disabled:cursor-not-allowed"
                             />
                             <div className="w-16 text-right font-mono text-holo-300 bg-space-900 px-2 py-1 rounded border border-slate-700 text-xs">
                                {totalChunks}
                             </div>
                        </div>
                        <p className="text-[10px] text-slate-600 leading-tight">
                           Defines the granularity of the golden permutation. Higher N = smoother degradation curve.
                        </p>
                     </div>

                     <div className="space-y-3">
                        <label className="text-[10px] font-mono font-bold text-slate-400 uppercase flex items-center gap-2">
                           <Database size={12} />
                           Packet Payload Size
                        </label>
                        <div className="flex items-center gap-4">
                             <input
                                type="range"
                                min="5"
                                max="100"
                                step="5"
                                disabled={isSimulating}
                                value={chunkSize}
                                onChange={(e) => setChunkSize(Number(e.target.value))}
                                className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-holo-500 disabled:opacity-50 disabled:cursor-not-allowed"
                             />
                             <div className="w-16 text-right font-mono text-holo-300 bg-space-900 px-2 py-1 rounded border border-slate-700 text-xs">
                                {chunkSize}KB
                             </div>
                        </div>
                         <p className="text-[10px] text-slate-600 leading-tight">
                           Size of each golden-ratio encoded fragment. Smaller = more robust, higher overhead.
                        </p>
                     </div>
                  </div>

                  {/* Divider */}
                  <div className="h-px bg-slate-700/50 w-full mb-8"></div>

                  {/* Interference Controls */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-end">
                    <div className="space-y-4">
                        <div className="flex justify-between text-xs font-mono text-slate-400">
                          <span className="flex items-center gap-2"><Radio size={12}/> CHANNEL INTERFERENCE (BER)</span>
                          <span className={interference > 50 ? 'text-red-400' : 'text-green-400'}>{interference}% ERR</span>
                        </div>
                        <div className="relative">
                            <input
                            type="range"
                            min="0"
                            max="100"
                            step="5"
                            disabled={!isSimulating || isUnstable}
                            value={interference}
                            onChange={(e) => applyInterference(parseInt(e.target.value))}
                            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-red-500 disabled:opacity-50 disabled:cursor-not-allowed z-10 relative"
                            />
                            {/* Ruler ticks */}
                            <div className="absolute top-3 left-0 right-0 flex justify-between px-1">
                                {Array.from({length: 11}).map((_, i) => (
                                    <div key={i} className="w-[1px] h-1 bg-slate-600"></div>
                                ))}
                            </div>
                        </div>
                    </div>

                    <div className="flex gap-4 justify-end">
                      {!isSimulating ? (
                        <div className="relative w-full">
                          <input 
                            type="file" 
                            accept="image/*" 
                            onChange={(e) => e.target.files?.[0] && startSimulation(e.target.files[0])}
                            className="absolute inset-0 opacity-0 cursor-pointer w-full h-full z-10"
                          />
                          <Button icon={<Upload size={16} />} className="w-full justify-center">
                            Initialize Simulation
                          </Button>
                        </div>
                      ) : (
                        <div className="flex gap-2 w-full">
                           <Button 
                             className="flex-1 justify-center" 
                             variant="secondary" 
                             onClick={() => {
                               setIsUnstable(false);
                               applyInterference(Math.floor(Math.random() * 80));
                             }} 
                             disabled={isUnstable}
                             icon={<Zap size={16}/>}
                           >
                              Random Jam
                           </Button>
                           
                           <Button 
                             className={`flex-1 justify-center ${isUnstable ? 'animate-pulse' : ''}`}
                             variant={isUnstable ? 'danger' : 'secondary'} 
                             onClick={() => setIsUnstable(!isUnstable)} 
                             icon={<Waves size={16}/>}
                           >
                              {isUnstable ? 'LINK UNSTABLE' : 'UNSTABLE LINK'}
                           </Button>

                           <Button 
                            className="flex-none justify-center px-3" 
                            variant="danger" 
                            onClick={resetSimulation} 
                            icon={<RefreshCw size={16}/>}
                           />
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Column: Analytics & Data */}
              <div className="lg:col-span-4 flex flex-col gap-4 min-h-[500px]">
                {/* Advanced Graph Component */}
                <PsnrGraph totalChunks={totalChunks} activeChunks={activeChunksCount} />

                {/* Packet Grid */}
                <div className="flex-1 flex flex-col min-h-[300px]">
                    <ChunkGrid chunks={chunks} onToggleChunk={toggleChunk} isSimulating={isSimulating} />
                </div>
                
                {/* Stats Panel */}
                <div className="bg-space-800 border border-slate-700 rounded-lg p-4 font-mono text-xs space-y-3 shadow-lg">
                   <div className="flex justify-between border-b border-slate-700 pb-2">
                     <span className="text-slate-500">CODEC_SCHEME</span>
                     <span className="text-holo-400 font-bold">GOLDEN_PERM_V2</span>
                   </div>
                   <div className="flex justify-between border-b border-slate-700 pb-2">
                     <span className="text-slate-500">SOURCE_FILE</span>
                     <span className="text-slate-300 truncate max-w-[150px]">{file ? file.name : 'NO_DATA'}</span>
                   </div>
                   <div className="flex justify-between border-b border-slate-700 pb-2">
                     <span className="text-slate-500">EFFECTIVE_BITRATE</span>
                     <span className="text-slate-300">{(activeChunksCount * chunkSize / 1024).toFixed(2)} MB/s</span>
                   </div>
                   <div className="flex justify-between">
                     <span className="text-slate-500">LATENCY_SIM</span>
                     <span className="text-green-400">~{20 + Math.floor(activeChunksCount * 0.5)}ms</span>
                   </div>
                </div>

                <div className="mt-2 text-center pb-4">
                  <a 
                    href="https://github.com/ciaoidea/Holo.Codec" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 text-slate-600 hover:text-holo-400 transition-colors text-xs uppercase tracking-widest font-bold group"
                  >
                    <Github size={14} className="group-hover:rotate-12 transition-transform" />
                    Holo.Codec Repo
                  </a>
                </div>
              </div>

            </div>
        </div>
      </main>
    </div>
  );
}