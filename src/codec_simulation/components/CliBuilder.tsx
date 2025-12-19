import React, { useState, useRef, useEffect } from 'react';
import { Button } from './Button';
import { Copy, Terminal, Layers, FileInput, FolderOutput, FolderOpen, FileSearch, Play, X, Loader2, FolderCog, HardDrive, Download, FileCode } from 'lucide-react';

export const CliBuilder: React.FC = () => {
  const [mode, setMode] = useState<'encode' | 'decode' | 'stack'>('encode');
  
  // New: Base Path configuration
  const [basePath, setBasePath] = useState('');
  
  const [inputPath, setInputPath] = useState('image.png');
  const [chunkSize, setChunkSize] = useState(20);
  const [stackFiles, setStackFiles] = useState('frame1.png frame2.png frame3.png');
  const [holoPath, setHoloPath] = useState('image.png.holo');
  
  // Console Simulation State
  const [isConsoleOpen, setConsoleOpen] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const consoleEndRef = useRef<HTMLDivElement>(null);

  // Hidden input refs
  const sourceInputRef = useRef<HTMLInputElement>(null);
  const holoInputRef = useRef<HTMLInputElement>(null);
  const stackInputRef = useRef<HTMLInputElement>(null);

  // Helper to format paths
  const getFullPath = (file: string) => {
    if (!basePath) return `"${file}"`;
    // Remove trailing slash from base path if present
    const cleanBase = basePath.replace(/[\\/]$/, '');
    // Handle Windows style backslashes just in case user types them
    const separator = cleanBase.includes('\\') ? '\\' : '/';
    return `"${cleanBase}${separator}${file}"`;
  };

  const getCommand = () => {
    switch (mode) {
      case 'encode':
        return `python3 holo.py ${getFullPath(inputPath)} ${chunkSize}`;
      case 'decode':
        return `python3 holo.py ${getFullPath(holoPath)}`;
      case 'stack':
        // Handle multiple files for stack
        const files = stackFiles.split(' ').map(f => {
            const cleanName = f.replace(/"/g, '');
            return getFullPath(cleanName);
        }).join(' ');
        return `python3 holo.py --stack ${chunkSize} ${files}`;
      default:
        return '';
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(getCommand());
  };

  const downloadScript = (type: 'bat' | 'sh') => {
    const cmd = getCommand();
    let content = '';
    let mimeType = '';
    let fileName = '';

    if (type === 'bat') {
        // Windows Batch File
        content = `@echo off\nREM Holo.Codec Auto-Generated Script\n\necho [HOLO] Initializing environment...\n\nREM Check if Python is installed\npython --version >nul 2>&1\nIF %ERRORLEVEL% NEQ 0 (\n  echo [ERROR] Python is not installed or not in PATH.\n  pause\n  exit /b\n)\n\necho [HOLO] Executing: ${cmd}\n${cmd.replace('python3', 'python')}\n\nIF %ERRORLEVEL% NEQ 0 (\n  echo [ERROR] Execution failed.\n) ELSE (\n  echo [SUCCESS] Operation completed successfully.\n)\n\npause`;
        mimeType = 'application/x-bat';
        fileName = 'run_holo.bat';
    } else {
        // Unix Shell Script
        content = `#!/bin/bash\n# Holo.Codec Auto-Generated Script\n\necho "[HOLO] Initializing environment..."\n\necho "[HOLO] Executing: ${cmd}"\n${cmd}\n\nif [ $? -eq 0 ]; then\n  echo "[SUCCESS] Operation completed successfully."\nelse\n  echo "[ERROR] Execution failed."\nfi\n`;
        mimeType = 'application/x-sh';
        fileName = 'run_holo.sh';
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Helper to handle file selection
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>, target: 'source' | 'holo' | 'stack') => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    // Reset base path when selecting new files as we assume relative context
    setBasePath('');

    if (target === 'source') {
        setInputPath(files[0].name);
    } else if (target === 'holo') {
        // Try to be smart about folder selection
        const path = files[0].webkitRelativePath || files[0].name;
        if (path.includes('.holo')) {
            const parts = path.split('/');
            const holoIndex = parts.findIndex(p => p.endsWith('.holo'));
            if (holoIndex !== -1) {
                setHoloPath(parts[holoIndex]); 
                return;
            }
        }
        setHoloPath(files[0].name);
    } else if (target === 'stack') {
        const fileList = Array.from(files).map(f => `${f.name}`).join(' ');
        setStackFiles(fileList);
    }
  };

  // Simulation Logic
  const addLog = (text: string) => {
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString('en-US', {hour12: false, fractionalSecondDigits: 2} as any)}] ${text}`]);
  };

  useEffect(() => {
    if (consoleEndRef.current) {
        consoleEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, isConsoleOpen]);

  const runSimulation = () => {
    setConsoleOpen(true);
    setIsRunning(true);
    setLogs([]);
    addLog("Initializing Interface...");
    
    let steps: {msg: string, delay: number}[] = [];

    if (mode === 'encode') {
        steps = [
            { msg: `Analyzing source: ${inputPath}...`, delay: 500 },
            { msg: "Generating command parameters...", delay: 1000 },
            { msg: "Optimizing for local execution...", delay: 1500 },
            { msg: "READY TO EXECUTE.", delay: 2000 },
        ];
    } else {
        steps = [
            { msg: "Preparing decode parameters...", delay: 500 },
            { msg: "READY TO EXECUTE.", delay: 1500 },
        ];
    }

    let timeouts: ReturnType<typeof setTimeout>[] = [];
    
    steps.forEach(({msg, delay}) => {
        const t = setTimeout(() => addLog(msg), delay);
        timeouts.push(t);
    });

    const finishT = setTimeout(() => {
        setIsRunning(false);
        addLog("----------------------------------------");
        addLog("NOTICE: Browsers cannot write to disk directly.");
        addLog("1. Download the runner script below.");
        addLog(`2. Place it in the folder with '${inputPath}'.`);
        addLog("3. Double-click to create folders for real.");
    }, steps[steps.length - 1].delay + 500);
    timeouts.push(finishT);
  };

  return (
    <div className="flex flex-col gap-6 w-full max-w-4xl mx-auto">
        <div className="bg-space-800 border border-slate-700 rounded-lg p-6 font-mono text-sm shadow-2xl relative overflow-hidden">
        
        {/* Header */}
        <div className="flex items-center gap-3 mb-6 border-b border-slate-700 pb-4 relative z-10">
            <Terminal className="w-5 h-5 text-holo-400" />
            <h2 className="text-lg font-bold text-slate-200">CLI Command Generator</h2>
        </div>

        {/* Global Settings (Path) */}
        <div className="mb-8 bg-slate-900/50 p-4 rounded border border-slate-700 relative z-10">
            <label className="text-xs uppercase text-holo-500 font-bold flex items-center gap-2 mb-2">
                <FolderCog size={14} /> Working Directory
            </label>
            <div className="relative">
                <span className="absolute left-3 top-2.5 text-slate-500"><HardDrive size={14}/></span>
                <input 
                    type="text" 
                    value={basePath} 
                    onChange={(e) => setBasePath(e.target.value)}
                    className="w-full bg-space-900 border border-slate-600 rounded pl-9 pr-3 py-2 text-slate-200 focus:border-holo-500 focus:outline-none placeholder-slate-600"
                    placeholder="(Optional) Leave empty to use the current folder where the script is located"
                />
            </div>
            <p className="text-[10px] text-slate-500 mt-2 flex items-start gap-1">
                <span className="text-amber-500 font-bold">INFO:</span>
                Browsers cannot detect your full local path. For best results, place the "run_holo" script 
                in the same folder as your images and leave this field empty.
            </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative z-10">
            {/* Left Column: Mode Selection */}
            <div className="space-y-2">
            <label className="text-xs uppercase text-slate-500 font-bold">Operation Mode</label>
            <div className="flex flex-col gap-2">
                <button 
                onClick={() => setMode('encode')}
                className={`text-left px-3 py-2 border rounded transition-colors ${mode === 'encode' ? 'bg-holo-900/30 border-holo-500 text-holo-300' : 'border-slate-700 text-slate-400 hover:border-slate-500'}`}
                >
                ENCODE (File → Holo)
                </button>
                <button 
                onClick={() => setMode('decode')}
                className={`text-left px-3 py-2 border rounded transition-colors ${mode === 'decode' ? 'bg-holo-900/30 border-holo-500 text-holo-300' : 'border-slate-700 text-slate-400 hover:border-slate-500'}`}
                >
                DECODE (Holo → File)
                </button>
                <button 
                onClick={() => setMode('stack')}
                className={`text-left px-3 py-2 border rounded transition-colors ${mode === 'stack' ? 'bg-holo-900/30 border-holo-500 text-holo-300' : 'border-slate-700 text-slate-400 hover:border-slate-500'}`}
                >
                STACK & ENCODE
                </button>
            </div>
            </div>

            {/* Right Column: Dynamic Inputs */}
            <div className="md:col-span-2 space-y-6">
            
            {/* ENCODE MODE */}
            {mode === 'encode' && (
                <>
                <div className="space-y-2">
                    <label className="text-xs uppercase text-slate-500 font-bold flex items-center gap-2">
                    <FileInput size={14} /> Source File Name
                    </label>
                    <div className="flex gap-2">
                        <input 
                        type="text" 
                        value={inputPath} 
                        onChange={(e) => setInputPath(e.target.value)}
                        className="flex-1 bg-space-900 border border-slate-600 rounded px-3 py-2 text-slate-200 focus:border-holo-500 focus:outline-none placeholder-slate-600"
                        placeholder="image.png"
                        />
                        <input 
                            type="file" 
                            ref={sourceInputRef} 
                            className="hidden" 
                            onChange={(e) => handleFileSelect(e, 'source')}
                        />
                        <Button 
                            variant="secondary" 
                            onClick={() => sourceInputRef.current?.click()}
                            title="Browse File"
                        >
                            <FileSearch size={16} />
                        </Button>
                    </div>
                </div>
                
                <div className="bg-space-900/50 p-3 rounded border border-slate-700/50 flex items-start gap-3 opacity-80">
                    <FolderOutput className="text-holo-500 mt-0.5 shrink-0" size={16} />
                    <div className="overflow-hidden">
                        <span className="text-[10px] uppercase text-holo-500 font-bold block mb-1">Expected Output Folder</span>
                        <span className="text-slate-300 break-all text-xs font-mono">{getFullPath(inputPath + '.holo')}</span>
                    </div>
                </div>

                <div className="space-y-2">
                    <label className="text-xs uppercase text-slate-500 font-bold block">Target Chunk Size (KB)</label>
                    <div className="flex items-center gap-4">
                        <input 
                        type="range" 
                        min="1" 
                        max="100" 
                        value={chunkSize} 
                        onChange={(e) => setChunkSize(parseInt(e.target.value))}
                        className="flex-1 accent-holo-500"
                        />
                        <span className="w-12 text-right text-holo-400">{chunkSize}KB</span>
                    </div>
                </div>
                </>
            )}

            {/* DECODE MODE */}
            {mode === 'decode' && (
                <div className="space-y-2">
                    <label className="text-xs uppercase text-slate-500 font-bold flex items-center gap-2">
                    <FolderOpen size={14} /> Holo Directory Name
                    </label>
                    <div className="flex gap-2">
                        <input 
                        type="text" 
                        value={holoPath} 
                        onChange={(e) => setHoloPath(e.target.value)}
                        className="flex-1 bg-space-900 border border-slate-600 rounded px-3 py-2 text-slate-200 focus:border-holo-500 focus:outline-none"
                        placeholder="image.png.holo"
                        />
                        <input 
                            type="file"
                            // @ts-ignore
                            webkitdirectory="" 
                            directory=""
                            ref={holoInputRef} 
                            className="hidden" 
                            onChange={(e) => handleFileSelect(e, 'holo')}
                        />
                        <Button 
                            variant="secondary" 
                            onClick={() => holoInputRef.current?.click()}
                            title="Browse Folder"
                        >
                            <FolderOpen size={16} />
                        </Button>
                    </div>
                    <p className="text-[10px] text-slate-500 mt-1">
                        Select the .holo folder that needs to be decoded.
                    </p>
                </div>
            )}

            {/* STACK MODE */}
            {mode === 'stack' && (
                <>
                <div className="space-y-2">
                    <label className="text-xs uppercase text-slate-500 font-bold block">Target Chunk Size (KB)</label>
                    <div className="flex items-center gap-4">
                        <input 
                        type="range" 
                        min="1" 
                        max="100" 
                        value={chunkSize} 
                        onChange={(e) => setChunkSize(parseInt(e.target.value))}
                        className="flex-1 accent-holo-500"
                        />
                        <span className="w-12 text-right text-holo-400">{chunkSize}KB</span>
                    </div>
                </div>
                <div className="space-y-2">
                    <label className="text-xs uppercase text-slate-500 font-bold block">Input Files</label>
                    <div className="flex gap-2 items-start">
                        <textarea 
                        value={stackFiles} 
                        onChange={(e) => setStackFiles(e.target.value)}
                        className="flex-1 h-20 bg-space-900 border border-slate-600 rounded px-3 py-2 text-slate-200 focus:border-holo-500 focus:outline-none resize-none font-mono text-xs"
                        placeholder="frame1.png frame2.png ..."
                        />
                        <input 
                            type="file" 
                            multiple
                            ref={stackInputRef} 
                            className="hidden" 
                            onChange={(e) => handleFileSelect(e, 'stack')}
                        />
                        <Button 
                            variant="secondary" 
                            className="h-20"
                            onClick={() => stackInputRef.current?.click()}
                            title="Select Multiple Files"
                        >
                            <Layers size={16} />
                        </Button>
                    </div>
                </div>
                </>
            )}
            </div>
        </div>

        {/* Action Buttons */}
        <div className="mt-8 bg-black/50 border border-slate-800 rounded-lg p-4 relative group">
             <div className="flex flex-col gap-4">
                <div className="font-mono text-green-400 break-all text-xs md:text-sm p-3 bg-black/80 rounded border border-slate-800">
                    <span className="text-slate-500 select-none">$ </span>
                    {getCommand()}
                </div>
                
                <div className="flex flex-wrap items-center gap-3">
                    <Button 
                        variant="primary"
                        onClick={runSimulation}
                        disabled={isRunning}
                        className="text-xs shadow-[0_0_15px_rgba(20,184,166,0.3)] flex-1 sm:flex-none justify-center"
                        icon={isRunning ? <Loader2 className="animate-spin" size={14} /> : <Play size={14} fill="currentColor" />}
                    >
                        {isRunning ? 'Running...' : 'Simulate'}
                    </Button>

                    <div className="h-8 w-px bg-slate-700 mx-2 hidden sm:block"></div>

                    <Button 
                        variant="secondary"
                        onClick={() => downloadScript('bat')}
                        className="text-xs flex-1 sm:flex-none justify-center border-amber-500/50 text-amber-500 hover:text-amber-300 hover:border-amber-400"
                        icon={<FileCode size={14} />}
                        title="Download Windows Batch File"
                    >
                        Download .bat (Windows)
                    </Button>
                     <Button 
                        variant="secondary"
                        onClick={() => downloadScript('sh')}
                        className="text-xs flex-1 sm:flex-none justify-center"
                        icon={<FileCode size={14} />}
                        title="Download Unix/Linux Shell Script"
                    >
                        Download .sh (Mac/Linux)
                    </Button>

                    <Button 
                        variant="ghost"
                        onClick={copyToClipboard}
                        className="text-xs ml-auto"
                        icon={<Copy size={14} />}
                    >
                        Copy
                    </Button>
                </div>
            </div>
        </div>
        </div>

        {/* Console / Terminal Output */}
        {isConsoleOpen && (
            <div className="bg-black border border-slate-700 rounded-lg p-4 font-mono text-xs shadow-2xl animate-in slide-in-from-bottom-4 fade-in duration-300">
                <div className="flex justify-between items-center border-b border-slate-800 pb-2 mb-2">
                    <span className="text-slate-500 uppercase font-bold tracking-wider text-[10px]">Holo.Codec Terminal Interface [SIMULATION]</span>
                    <button onClick={() => setConsoleOpen(false)} className="text-slate-500 hover:text-white"><X size={14}/></button>
                </div>
                <div className="h-48 overflow-y-auto space-y-1 scroll-smooth">
                    {logs.map((log, i) => (
                        <div key={i} className="text-green-500/90 border-l-2 border-green-500/20 pl-2 break-all">
                            {log}
                        </div>
                    ))}
                     {isRunning && (
                        <div className="text-green-500/50 pl-2 animate-pulse">_</div>
                    )}
                    <div ref={consoleEndRef} />
                </div>
            </div>
        )}
    </div>
  );
};