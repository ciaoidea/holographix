export type AppMode = 'SIMULATION' | 'CLI_GENERATOR';

export interface Chunk {
  id: number;
  status: 'active' | 'lost' | 'pending';
  sizeKb: number;
}

export interface FileData {
  name: string;
  size: number;
  type: 'image' | 'audio' | 'binary';
  url: string;
}

export enum SignalStatus {
  EXCELLENT = 'EXCELLENT',
  DEGRADED = 'DEGRADED',
  CRITICAL = 'CRITICAL',
  OFFLINE = 'OFFLINE'
}
