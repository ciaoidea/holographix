import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './', // CRITICO: Assicura che i percorsi degli asset siano relativi per Electron
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  }
});