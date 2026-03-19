import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@mediapipe/pose': fileURLToPath(new URL('./src/shims/mediapipe-pose.ts', import.meta.url)),
    },
  },
})
