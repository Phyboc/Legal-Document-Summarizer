import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The deployed site is the standalone page in `lexnarrate-web/` (see vercel.json,
// which copies lexnarrate-web/index.html to dist). Point Vite's root there so
// `npm run dev` serves exactly what ships, instead of the legacy React app in src/.
// https://vite.dev/config/
export default defineConfig({
  root: 'lexnarrate-web',
  plugins: [react()],
})
