import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 5173 is fixed (strictPort): it is in the backend CORS allowlist. Changing it
// means changing it there too. /api proxies to the backend (8010 by default).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: process.env.FV_API_URL || "http://127.0.0.1:8010",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
