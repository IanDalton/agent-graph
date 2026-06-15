import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Dev: proxy the API so the browser sees a same-origin /api and SSE streams cleanly.
// API_PROXY_TARGET lets the containerized dev server point at the `backend` service
// (http://backend:8000) instead of the host default.
const apiProxyTarget = process.env.API_PROXY_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    host: true, // listen on 0.0.0.0 so the port is reachable from outside the container
    port: 5173,
    proxy: {
      "/api": {
        target: apiProxyTarget,
        changeOrigin: true,
      },
    },
  },
});
