# Frontend image: build the Vite SPA, serve it with nginx (which also proxies /api -> backend).
#
# This is the PRODUCTION image (the `prod` compose profile). The `dev` profile ignores it and
# runs the Vite dev server directly for hot-module reload (see the frontend-dev service).

# --- build stage ---
FROM node:22-alpine AS build
WORKDIR /app

COPY package.json package-lock.json* ./
# Use a clean, reproducible install when a lockfile is present; fall back otherwise.
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY . .
RUN npm run build

# --- serve stage ---
FROM nginx:1.27-alpine AS serve
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
