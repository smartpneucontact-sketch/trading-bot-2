# Multi-stage Docker build for the Next.js 16 frontend.
# Targets the `output: "standalone"` build mode from next.config.ts — produces
# a self-contained server bundle that doesn't need node_modules at runtime,
# dropping image size from ~1GB to ~150MB.
#
# Railway deployment:
#   1. Create a second service in the bot8 Railway project
#   2. Root Directory: `frontend`
#   3. Dockerfile Path: `Dockerfile`
#   4. Env vars:
#        NEXT_PUBLIC_API_BASE  = https://<api-service-public-url>
#        DASHBOARD_API_KEY     = same hex string as the API service
#
# Note: NEXT_PUBLIC_* vars are baked in at build time — if you change the API
# URL you must redeploy the frontend (not just restart it).

# --- deps ---------------------------------------------------------------------
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --prefer-offline --no-audit --progress=false

# --- builder ------------------------------------------------------------------
FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1

# NEXT_PUBLIC_* must be present at build time to be baked into the bundle.
# Railway injects env vars during build automatically.
ARG NEXT_PUBLIC_API_BASE
ENV NEXT_PUBLIC_API_BASE=${NEXT_PUBLIC_API_BASE}

RUN npm run build

# --- runner -------------------------------------------------------------------
FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Don't run as root.
RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 nextjs

# Copy only what's needed at runtime (standalone bundle + static assets).
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000

# Railway sets $PORT; fall back for local Docker runs.
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

CMD ["node", "server.js"]
