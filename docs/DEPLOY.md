# Deployment Guide — Railway

Deploys both services (FastAPI backend + Next.js frontend) to a single Railway project.

## Architecture

```
Railway project: bot8
│
├─ Service: bot8-api           Python + DuckDB + cron jobs
│   • Root directory: /
│   • Dockerfile: Dockerfile.api
│   • Volume: /app/data   (persists DB across redeploys)
│   • Public URL: bot8-api.up.railway.app
│
└─ Service: bot8-frontend      Next.js 16 standalone
    • Root directory: /frontend
    • Dockerfile: Dockerfile
    • Public URL: bot8-frontend.up.railway.app
```

## One-time setup

### 1. Push the repo to GitHub

Railway deploys from GitHub. If this isn't a git repo yet:

```bash
cd "/Users/arsenkhanguieldyan/Documents/Traiding bot 8"
git init
# add a .gitignore for frontend if needed — next already created one
git add -A
git commit -m "initial bot8 commit"
gh repo create bot8 --private --source=. --push
```

### 2. Generate a shared dashboard key

```bash
openssl rand -hex 32
```

Copy the output — you'll paste it into both services' env vars.

## Railway service setup

### Step 1 — Create the project and API service

1. https://railway.com/new → "Deploy from GitHub repo" → pick `bot8`.
2. Railway auto-detects `railway.toml` at repo root → builds from `Dockerfile.api`.
3. Rename the service to `bot8-api` (Settings → Service Name).
4. **Add a volume**: Settings → Volumes → New Volume, mount path `/app/data`. This persists the DuckDB file across redeploys.
5. **Generate a public domain**: Settings → Networking → Generate Domain.  Copy the resulting URL (e.g. `bot8-api-production.up.railway.app`).
6. **Environment variables**:

   | Key | Value |
   | --- | --- |
   | `ALPACA_API_KEY` | your paper key |
   | `ALPACA_SECRET_KEY` | your paper secret |
   | `ANTHROPIC_API_KEY` | (optional — only if re-running Claude scorer) |
   | `DASHBOARD_API_KEY` | the generated hex string |
   | `FRONTEND_ORIGINS` | `https://<frontend-service-url>` *(fill after step 2)* |
   | `DATA_DIR` | `/app/data` |

7. Click Deploy.

### Step 2 — Create the frontend service in the same project

1. In the same Railway project → "+ New" → "GitHub Repo" → pick `bot8` again (yes, the same repo).
2. Railway creates a new service.
3. Settings → Service Name: `bot8-frontend`.
4. Settings → **Root Directory**: `frontend`.
5. Settings → Build: should auto-detect `frontend/railway.toml` → `frontend/Dockerfile`.
6. **Generate a public domain** (Settings → Networking → Generate Domain).
7. **Environment variables**:

   | Key | Value |
   | --- | --- |
   | `NEXT_PUBLIC_API_BASE` | `https://<bot8-api-url>` from step 1.5 |
   | `DASHBOARD_API_KEY` | same hex string as step 1.6 |

8. Click Deploy.

### Step 3 — Wire CORS

Back in `bot8-api` settings → Variables:

- Set `FRONTEND_ORIGINS` = `https://<bot8-frontend-url>` (from step 2.6).
- Railway auto-redeploys the API service.

### Step 4 — Upload the DuckDB file

The local DuckDB has all features, labels, and predictions already computed. Two options to get it onto the Railway volume:

**Option A — Re-run the pipeline remotely** (clean but slow, ~2 hours):

```bash
railway run --service bot8-api bash
# Inside the container:
bot8 data init
bot8 data universe
bot8 data bars --since 2019-01-01
bot8 data macro --since 2019-01-01
bot8 data fnspid --since 2020-01-01 --file 'Stock_news/nasdaq_exteral_data.csv'
bot8 features news --backfill
bot8 features news-daily
bot8 features quant --since 2019-01-01
bot8 train quant --fast
bot8 backtest --compare --slippage-bps 1
```

**Option B — Direct file copy** (faster, ~5 min):

```bash
# From your laptop, compress the DuckDB file:
cd "/Users/arsenkhanguieldyan/Documents/Traiding bot 8"
gzip -k data/db/bot8.duckdb    # makes bot8.duckdb.gz

# Railway CLI: upload to the volume via a one-off shell
railway run --service bot8-api sh -c "mkdir -p /app/data/db"

# Use rsync or scp-equivalent. Easiest: stream it via railway run + cat:
cat data/db/bot8.duckdb.gz | railway run --service bot8-api sh -c "gunzip > /app/data/db/bot8.duckdb"
```

(The cat-pipe approach depends on Railway's exec mode allowing stdin. If it doesn't, use Option A.)

## Verify

```bash
# API health
curl https://bot8-api.up.railway.app/health
# → {"status":"ok","service":"bot8-api","version":"0.1.0"}

# API with auth
curl -H "X-API-Key: <your-key>" https://bot8-api.up.railway.app/api/backtest/compare

# Frontend
open https://bot8-frontend.up.railway.app
# Should render the dashboard with live data.
```

## Cost

- **API service** (always-on, small): ~$5/mo
- **Frontend service** (always-on, very small): ~$3/mo
- **Volume** (1GB DuckDB): free tier covers it
- **Total**: ~$5–10/mo

## Ongoing dev flow

- `git push` → Railway auto-deploys both services on the changed paths
- Env var changes in Railway dashboard → automatic redeploy of that service
- View logs: Railway UI → service → Deployments → Logs
- SSH into container: `railway run --service bot8-api bash`

## Troubleshooting

**API returns 401 after deploy.** `DASHBOARD_API_KEY` must match **exactly** on both services. Whitespace matters.

**Frontend shows "connection refused".** `NEXT_PUBLIC_API_BASE` is baked into the JS bundle at build time — if you changed it, **redeploy** the frontend, don't just restart.

**CORS error in browser.** `FRONTEND_ORIGINS` on the API must include the exact frontend URL, including `https://`.

**Frontend build OOM.** Next.js builds can spike memory. Railway free tier has 512MB which is tight. If you hit OOM, upgrade the frontend service to the $5 plan or reduce `npm run build` memory with `NODE_OPTIONS=--max-old-space-size=1024`.

**DuckDB file missing after redeploy.** Make sure the volume is mounted at `/app/data` and `DATA_DIR=/app/data` is set.
