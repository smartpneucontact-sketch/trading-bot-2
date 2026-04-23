// Typed API client for the bot8 FastAPI backend.
// Single source of truth for every request — mirrors the Pydantic models on
// the server side. Update here when the backend schema changes.

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

// Server-side only — never exposed to the browser. Used by server components
// when they fetch from the API during SSR.
const API_KEY = process.env.DASHBOARD_API_KEY ?? "";

async function get<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(`${API_BASE}${path}`);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      url.searchParams.set(k, String(v));
    }
  }
  const headers: Record<string, string> = {};
  if (API_KEY) {
    headers["X-API-Key"] = API_KEY;
  }
  const res = await fetch(url.toString(), {
    // Always fresh — trading data is live.
    cache: "no-store",
    headers,
  });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText} for ${path}`);
  }
  return res.json();
}

// ---------------- Types (mirror FastAPI models) ----------------

export type Position = {
  symbol: string;
  weight: number;
  side: "long" | "short";
  score: number;
};

export type PortfolioSnapshot = {
  as_of_date: string;
  n_longs: number;
  n_shorts: number;
  gross_exposure: number;
  net_exposure: number;
  positions: Position[];
};

export type EquityPoint = {
  date: string;
  equity: number;
  drawdown: number;
  gross_pnl: number;
  net_pnl: number;
};

export type MetricsSummary = {
  annual_return: number;
  annual_vol: number;
  sharpe: number;
  max_drawdown: number;
  hit_rate: number;
  avg_turnover: number;
  n_days: number;
  score_col: string;
};

export type Trade = {
  date: string;
  symbol: string;
  action: "buy" | "sell" | "short" | "cover";
  weight_before: number;
  weight_after: number;
  score: number;
};

export type BacktestRow = {
  score_col: string;
  annual_return: number;
  annual_vol: number;
  sharpe: number;
  max_drawdown: number;
  hit_rate: number;
  avg_turnover: number;
  n_days: number;
};

export type ScoredHeadline = {
  symbol: string;
  published_at: string;
  headline: string;
  sentiment_label: string | null;
  sentiment_score: number | null;
  catalyst_tags: string | null;
};

// ---------------- Functions ----------------

export const api = {
  health: () => get<{ status: string; service: string; version: string }>("/health"),
  portfolio: (scoreCol = "oof_meta_with_news") =>
    get<PortfolioSnapshot>("/api/portfolio", { score_col: scoreCol }),
  equityCurve: (scoreCol = "oof_meta_with_news", days = 365) =>
    get<EquityPoint[]>("/api/equity-curve", { score_col: scoreCol, days }),
  metrics: (scoreCol = "oof_meta_with_news") =>
    get<MetricsSummary>("/api/metrics", { score_col: scoreCol }),
  trades: (scoreCol = "oof_meta_with_news", limit = 50) =>
    get<Trade[]>("/api/trades", { score_col: scoreCol, limit }),
  backtestCompare: () => get<BacktestRow[]>("/api/backtest/compare"),
  news: (limit = 50, symbol?: string) => {
    const params: Record<string, string | number> = { limit };
    if (symbol) params.symbol = symbol;
    return get<ScoredHeadline[]>("/api/news/recent", params);
  },
};
