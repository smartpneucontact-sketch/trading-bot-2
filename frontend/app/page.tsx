// Dashboard — main entry view.
// Server component that parallel-fetches everything it needs in one render pass.

import { BacktestCompare } from "@/components/BacktestCompare";
import { EquityChart } from "@/components/EquityChart";
import { MetricsBar } from "@/components/MetricsBar";
import { PositionsTable } from "@/components/PositionsTable";
import { api } from "@/lib/api";

export const dynamic = "force-dynamic";  // always fetch fresh

export default async function Dashboard({
  searchParams,
}: {
  searchParams: Promise<{ score?: string }>;
}) {
  const { score = "oof_meta_with_news" } = await searchParams;

  // Parallel fetch — all endpoints are independent.
  const [metrics, equity, portfolio, backtests] = await Promise.all([
    api.metrics(score).catch(() => null),
    api.equityCurve(score, 2000).catch(() => []),
    api.portfolio(score).catch(() => null),
    api.backtestCompare().catch(() => []),
  ]);

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="flex items-baseline justify-between mb-6">
          <div>
            <h1 className="text-2xl font-semibold">Bot 8 · Dashboard</h1>
            <p className="text-sm text-zinc-500 mt-1">
              Long/short equity · Daily rebalance · Meta-learner:{" "}
              <span className="font-mono text-zinc-300">{score}</span>
            </p>
          </div>
          <VariantSwitcher current={score} />
        </div>

        {metrics ? (
          <section className="mb-6">
            <MetricsBar m={metrics} />
          </section>
        ) : (
          <section className="mb-6 text-zinc-500">Metrics unavailable.</section>
        )}

        <section className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-5 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-medium text-zinc-200">Equity curve</h2>
            <span className="text-xs text-zinc-500">{equity.length} days</span>
          </div>
          <EquityChart data={equity} />
        </section>

        <section className="mb-6">
          <BacktestCompare rows={backtests} />
        </section>

        <section>
          <div className="flex items-baseline justify-between mb-3">
            <h2 className="text-sm font-medium text-zinc-200">Current target portfolio</h2>
            {portfolio && (
              <span className="text-xs text-zinc-500 tabular-nums">
                {portfolio.as_of_date} · gross{" "}
                {(portfolio.gross_exposure * 100).toFixed(1)}% · net{" "}
                {(portfolio.net_exposure * 100).toFixed(2)}%
              </span>
            )}
          </div>
          {portfolio ? (
            <PositionsTable positions={portfolio.positions} />
          ) : (
            <div className="text-zinc-500 text-sm">No portfolio data.</div>
          )}
        </section>
      </div>
    </main>
  );
}

function VariantSwitcher({ current }: { current: string }) {
  const options = [
    { val: "oof_meta_quant_only", label: "Quant" },
    { val: "oof_meta_with_news", label: "+ FinBERT" },
    { val: "oof_meta_with_claude", label: "+ Claude" },
    { val: "oof_meta_with_all", label: "All" },
  ];
  return (
    <div className="flex gap-1 bg-zinc-900 border border-zinc-800 rounded-lg p-1">
      {options.map((o) => {
        const active = o.val === current;
        return (
          <a
            key={o.val}
            href={`/?score=${o.val}`}
            className={`px-3 py-1 text-xs rounded-md transition ${
              active
                ? "bg-zinc-700 text-zinc-100"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            {o.label}
          </a>
        );
      })}
    </div>
  );
}
