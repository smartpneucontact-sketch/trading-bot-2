// Backtest detail — equity curves for all 4 variants side by side.

import { BacktestCompare } from "@/components/BacktestCompare";
import { EquityChart } from "@/components/EquityChart";
import { api } from "@/lib/api";

export const dynamic = "force-dynamic";

const VARIANTS = [
  { key: "oof_meta_quant_only", label: "Quant only" },
  { key: "oof_meta_with_news", label: "Quant + FinBERT" },
  { key: "oof_meta_with_claude", label: "Quant + Claude" },
  { key: "oof_meta_with_all", label: "Quant + Both" },
];

export default async function BacktestPage() {
  const [compare, ...curves] = await Promise.all([
    api.backtestCompare().catch(() => []),
    ...VARIANTS.map((v) => api.equityCurve(v.key, 2000).catch(() => [])),
  ]);

  return (
    <main className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold">Backtests</h1>
        <p className="text-sm text-zinc-500 mt-1">
          All meta-learner variants, purged walk-forward CV, 1bps slippage.
        </p>
      </div>

      <section className="mb-6">
        <BacktestCompare rows={compare} />
      </section>

      <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {VARIANTS.map((v, i) => (
          <div
            key={v.key}
            className="bg-zinc-900/60 border border-zinc-800 rounded-lg p-4"
          >
            <div className="flex items-baseline justify-between mb-2">
              <h3 className="text-sm font-medium text-zinc-200">{v.label}</h3>
              <span className="text-xs text-zinc-500 font-mono">{v.key}</span>
            </div>
            <EquityChart data={curves[i]} />
          </div>
        ))}
      </section>
    </main>
  );
}
