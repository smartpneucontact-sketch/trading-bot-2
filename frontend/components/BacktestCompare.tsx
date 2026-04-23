"use client";

import { BacktestRow } from "@/lib/api";
import { fmtPct, fmtSignedPct } from "@/lib/format";

const PRETTY_NAMES: Record<string, string> = {
  oof_meta_quant_only: "Quant only",
  oof_meta_with_news: "Quant + FinBERT news",
  oof_meta_with_claude: "Quant + Claude news",
  oof_meta_with_all: "Quant + FinBERT + Claude",
};

export function BacktestCompare({ rows }: { rows: BacktestRow[] }) {
  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-zinc-800">
        <h3 className="text-sm font-medium text-zinc-200">Strategy comparison</h3>
        <p className="text-xs text-zinc-500 mt-1">
          Backtest across all meta variants. 1bps slippage, daily rebalance.
        </p>
      </div>
      <table className="w-full text-sm">
        <thead className="text-xs text-zinc-500 border-b border-zinc-800">
          <tr>
            <th className="text-left px-4 py-2">Variant</th>
            <th className="text-right px-4 py-2">Annual</th>
            <th className="text-right px-4 py-2">Vol</th>
            <th className="text-right px-4 py-2">Sharpe</th>
            <th className="text-right px-4 py-2">Max DD</th>
            <th className="text-right px-4 py-2">Hit</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const sharpeColor =
              r.sharpe >= 1 ? "text-emerald-400"
              : r.sharpe >= 0.5 ? "text-emerald-300"
              : r.sharpe >= 0 ? "text-zinc-300"
              : "text-red-400";
            const retColor = r.annual_return >= 0 ? "text-emerald-400" : "text-red-400";
            return (
              <tr key={r.score_col} className="border-b border-zinc-800/60 hover:bg-zinc-800/30">
                <td className="px-4 py-2">
                  <div className="font-medium text-zinc-100">
                    {PRETTY_NAMES[r.score_col] ?? r.score_col}
                  </div>
                  <div className="text-xs text-zinc-500 font-mono">{r.score_col}</div>
                </td>
                <td className={`px-4 py-2 text-right tabular-nums font-medium ${retColor}`}>
                  {fmtSignedPct(r.annual_return)}
                </td>
                <td className="px-4 py-2 text-right tabular-nums text-zinc-400">
                  {fmtPct(r.annual_vol)}
                </td>
                <td className={`px-4 py-2 text-right tabular-nums font-medium ${sharpeColor}`}>
                  {r.sharpe.toFixed(2)}
                </td>
                <td className="px-4 py-2 text-right tabular-nums text-red-400">
                  {fmtPct(r.max_drawdown)}
                </td>
                <td className="px-4 py-2 text-right tabular-nums text-zinc-400">
                  {fmtPct(r.hit_rate)}
                </td>
              </tr>
            );
          })}
          {!rows.length && (
            <tr>
              <td colSpan={6} className="px-4 py-8 text-center text-zinc-500">
                No backtests run yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
