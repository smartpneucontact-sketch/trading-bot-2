// Trades page — chronological list of target-weight changes.

import { api } from "@/lib/api";
import { fmtSignedPct } from "@/lib/format";

export const dynamic = "force-dynamic";

const ACTION_COLORS: Record<string, string> = {
  buy: "text-emerald-400 bg-emerald-950/40 border-emerald-900/60",
  sell: "text-red-400 bg-red-950/40 border-red-900/60",
  short: "text-orange-400 bg-orange-950/40 border-orange-900/60",
  cover: "text-sky-400 bg-sky-950/40 border-sky-900/60",
};

export default async function TradesPage({
  searchParams,
}: {
  searchParams: Promise<{ score?: string; limit?: string }>;
}) {
  const { score = "oof_meta_with_news", limit = "100" } = await searchParams;
  const trades = await api.trades(score, parseInt(limit)).catch(() => []);

  return (
    <main className="max-w-7xl mx-auto px-6 py-8">
      <div className="flex items-baseline justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold">Trades</h1>
          <p className="text-sm text-zinc-500 mt-1">
            {trades.length} most-recent target-weight changes · {score}
          </p>
        </div>
      </div>

      <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead className="text-xs text-zinc-500 border-b border-zinc-800">
            <tr>
              <th className="text-left px-4 py-2">Date</th>
              <th className="text-left px-4 py-2">Symbol</th>
              <th className="text-left px-4 py-2">Action</th>
              <th className="text-right px-4 py-2">From</th>
              <th className="text-right px-4 py-2">To</th>
              <th className="text-right px-4 py-2">Score</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t, i) => (
              <tr
                key={`${t.date}-${t.symbol}-${i}`}
                className="border-b border-zinc-800/60 hover:bg-zinc-800/30"
              >
                <td className="px-4 py-2 tabular-nums text-zinc-400">{t.date}</td>
                <td className="px-4 py-2 font-mono font-medium">{t.symbol}</td>
                <td className="px-4 py-2">
                  <span
                    className={`px-2 py-0.5 text-xs uppercase rounded border ${
                      ACTION_COLORS[t.action] ?? ""
                    }`}
                  >
                    {t.action}
                  </span>
                </td>
                <td className="px-4 py-2 text-right tabular-nums text-zinc-500">
                  {fmtSignedPct(t.weight_before)}
                </td>
                <td className="px-4 py-2 text-right tabular-nums">
                  {fmtSignedPct(t.weight_after)}
                </td>
                <td className="px-4 py-2 text-right tabular-nums text-zinc-400">
                  {t.score.toFixed(4)}
                </td>
              </tr>
            ))}
            {!trades.length && (
              <tr>
                <td colSpan={6} className="px-4 py-12 text-center text-zinc-500">
                  No trades yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </main>
  );
}
