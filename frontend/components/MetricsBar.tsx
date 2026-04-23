"use client";

import { MetricsSummary } from "@/lib/api";
import { fmtPct, fmtSignedPct } from "@/lib/format";

function Tile({
  label,
  value,
  sub,
  tone = "neutral",
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "pos" | "neg" | "neutral";
}) {
  const valueColor =
    tone === "pos" ? "text-emerald-400"
    : tone === "neg" ? "text-red-400"
    : "text-zinc-100";
  return (
    <div className="flex flex-col gap-1 bg-zinc-900/60 border border-zinc-800 rounded-lg px-4 py-3 min-w-[140px]">
      <span className="text-xs uppercase tracking-wide text-zinc-500">{label}</span>
      <span className={`text-2xl font-semibold tabular-nums ${valueColor}`}>{value}</span>
      {sub && <span className="text-xs text-zinc-500">{sub}</span>}
    </div>
  );
}

export function MetricsBar({ m }: { m: MetricsSummary }) {
  const sharpeTone = m.sharpe >= 1 ? "pos" : m.sharpe < 0 ? "neg" : "neutral";
  const retTone = m.annual_return >= 0 ? "pos" : "neg";

  return (
    <div className="flex flex-wrap gap-3">
      <Tile
        label="Annual Return"
        value={fmtSignedPct(m.annual_return)}
        tone={retTone}
      />
      <Tile
        label="Sharpe"
        value={m.sharpe.toFixed(2)}
        tone={sharpeTone}
      />
      <Tile label="Annual Vol" value={fmtPct(m.annual_vol)} />
      <Tile
        label="Max Drawdown"
        value={fmtPct(m.max_drawdown)}
        tone={m.max_drawdown < -0.2 ? "neg" : "neutral"}
      />
      <Tile label="Hit Rate" value={fmtPct(m.hit_rate)} />
      <Tile label="Turnover / day" value={m.avg_turnover.toFixed(2)} />
      <Tile label="Days" value={m.n_days.toString()} sub={m.score_col} />
    </div>
  );
}
