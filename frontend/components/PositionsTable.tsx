"use client";

import { Position } from "@/lib/api";
import { fmtSignedPct } from "@/lib/format";

export function PositionsTable({ positions }: { positions: Position[] }) {
  const longs = positions.filter((p) => p.side === "long");
  const shorts = positions.filter((p) => p.side === "short");

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <Column title="Longs" rows={longs} tone="pos" />
      <Column title="Shorts" rows={shorts} tone="neg" />
    </div>
  );
}

function Column({
  title,
  rows,
  tone,
}: {
  title: string;
  rows: Position[];
  tone: "pos" | "neg";
}) {
  const labelColor = tone === "pos" ? "text-emerald-400" : "text-red-400";
  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg overflow-hidden">
      <div className="px-4 py-2 border-b border-zinc-800 flex items-center justify-between">
        <span className={`text-sm font-medium ${labelColor}`}>{title}</span>
        <span className="text-xs text-zinc-500">{rows.length} positions</span>
      </div>
      <div className="max-h-[500px] overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="text-xs text-zinc-500 sticky top-0 bg-zinc-900 border-b border-zinc-800">
            <tr>
              <th className="text-left px-4 py-2">Symbol</th>
              <th className="text-right px-4 py-2">Weight</th>
              <th className="text-right px-4 py-2">Score</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.symbol} className="border-b border-zinc-800/60 hover:bg-zinc-800/30">
                <td className="px-4 py-2 font-mono">{p.symbol}</td>
                <td className={`px-4 py-2 text-right tabular-nums ${labelColor}`}>
                  {fmtSignedPct(p.weight)}
                </td>
                <td className="px-4 py-2 text-right tabular-nums text-zinc-400">
                  {p.score.toFixed(4)}
                </td>
              </tr>
            ))}
            {!rows.length && (
              <tr>
                <td colSpan={3} className="px-4 py-8 text-center text-zinc-500 text-sm">
                  No {title.toLowerCase()} today
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
