// Recharts' ResponsiveContainer can't measure layout during SSR, so we render
// the chart client-only. The parent uses next/dynamic with ssr:false.

"use client";

import { EquityPoint } from "@/lib/api";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function EquityChartClient({ data }: { data: EquityPoint[] }) {
  if (!data.length) {
    return (
      <div className="h-80 flex items-center justify-center text-zinc-500">
        No equity data — run a backtest first.
      </div>
    );
  }

  const chartData = data.map((p) => ({
    date: p.date,
    equity: p.equity,
    drawdown: p.drawdown * 100,
  }));

  return (
    <div className="h-80 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 10, right: 16, left: 8, bottom: 8 }}>
          <defs>
            <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#10b981" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#27272a" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="date"
            stroke="#71717a"
            fontSize={11}
            tickFormatter={(d: string) => d.slice(0, 7)}
            minTickGap={40}
          />
          <YAxis
            stroke="#71717a"
            fontSize={11}
            tickFormatter={(v: number) => v.toFixed(2)}
            domain={["auto", "auto"]}
            width={50}
          />
          <Tooltip
            contentStyle={{
              background: "#18181b",
              border: "1px solid #3f3f46",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "#a1a1aa" }}
            formatter={(value, name) => {
              // Recharts types `value` as ValueType | undefined (can be string,
              // number, or an array). Normalize to a number before formatting.
              const n = typeof value === "number" ? value : Number(value ?? 0);
              if (name === "equity") return [n.toFixed(4), "Equity"];
              if (name === "drawdown") return [`${n.toFixed(2)}%`, "Drawdown"];
              return [String(value ?? ""), String(name ?? "")];
            }}
          />
          <Area
            type="monotone"
            dataKey="equity"
            stroke="#10b981"
            strokeWidth={2}
            fill="url(#equityGradient)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
