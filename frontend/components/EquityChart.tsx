// Server-facing wrapper — loads the Recharts component only on the client
// so ResponsiveContainer can measure its parent's dimensions without SSR
// emitting a "width(-1)" warning.

"use client";

import dynamic from "next/dynamic";
import { EquityPoint } from "@/lib/api";

const EquityChartClient = dynamic(() => import("./EquityChart.client"), {
  ssr: false,
  loading: () => (
    <div className="h-80 flex items-center justify-center text-zinc-500 text-sm">
      Loading chart…
    </div>
  ),
});

export function EquityChart({ data }: { data: EquityPoint[] }) {
  return <EquityChartClient data={data} />;
}
