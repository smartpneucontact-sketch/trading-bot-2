// Scored news feed — what the bot is seeing, ranked by recency + confidence.

import { api } from "@/lib/api";

export const dynamic = "force-dynamic";

function sentimentColor(score: number | null) {
  if (score === null) return "text-zinc-500";
  if (score > 0.3) return "text-emerald-400";
  if (score < -0.3) return "text-red-400";
  return "text-zinc-400";
}

function sentimentBar(score: number | null) {
  if (score === null) return null;
  const pct = Math.abs(score) * 100;
  const bg = score > 0 ? "bg-emerald-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2 w-24">
      {score < 0 ? (
        <>
          <div className="flex-1 flex justify-end">
            <div className={`h-1 ${bg}`} style={{ width: `${pct}%` }} />
          </div>
          <span className="w-px h-3 bg-zinc-700" />
          <div className="flex-1" />
        </>
      ) : (
        <>
          <div className="flex-1" />
          <span className="w-px h-3 bg-zinc-700" />
          <div className="flex-1">
            <div className={`h-1 ${bg}`} style={{ width: `${pct}%` }} />
          </div>
        </>
      )}
    </div>
  );
}

export default async function NewsPage({
  searchParams,
}: {
  searchParams: Promise<{ symbol?: string; limit?: string }>;
}) {
  const { symbol, limit = "100" } = await searchParams;
  const news = await api.news(parseInt(limit), symbol).catch(() => []);

  return (
    <main className="max-w-7xl mx-auto px-6 py-8">
      <div className="flex items-baseline justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold">News feed</h1>
          <p className="text-sm text-zinc-500 mt-1">
            {news.length} scored headlines
            {symbol ? ` · ${symbol}` : " · all tickers"}
          </p>
        </div>
        {symbol && (
          <a
            href="/news"
            className="text-xs text-zinc-400 hover:text-zinc-200 border border-zinc-800 rounded-md px-3 py-1"
          >
            Clear filter
          </a>
        )}
      </div>

      <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg overflow-hidden">
        <div className="divide-y divide-zinc-800/60">
          {news.map((n, i) => (
            <div
              key={`${n.symbol}-${n.published_at}-${i}`}
              className="px-4 py-3 hover:bg-zinc-800/30 flex gap-4 items-center"
            >
              <div className="shrink-0 w-24">
                <a
                  href={`/news?symbol=${n.symbol}`}
                  className="font-mono text-sm font-medium hover:text-emerald-400"
                >
                  {n.symbol}
                </a>
                <div className="text-xs text-zinc-500 tabular-nums">
                  {new Date(n.published_at).toLocaleDateString()}
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm text-zinc-200 truncate">{n.headline}</div>
                {n.catalyst_tags && (
                  <div className="flex gap-1 mt-1">
                    {n.catalyst_tags.split(",").filter(Boolean).map((tag) => (
                      <span
                        key={tag}
                        className="text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <div className="shrink-0 text-right">
                <div className={`text-sm font-medium tabular-nums ${sentimentColor(n.sentiment_score)}`}>
                  {n.sentiment_score !== null ? n.sentiment_score.toFixed(2) : "—"}
                </div>
                {sentimentBar(n.sentiment_score)}
              </div>
            </div>
          ))}
          {!news.length && (
            <div className="px-4 py-12 text-center text-zinc-500">
              No scored headlines found.
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
