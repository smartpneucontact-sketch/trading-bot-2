// Top nav — same on every page. Uses a client component only for usePathname
// so the active-link highlight works without full-page reloads.

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Dashboard" },
  { href: "/trades", label: "Trades" },
  { href: "/news", label: "News" },
  { href: "/backtest", label: "Backtests" },
];

export function NavHeader() {
  const pathname = usePathname();

  return (
    <header className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur sticky top-0 z-10">
      <div className="max-w-7xl mx-auto px-6 py-3 flex items-center gap-6">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400" />
          <span className="font-semibold text-zinc-100">Bot 8</span>
        </Link>
        <nav className="flex gap-1">
          {LINKS.map((l) => {
            const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`px-3 py-1.5 text-sm rounded-md transition ${
                  active
                    ? "bg-zinc-800 text-zinc-100"
                    : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto text-xs text-zinc-500">
          paper · {process.env.NEXT_PUBLIC_API_BASE ?? "localhost:8000"}
        </div>
      </div>
    </header>
  );
}
