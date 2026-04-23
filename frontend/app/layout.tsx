import type { Metadata } from "next";
import { NavHeader } from "@/components/NavHeader";
import "./globals.css";

export const metadata: Metadata = {
  title: "Bot 8 · Dashboard",
  description: "ML-driven long/short equity — monitoring & control",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="min-h-full flex flex-col bg-zinc-950 text-zinc-100">
        <NavHeader />
        <div className="flex-1">{children}</div>
      </body>
    </html>
  );
}
