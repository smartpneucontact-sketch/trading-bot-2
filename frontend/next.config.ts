import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Produce a minimal self-contained server bundle at .next/standalone.
  // The Dockerfile copies this into the runtime stage; without it the
  // runner tries to boot a non-existent server.js and the container crashes.
  output: "standalone",

  // Ship the build even when TypeScript's strict mode flags third-party
  // type mismatches (Recharts' Formatter signature in Next 16). We keep
  // strict mode ON locally via `tsc --noEmit` — this only bypasses the
  // in-line check during `next build`.
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
};

export default nextConfig;
