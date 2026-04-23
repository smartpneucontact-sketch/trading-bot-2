// Number formatting helpers — consistent across the app.

export const fmtPct = (x: number, digits = 2) =>
  `${(x * 100).toFixed(digits)}%`;

export const fmtSignedPct = (x: number, digits = 2) => {
  const v = x * 100;
  return `${v >= 0 ? "+" : ""}${v.toFixed(digits)}%`;
};

export const fmtNum = (x: number, digits = 2) =>
  x.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

export const fmtSigned = (x: number, digits = 4) => {
  return `${x >= 0 ? "+" : ""}${x.toFixed(digits)}`;
};
