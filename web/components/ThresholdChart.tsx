"use client";

import LineChart from "./LineChart";

type Row = { threshold: number; precision: number; recall: number };

export default function ThresholdChart({ curve, cut }: { curve: Row[]; cut: number }) {
  const rows = curve.filter((r) => r.threshold <= 0.42);
  const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
  return (
    <LineChart
      xs={rows.map((r) => r.threshold)}
      xLabel="Probability cut-off"
      xFormat={pct}
      yMax={1}
      marker={{ x: cut, text: "Our cut-off" }}
      series={[
        { name: "Recall", color: "var(--ember)", values: rows.map((r) => r.recall), format: pct },
        { name: "Precision", color: "var(--text)", values: rows.map((r) => r.precision), format: pct },
      ]}
    />
  );
}
