"use client";

import LineChart from "./LineChart";

export default function QuestionChart({ points }: { points: { asked: number; auc: number; removed: string | null }[] }) {
  const sorted = [...points].sort((a, b) => a.asked - b.asked);
  return (
    <LineChart
      xs={sorted.map((p) => p.asked)}
      xLabel="Questions asked"
      xFormat={(v) => `${v}`}
      yMin={0.5}
      yMax={0.8}
      marker={{ x: 8, text: "What we ask" }}
      series={[{ name: "AUC", color: "var(--text)", values: sorted.map((p) => p.auc), format: (v) => v.toFixed(3) }]}
    />
  );
}
