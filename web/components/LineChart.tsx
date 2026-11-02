"use client";

import { useState } from "react";

type Series = { name: string; color: string; values: number[]; format: (v: number) => string };

export default function LineChart({ xs, series, xLabel, marker, xFormat, yMax, yMin = 0 }: {
  xs: number[]; series: Series[]; xLabel: string;
  marker?: { x: number; text: string };
  xFormat: (v: number) => string; yMax?: number; yMin?: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const w = 760, h = 300, left = 46, right = 92, top = 16, bottom = 44;
  const lo = Math.min(...xs), hi = Math.max(...xs);
  const every = series.flatMap((s) => s.values);
  const ceil = yMax ?? Math.max(...every) * 1.05;
  const px = (x: number) => left + ((x - lo) / (hi - lo || 1)) * (w - left - right);
  const py = (y: number) => top + (1 - (y - yMin) / (ceil - yMin)) * (h - top - bottom);
  const ticks = Array.from({ length: 5 }, (_, i) => yMin + ((ceil - yMin) * i) / 4);

  function move(e: React.PointerEvent<SVGRectElement>) {
    const box = e.currentTarget.getBoundingClientRect();
    const x = lo + ((e.clientX - box.left) / box.width) * (hi - lo);
    let best = 0;
    xs.forEach((v, i) => { if (Math.abs(v - x) < Math.abs(xs[best] - x)) best = i; });
    setHover(best);
  }

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`${series.map((s) => s.name).join(" and ")} against ${xLabel}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={left} x2={w - right} y1={py(t)} y2={py(t)} stroke="var(--line)" strokeWidth="1" />
            <text x={left - 8} y={py(t) + 4} textAnchor="end">{series[0].format(t)}</text>
          </g>
        ))}
        {xs.filter((_, i) => i % Math.ceil(xs.length / 7) === 0).map((x) => (
          <text key={x} x={px(x)} y={h - bottom + 18} textAnchor="middle">{xFormat(x)}</text>
        ))}
        <text x={(left + w - right) / 2} y={h - 4} textAnchor="middle">{xLabel}</text>
        {marker && (
          <g>
            <line x1={px(marker.x)} x2={px(marker.x)} y1={top} y2={h - bottom} stroke="var(--ember)" strokeWidth="1.5" />
            <text x={px(marker.x) + 6} y={h - bottom - 8} className="label">{marker.text}</text>
          </g>
        )}
        {series.map((s) => (
          <g key={s.name}>
            <path d={s.values.map((v, i) => `${i ? "L" : "M"}${px(xs[i])},${py(v)}`).join(" ")}
                  fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
            <text x={px(xs[xs.length - 1]) + 8} y={py(s.values[s.values.length - 1]) + 4} className="label">{s.name}</text>
          </g>
        ))}
        {hover !== null && (
          <g>
            <line x1={px(xs[hover])} x2={px(xs[hover])} y1={top} y2={h - bottom} stroke="var(--faint)" strokeWidth="1" />
            {series.map((s) => (
              <circle key={s.name} cx={px(xs[hover])} cy={py(s.values[hover])} r="4.5" fill={s.color} stroke="var(--ink)" strokeWidth="2" />
            ))}
          </g>
        )}
        <rect x={left} y={top} width={w - left - right} height={h - top - bottom} fill="transparent"
              onPointerMove={move} onPointerLeave={() => setHover(null)} />
      </svg>
      {hover !== null && (
        <div className="tip" style={{ left: `${(px(xs[hover]) / w) * 100}%`, top: `${(py(Math.max(...series.map((s) => s.values[hover]))) / h) * 100}%` }}>
          <b>{xFormat(xs[hover])}</b>
          {series.map((s) => <div key={s.name}>{s.name}: {s.format(s.values[hover])}</div>)}
        </div>
      )}
    </div>
  );
}
