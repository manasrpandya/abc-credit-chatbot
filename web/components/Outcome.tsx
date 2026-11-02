"use client";

import { useEffect, useRef, useState } from "react";
import { post, Result, rupees } from "@/lib/api";
import PriceBar from "./PriceBar";

function Verdict({ word, cls }: { word: string; cls: string }) {
  return (
    <h1 className={`verdict ${cls}`} aria-label={word}>
      {word.split("").map((c, i) => (
        <span key={i} aria-hidden style={{ animationDelay: `${i * 35}ms` }}>{c}</span>
      ))}
    </h1>
  );
}

function Gauge({ p, cut }: { p: number; cut: number }) {
  const top = Math.max(cut * 2.4, Math.min(p * 1.2, 0.8));
  const at = (v: number) => `${Math.min(v / top, 1) * 100}%`;
  // keep the label on screen at either end of the line
  const share = Math.min(p / top, 1);
  const shift = share < 0.2 ? "translateX(0)" : share > 0.8 ? "translateX(-100%)" : "translateX(-50%)";
  return (
    <div className="gauge">
      <b style={{ left: at(cut) }} />
      <span style={{ left: at(cut) }}>cut-off {(cut * 100).toFixed(1)}%</span>
      <i style={{ left: at(p) }} />
      <span className="you" style={{ left: at(p), transform: shift }}>this application {(p * 100).toFixed(1)}%</span>
    </div>
  );
}

function ShapBars({ rows }: { rows: Result["shap"] }) {
  const most = Math.max(...rows.map((r) => Math.abs(r.effect)), 0.01);
  return (
    <div>
      {rows.slice(0, 7).map((r) => {
        const width = (Math.abs(r.effect) / most) * 50;
        return (
          <div className="shap-row" key={r.factor}>
            <span>{r.label}</span>
            <div className="shap-track" title={`${r.effect > 0 ? "+" : ""}${r.effect.toFixed(3)}`}>
              <div className={r.effect > 0 ? "shap-fill up" : "shap-fill"}
                   style={r.effect > 0 ? { left: "50%", width: `${width}%` } : { right: "50%", width: `${width}%` }} />
            </div>
          </div>
        );
      })}
      <p className="why">Bars to the right pushed towards a decline, bars to the left towards approval.</p>
    </div>
  );
}

type Sim = { decision: string; probability: number };

export default function Outcome({ result, price, down, onRetry, onRestart }: {
  result: Result; price: number; down: number;
  onRetry: (down: number) => void; onRestart: () => void;
}) {
  const [note, setNote] = useState("");
  const [tryDown, setTryDown] = useState(down);
  const [sim, setSim] = useState<Sim>({ decision: result.decision, probability: result.probability });
  const [phone, setPhone] = useState("");
  const [phoneNote, setPhoneNote] = useState("");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const declined = result.decision === "declined";

  useEffect(() => {
    let live = true;
    post("summary", { state: result.state }).then((r) => live && setNote(r.text)).catch(() => {
      if (live) setNote(declined ? "Your application wasn't approved this time." : "Your application was approved.");
    });
    return () => { live = false; };
  }, [result.state, declined]);

  useEffect(() => {
    setTryDown(down);
    setSim({ decision: result.decision, probability: result.probability });
  }, [down, result.decision, result.probability]);

  function slide(v: number) {
    setTryDown(v);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      post<Sim>("simulate", { state: result.state, down: v }).then(setSim).catch(() => {});
    }, 90);
  }

  async function sendPhone(e: React.FormEvent) {
    e.preventDefault();
    const r = await post("callback", { state: result.state, phone }).catch(() => null);
    setPhoneNote(r?.message || "That didn't go through. Please try again.");
    if (r?.ok) setPhone("");
  }

  const maxDown = Math.round(price * 0.8 / 500) * 500;
  const changed = tryDown !== down;

  return (
    <main className="result">
      <Verdict word={declined ? "Declined" : "Approved"} cls={result.decision} />
      <p className="verdict-note" aria-live="polite">{note}</p>

      <div className="result-grid">
        <div>
          <section>
            <h2>What influenced this</h2>
            <ul className="reasons">
              {result.reasons.map((r) => (
                <li key={r.factor}>{r.text}<small>{r.label}</small></li>
              ))}
            </ul>
            {result.against.length > 0 && (
              <ul className="reasons other">
                {result.against.map((r) => (
                  <li key={r.factor}>{r.text}<small>Working the other way</small></li>
                ))}
              </ul>
            )}
            <p className="fine">{result.disclaimer}</p>
          </section>

          <section>
            <h2>What happens next</h2>
            <ol className="steps">
              {result.next_steps.map((s) => <li key={s}>{s}</li>)}
            </ol>
            {declined && (
              <form className="callback" onSubmit={sendPhone}>
                <div className="typed">
                  <input value={phone} onChange={(e) => setPhone(e.target.value)} inputMode="tel"
                         placeholder="Your mobile number" aria-label="Mobile number for a callback" />
                  <button type="submit" className="chip">Ask for a callback</button>
                </div>
              </form>
            )}
            {phoneNote && <p className="why">{phoneNote}</p>}
            <p style={{ marginTop: 22 }}>
              <button className="quiet" onClick={onRestart}>Start a new application</button>
            </p>
          </section>
        </div>

        <aside>
          <section className="slider-box">
            <h2>Try a different down payment</h2>
            <p>Drag to see what the same model says. This doesn't change your result unless you apply with it.</p>
            <PriceBar price={price} down={tryDown} />
            <input type="range" min={0} max={maxDown} step={500} value={tryDown}
                   onChange={(e) => slide(Number(e.target.value))}
                   aria-label="Down payment" aria-valuetext={rupees(tryDown)} />
            <div className="sim-line" aria-live="polite">
              <span className={`sim-word ${sim.decision}`}>{sim.decision === "declined" ? "Declined" : "Approved"}</span>
              {changed && <button className="primary" onClick={() => onRetry(tryDown)}>Apply with {rupees(tryDown)}</button>}
            </div>
            {result.what_if && !changed && (
              <p style={{ margin: "14px 0 0" }}>
                About {rupees(result.what_if.extra_down_payment)} more upfront would likely change the result.
              </p>
            )}
          </section>
        </aside>
      </div>

      <details className="under">
        <summary>See how this was decided</summary>
        <div className="inner">
          <div>
            <p className="why" style={{ marginTop: 0 }}>Estimated chance this application would be declined, against the cut-off we chose.</p>
            <Gauge p={sim.probability} cut={result.threshold} />
            <ShapBars rows={result.shap} />
          </div>
          <dl>
            <dt>Decision made by</dt><dd>{result.model_kind === "lightgbm" ? "LightGBM model" : "LightGBM and CatBoost blend"}</dd>
            <dt>Model version</dt><dd>{result.model_version}</dd>
            <dt>Time to score</dt><dd>{result.timing.decision_ms} ms</dd>
            <dt>Time to work out reasons</dt><dd>{result.timing.shap_ms} ms</dd>
            <dt>Time to write the log</dt><dd>{result.timing.log_ms} ms</dd>
            <dt>Total on the server</dt><dd>{result.timing.total_ms} ms</dd>
            <dt>Application took</dt><dd>{result.timing.seconds_to_complete} s</dd>
            <dt>Language model</dt><dd>{result.llm_available ? "Only rewords the text" : "Not connected, scripted text"}</dd>
          </dl>
        </div>
      </details>
    </main>
  );
}
