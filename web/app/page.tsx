"use client";

import { useEffect, useRef, useState } from "react";
import Outcome from "@/components/Outcome";
import PriceBar from "@/components/PriceBar";
import { post, Question, Result, Turn } from "@/lib/api";

type Line = { id: number; from: "bot" | "me"; text: string; small?: boolean };

export default function Home() {
  const [turn, setTurn] = useState<Turn | null>(null);
  const [lines, setLines] = useState<Line[]>([]);
  const [busy, setBusy] = useState(false);
  const [typed, setTyped] = useState("");
  const [problem, setProblem] = useState("");
  const [showWhy, setShowWhy] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [deciding, setDeciding] = useState(false);
  const counter = useRef(0);
  const logEnd = useRef<HTMLDivElement>(null);
  const field = useRef<HTMLInputElement>(null);

  const say = (from: Line["from"], text: string, small?: boolean) =>
    setLines((all) => [...all, { id: ++counter.current, from, text, small }]);

  function takeTurn(next: Turn) {
    next.messages.forEach((m) => say("bot", m, Boolean(next.question)));
    if (next.question) say("bot", next.question.text);
    setTurn(next);
    setShowWhy(false);
  }

  useEffect(() => {
    logEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    if (window.matchMedia("(min-width: 861px)").matches) field.current?.focus();
  }, [lines, busy, turn?.stage]);

  async function run(work: () => Promise<void>) {
    setBusy(true);
    setProblem("");
    try {
      await work();
    } catch (err) {
      setProblem(err instanceof Error ? err.message : "Something went wrong. Please try again.");
    }
    setBusy(false);
  }

  const begin = () => run(async () => {
    setLines([]);
    setResult(null);
    takeTurn(await post<Turn>("start"));
  });

  const reply = (shown: string, body: { text?: string; value?: string }) => {
    if (!turn) return;
    say("me", shown);
    setTyped("");
    run(async () => takeTurn(await post<Turn>("answer", { state: turn.state, ...body })));
  };

  function sendTyped(e: React.FormEvent) {
    e.preventDefault();
    const text = typed.trim();
    if (text) reply(text, { text });
  }

  const change = (key: string) => {
    if (!turn) return;
    run(async () => {
      const next = await post<Turn>("edit", { state: turn.state, key });
      setResult(null);
      takeTurn(next);
    });
  };

  const decide = (state: string) => run(async () => {
    setDeciding(true);
    setResult(await post<Result>("submit", { state }));
    setDeciding(false);
  }).finally(() => setDeciding(false));

  async function retry(down: number) {
    if (!turn) return;
    const cleared = await post<Turn>("edit", { state: result?.state ?? turn.state, key: "down_payment" });
    const filled = await post<Turn>("answer", { state: cleared.state, value: String(down) });
    setTurn(filled);
    decide(filled.state);
  }

  if (!turn && !busy) {
    return (
      <main className="intro">
        <h1>Your two-wheeler loan answer, in about two minutes</h1>
        <PriceBar price={90000} down={18000} opening />
        <p className="lede">Eight short questions. No documents yet. You&apos;ll see the answer and what influenced it straight away.</p>
        <button className="primary" onClick={begin}>Start my application</button>
        {problem && <p className="error-line">{problem}</p>}
        <p className="privacy">
          By starting you agree that we keep a masked copy of your answers for audit. Your pincode is cut to three
          digits, and anything like a phone number or PAN typed into the chat is removed before it is stored or sent
          to our language model.
        </p>
      </main>
    );
  }

  if (result && turn?.price != null && turn.down != null) {
    return <Outcome result={result} price={turn.price} down={turn.down} onRetry={retry}
                    onRestart={() => { setTurn(null); setResult(null); setLines([]); }} />;
  }

  const q: Question | null = turn?.question ?? null;
  const share = turn ? (turn.progress.done / Math.max(turn.progress.total, 1)) * 100 : 0;

  return (
    <>
      <div className="progress" style={{ width: `${share}%` }} />
      <main className="app">
        <div className="talk">
          <div className="log" aria-live="polite">
            {lines.map((l) => (
              <div key={l.id} className={`msg ${l.from}${l.small ? " small" : ""}`}>{l.text}</div>
            ))}
            {(busy || deciding) && <div className="dots" aria-label="Working"><i /><i /><i /></div>}
          </div>

          {turn?.stage === "review" && !busy && (
            <div className="review">
              <h2>Does this look right?</h2>
              <ul className="rows">
                {turn.answers.map((a) => (
                  <li key={a.label}>
                    <span>{a.label}</span>
                    <span>{a.value}{!a.derived && (
                      <button className="quiet" onClick={() => change(a.key)} aria-label={`Change ${a.label}`}>Change</button>
                    )}</span>
                  </li>
                ))}
              </ul>
              <button className="primary" onClick={() => decide(turn.state)}>Get my decision</button>
            </div>
          )}

          {turn?.stage === "ineligible" && (
            <p><button className="quiet" onClick={() => { setTurn(null); setLines([]); }}>Back to the start</button></p>
          )}

          {turn?.stage !== "asking" && <div ref={logEnd} />}

          {q && (
            <div className="answer-area">
              {q.chips.length > 0 && !busy && (
                <div className="chips">
                  {q.chips.map((c) => (
                    <button key={c.value} className="chip" onClick={() => reply(c.label, { value: c.value })}>{c.label}</button>
                  ))}
                </div>
              )}
              {(q.kind !== "chips" || q.key === "employment" || q.key === "active_loans") && (
                <form className="typed" onSubmit={sendTyped}>
                  <input ref={field} value={typed} onChange={(e) => setTyped(e.target.value)}
                         placeholder={q.placeholder || "Or type your answer"} aria-label={q.text} autoComplete="off" />
                  <button type="submit" className="primary" disabled={!typed.trim() || busy}>Send</button>
                </form>
              )}
              <p className="why">
                {showWhy ? q.why : <button className="quiet" onClick={() => setShowWhy(true)}>Why we ask this</button>}
              </p>
              {problem && <p className="error-line">{problem}</p>}
              <div ref={logEnd} />
            </div>
          )}
        </div>

        <aside className="sheet">
          <h2>Your application</h2>
          {turn?.price != null && <PriceBar price={turn.price} down={turn.down} />}
          {turn && turn.answers.length > 0 ? (
            <ul className="rows">
              {turn.answers.map((a) => (
                <li key={a.label}><span>{a.label}</span><span>{a.value}</span></li>
              ))}
            </ul>
          ) : (
            <p className="sheet-empty">Your answers show up here as you go.</p>
          )}
        </aside>
      </main>
    </>
  );
}
