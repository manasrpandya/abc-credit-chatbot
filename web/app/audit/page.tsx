"use client";

import { useCallback, useEffect, useState } from "react";

type Session = { session_id: string; started: number; events: number; decision: { decision: string; probability: number } | null };
type Event = { seq: number; at_ms: number; kind: string; body: unknown };

const when = (ms: number) => new Date(ms).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" });

export default function Audit() {
  const [key, setKey] = useState("");
  const [sessions, setSessions] = useState<Session[] | null>(null);
  const [store, setStore] = useState("");
  const [problem, setProblem] = useState("");
  const [open, setOpen] = useState<{ id: string; events: Event[]; intact: boolean } | null>(null);

  const load = useCallback(async (k: string, quiet = false) => {
    const res = await fetch("/api/audit/sessions", { headers: { "x-audit-key": k } });
    if (!res.ok) {
      if (!quiet) setProblem("That key wasn't accepted.");
      return;
    }
    const data = await res.json();
    setSessions(data.sessions);
    setStore(data.store);
    setProblem("");
    try { sessionStorage.setItem("audit-key", k); } catch {}
  }, []);

  useEffect(() => {
    let saved = "";
    try { saved = sessionStorage.getItem("audit-key") || ""; } catch {}
    setKey(saved);
    load(saved, true);
  }, [load]);

  async function show(id: string) {
    const res = await fetch(`/api/audit/sessions/${id}`, { headers: { "x-audit-key": key } });
    if (res.ok) {
      const d = await res.json();
      setOpen({ id, events: d.events, intact: d.chain_intact });
    }
  }

  return (
    <main className="page">
      <h1>Audit log</h1>
      <p>Every question, answer, feature value and score, with personal details masked. Each entry is chained to the one before it, so a quiet edit shows up.</p>

      {!sessions && (
        <form className="key-form" onSubmit={(e) => { e.preventDefault(); load(key); }}>
          <div className="typed">
            <input type="password" value={key} onChange={(e) => setKey(e.target.value)} placeholder="Audit key" aria-label="Audit key" />
            <button className="primary" type="submit">Open</button>
          </div>
        </form>
      )}
      {problem && <p className="error-line">{problem}</p>}

      {sessions && !open && (
        <>
          <p>{sessions.length ? `Latest ${sessions.length} sessions, stored in ${store}.` : "No sessions yet. Finish an application and it will show up here."}</p>
          {sessions.length > 0 && (
            <div className="session-row head"><span>Started</span><span>Result</span><span>Chance of decline</span><span>Entries</span></div>
          )}
          {sessions.map((s) => (
            <button key={s.session_id} className="session-row" onClick={() => show(s.session_id)}>
              <span>{when(s.started)}</span>
              <span>{s.decision ? s.decision.decision : "Not finished"}</span>
              <span>{s.decision ? `${(s.decision.probability * 100).toFixed(1)}%` : ""}</span>
              <span>{s.events}</span>
            </button>
          ))}
        </>
      )}

      {open && (
        <>
          <p><button className="quiet" onClick={() => setOpen(null)}>Back to all sessions</button></p>
          <p>Session {open.id}. {open.intact ? "The hash chain checks out." : "The hash chain is broken, so this log may have been edited."}</p>
          {open.events.map((e) => (
            <div key={e.seq} className="event">
              <header><b>{e.kind.replaceAll("_", " ")}</b><span>{new Date(e.at_ms).toLocaleTimeString("en-IN")}</span></header>
              <pre>{JSON.stringify(e.body, null, 1)}</pre>
            </div>
          ))}
        </>
      )}
    </main>
  );
}
