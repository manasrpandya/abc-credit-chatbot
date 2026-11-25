"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

type Decision = { decision: string; probability: number; threshold?: number; seconds_to_complete?: number; model_version?: string };
type Session = { session_id: string; started: number; events: number; decision: Decision | null };
type Event = { seq: number; at_ms: number; kind: string; body: Record<string, any> };

const when = (ms: number) => new Date(ms).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" });
const clock = (ms: number) => new Date(ms).toLocaleTimeString("en-IN", { timeStyle: "short" });
const pct = (p: number) => `${(p * 100).toFixed(1)}%`;
const rupees = (v: number) => `₹${Math.round(v).toLocaleString("en-IN")}`;

const labels: Record<string, string> = {
  vehicle: "Vehicle", on_road_price: "On-road price", down_payment: "Down payment", pincode: "Pincode",
  age: "Age", employment: "Employment", income: "Monthly income", active_loans: "Loans running",
};
const work: Record<string, string> = {
  SAL: "Salaried", SEP: "Self-employed", AGR: "Farming", NREGI: "Irregular income", NPP: "On probation",
  STU: "Student", PEN: "Pensioner", NONEARNMEM: "Not earning",
};
const how: Record<string, string> = { typed: "typed", button: "tapped", llm: "read from the chat" };

function shown(key: string, v: any) {
  if (v === null || v === undefined) return "";
  if (["on_road_price", "down_payment", "income"].includes(key) && typeof v === "number") return rupees(v);
  if (key === "employment") return work[v] || String(v);
  if (key === "active_loans") return v ? "Yes" : "No";
  return String(v);
}

// one readable line per thing that happened, leaving out the plumbing
function story(events: Event[]) {
  const lines: { t: number; who: "bot" | "user" | "note"; text: string }[] = [];
  let lastAsked = "";
  for (const e of events) {
    const b = e.body || {};
    if (e.kind === "question_asked") {
      if (b.text !== lastAsked) lines.push({ t: e.at_ms, who: "bot", text: b.text });
      lastAsked = b.text;
    } else if (e.kind === "answer") {
      lines.push({ t: e.at_ms, who: "user", text: `${labels[b.key] || b.key}: ${shown(b.key, b.value)}  (${how[b.source] || b.source}${b.source !== "button" && b.text ? `, "${b.text}"` : ""})` });
    } else if (e.kind === "answer_rejected") {
      lines.push({ t: e.at_ms, who: "note", text: `Not accepted for ${labels[b.key] || b.key}: "${b.text}". ${b.reason}` });
    } else if (e.kind === "answer_cleared") {
      lines.push({ t: e.at_ms, who: "note", text: `${labels[b.key] || b.key} cleared${b.why ? `, ${b.why}` : ""}.` });
    } else if (e.kind === "policy_flag" || e.kind === "policy_stop") {
      lines.push({ t: e.at_ms, who: "note", text: `Policy: ${b.rule}.` });
    } else if (e.kind === "price_lookup") {
      const f = b.found;
      lines.push({ t: e.at_ms, who: "note", text: f?.low ? `Looked up ${b.vehicle}: about ${rupees(f.low)} to ${rupees(f.high)} (${f.source}).` : `Looked up ${b.vehicle}: ${f?.is_two_wheeler === false ? "not a two-wheeler" : "nothing reliable found"}.` });
    } else if (e.kind === "llm_reply_blocked") {
      lines.push({ t: e.at_ms, who: "note", text: "A language model reply was blocked because it talked about the decision." });
    } else if (e.kind === "callback_requested") {
      lines.push({ t: e.at_ms, who: "note", text: "Asked for a callback." });
    }
  }
  return lines;
}

function Detail({ id, events, intact, back }: { id: string; events: Event[]; intact: boolean; back: () => void }) {
  const find = (k: string) => events.find((e) => e.kind === k)?.body;
  const app = find("application");
  const dec = find("decision") as Decision | undefined;
  const why = find("reasons");
  const llm = events.filter((e) => e.kind === "llm_call");
  const llmMs = llm.map((e) => e.body.ms as number).sort((a, b) => a - b);
  const sources: Record<string, string> = {};
  events.filter((e) => e.kind === "answer").forEach((e) => { sources[e.body.key] = how[e.body.source] || e.body.source; });
  const lines = useMemo(() => story(events), [events]);

  return (
    <>
      <p><button className="quiet" onClick={back}>Back to applications</button></p>
      <h2>{events.length ? when(events[0].at_ms) : "Application"}</h2>
      <p className={intact ? "chain ok" : "chain bad"}>
        {intact ? "Record intact: every entry still matches its fingerprint, so nothing was edited or removed." : "Record broken: some entries don't match, so this log may have been edited."}
      </p>

      {dec ? (
        <dl className="audit-facts">
          <div><dt>Result</dt><dd className={dec.decision === "approved" ? "good" : "bad"}>{dec.decision}</dd></div>
          <div><dt>Chance of decline</dt><dd>{pct(dec.probability)}{dec.threshold ? ` (cut-off ${pct(dec.threshold)})` : ""}</dd></div>
          {dec.seconds_to_complete !== undefined && <div><dt>Time to finish</dt><dd>{Math.round(dec.seconds_to_complete)} s</dd></div>}
          <div><dt>Model</dt><dd>{dec.model_version || ""}</dd></div>
        </dl>
      ) : <p>This application wasn't finished, so no decision was made.</p>}

      {app && (
        <>
          <h3>Answers</h3>
          <div className="audit-table">
            {Object.keys(labels).filter((k) => k in app.answers).map((k) => (
              <div key={k}><span>{labels[k]}</span><b>{shown(k, app.answers[k])}</b><em>{sources[k] || ""}</em></div>
            ))}
            {app.derived && <div><span>Loan amount</span><b>{rupees(app.derived.loan_amount)}</b><em>{app.derived.ltv}% of the price</em></div>}
          </div>
        </>
      )}

      {why && why.shown?.length > 0 && (
        <>
          <h3>Reasons shown to the applicant</h3>
          <ul className="audit-reasons">
            {[...why.shown, ...(why.against || [])].map((r: any, i: number) => <li key={i}><b>{r.label}</b> {r.text}</li>)}
          </ul>
        </>
      )}

      <h3>What happened</h3>
      <ol className="audit-story">
        {lines.map((l, i) => <li key={i} className={l.who}><time>{clock(l.t)}</time><span>{l.text}</span></li>)}
      </ol>
      {llm.length > 0 && (
        <p className="aside">
          {llm.length} language model calls, typically {(llmMs[Math.floor(llmMs.length / 2)] / 1000).toFixed(1)} s each.
          Pincodes are kept to 3 digits and phone numbers, PAN, Aadhaar and emails are removed before anything is stored.
        </p>
      )}

      <details className="raw">
        <summary>All {events.length} raw entries</summary>
        {events.map((e) => (
          <div key={e.seq} className="event">
            <header><b>{e.kind.replaceAll("_", " ")}</b><span>{new Date(e.at_ms).toLocaleTimeString("en-IN")}</span></header>
            <pre>{JSON.stringify(e.body, null, 1)}</pre>
          </div>
        ))}
      </details>
      <p className="aside">Session {id}</p>
    </>
  );
}

export default function Audit() {
  const [key, setKey] = useState("");
  const [sessions, setSessions] = useState<Session[] | null>(null);
  const [problem, setProblem] = useState("");
  const [everything, setEverything] = useState(false);
  const [open, setOpen] = useState<{ id: string; events: Event[]; intact: boolean } | null>(null);

  const load = useCallback(async (k: string, quiet = false) => {
    const res = await fetch("/api/audit/sessions", { headers: { "x-audit-key": k } });
    if (!res.ok) {
      if (!quiet) setProblem("That key wasn't accepted.");
      return;
    }
    const data = await res.json();
    setSessions(data.sessions);
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
      window.scrollTo(0, 0);
    }
  }

  const finished = sessions?.filter((s) => s.decision) || [];
  const list = everything ? sessions || [] : finished;

  return (
    <main className="page">
      <h1>Audit log</h1>
      {!open && <p>Every application, with personal details masked. Each entry is chained to the one before it, so a quiet edit shows up.</p>}

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
          <div className="audit-bar">
            <p>{everything ? `${sessions.length} applications, ${sessions.length - finished.length} not finished.` : `${finished.length} finished applications.`}</p>
            <button className="quiet" onClick={() => setEverything(!everything)}>
              {everything ? "Show finished only" : "Include unfinished"}
            </button>
          </div>
          {list.length === 0 && <p>Nothing here yet. Finish an application and it will show up.</p>}
          {list.length > 0 && (
            <div className="session-row head"><span>Started</span><span>Result</span><span>Chance of decline</span><span>Entries</span></div>
          )}
          {list.map((s) => (
            <button key={s.session_id} className="session-row" onClick={() => show(s.session_id)}>
              <span>{when(s.started)}</span>
              <span className={s.decision ? (s.decision.decision === "approved" ? "good" : "bad") : "muted"}>{s.decision ? s.decision.decision : "Not finished"}</span>
              <span>{s.decision ? pct(s.decision.probability) : ""}</span>
              <span>{s.events}</span>
            </button>
          ))}
        </>
      )}

      {open && <Detail {...open} back={() => setOpen(null)} />}
    </main>
  );
}
