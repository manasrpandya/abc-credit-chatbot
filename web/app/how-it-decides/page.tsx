import type { Metadata } from "next";
import Link from "next/link";
import data from "@/lib/site_metrics.json";

export const metadata: Metadata = { title: "How it decides, ABC Credit" };

const pct = (v: number) => `${Math.round(v * 100)}%`;

const matters = [
  ["Your down payment", "The bigger the share you pay upfront, the smaller the loan. This counts the most."],
  ["Where you live", "Past applications from some areas were declined more often than others."],
  ["The vehicle and its price", "Some models and price points were declined more often in the past."],
  ["Age, work, income and other loans", "These count too, but less than you might expect."],
];

export default function HowItDecides() {
  const t = data.test.hosted;

  return (
    <main className="page short">
      <h1>The chat collects your answers. A trained model makes the call.</h1>
      <p>
        The assistant you talk to only asks questions and explains things. The approve or decline call comes from a
        model trained on {data.rows.total.toLocaleString("en-IN")} past applications. It gives a risk score, and anything
        above a fixed cut-off is declined. The same answers always get the same result, and the assistant can&apos;t change it.
      </p>

      <h2>What matters most</h2>
      <dl className="matters">
        {matters.map(([a, b]) => <div key={a}><dt>{a}</dt><dd>{b}</dd></div>)}
      </dl>
      <p>
        Gender isn&apos;t asked and isn&apos;t used. After a decision you&apos;ll see the main factors behind it, and a slider
        that shows how a bigger down payment would change the result.
      </p>

      <h2>How far to trust it</h2>
      <p>
        It catches about {pct(t.recall)} of the applications that were declined in the past, and it isn&apos;t perfect:
        some people it declines were approved before. Treat the result as a quick first answer. Any real loan still goes
        through document and identity checks.
      </p>

      <h2>Your data</h2>
      <p>
        We keep a masked record of each application so every decision can be checked later. Your pincode is cut to
        three digits. A phone number, PAN, Aadhaar or email typed into the chat is removed before anything is stored or
        sent to the language model.
      </p>

      <p className="aside">The full method, numbers and data checks are in the project report and README.</p>
      <p><Link className="primary-link" href="/">Start an application</Link></p>
    </main>
  );
}
