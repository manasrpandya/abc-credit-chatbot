import type { Metadata } from "next";
import ThresholdChart from "@/components/ThresholdChart";
import QuestionChart from "@/components/QuestionChart";
import data from "@/lib/site_metrics.json";

export const metadata: Metadata = { title: "How it decides, ABC Credit" };

const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
const n = (v: number) => v.toLocaleString("en-IN");
// labels are stored in capitals by the cleaning step
const nice = (s: string) => {
  const t = s.replace(/^\d+\s*/, "").toLowerCase();
  return (t.charAt(0).toUpperCase() + t.slice(1)).replace(/\((a|b)\)/, (m) => m.toUpperCase());
};

const fixes = [
  ["One fully empty row at the end of the file", "Dropped. It's also where the odd blank-string age came from."],
  ["Placeholder incomes like ₹1, ₹2, ₹25 and ₹100", "Treated as missing, with a flag."],
  ["Zero income on students and non-earners", "Kept. It's real, not an error."],
  ["65 rows with an LTV of exactly 0", "Estimated from the loan and the model's usual price, with a flag."],
  ["1,105 applicants above the stated age limit of 60", "Kept and flagged, since they were clearly processed."],
  ["Broken characters in model names, and 125 CC DISC vs 125CC DISC", "Cleaned so the same thing is always spelled the same way."],
  ["Blank tier, make, variant and product cells", "Filled from the pincode or model, which fully determine them."],
  ["14,000 pincodes, most with under 20 rows", "Reduced to two and three digit postal areas."],
];

export default function HowItDecides() {
  const t = data.test.hosted;
  const local = data.test.local;
  const naive = data.test.always_approve;
  const groups = data.fairness as Record<string, Record<string, { n: number; actual_decline: number; flagged: number; recall: number }>>;

  return (
    <main className="page">
      <h1>A trained model decides. The language model only talks.</h1>
      <p>
        The approve or decline call is a probability from a gradient boosted tree model, compared against a fixed
        cut-off. The language model helps read messy messages and words the explanation, and every answer it
        suggests passes through the same checks as a typed one. It can&apos;t reach the decision.
      </p>

      <div className="big-numbers">
        <div><strong>{data.stability.mean_auc.toFixed(2)}</strong><span>typical AUC on rows the model never saw</span></div>
        <div><strong>{pct(t.recall, 0)}</strong><span>of past declines caught</span></div>
        <div><strong>{pct(t.precision, 0)}</strong><span>of our declines were declined in the past too</span></div>
        <div><strong>{pct(t.accuracy, 1)}</strong><span>accuracy at the chosen cut-off</span></div>
      </div>

      <h2>What these numbers mean</h2>
      <p>
        Only {pct(data.rows.decline_rate)} of past applications were declined, so approving everyone scores{" "}
        {pct(naive.accuracy, 2)} accuracy while catching none of them. <strong>Catching declines has to cost some
        accuracy</strong>, which is why we rank with AUC and average precision, and treat the cut-off as a business choice.
      </p>
      <p>
        The score tops out near {data.stability.mean_auc.toFixed(2)} across every model we tried. Past decisions likely
        depend on things this form doesn&apos;t capture, such as credit bureau data, so a much higher number here would
        have been a warning sign.
      </p>
      <p>
        <strong>Our locked test rows happened to be a kind split.</strong> Refitting on eight other random splits gave
        AUCs from {data.stability.min_auc.toFixed(3)} to {data.stability.max_auc.toFixed(3)}, so expect about{" "}
        {data.stability.mean_auc.toFixed(2)} on new data rather than the {t.auc.toFixed(3)} in the table below. With the
        labels shuffled, the same pipeline scores {data.stability.shuffled_labels[0][0].toFixed(2)} and{" "}
        {data.stability.shuffled_labels[1][0].toFixed(2)}, about what pure chance gives, which is good evidence that
        nothing leaks.
      </p>

      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>On the locked test rows ({n(data.rows.test)})</th><th className="num">AUC</th><th className="num">Avg. precision</th><th className="num">Accuracy</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th></tr>
          </thead>
          <tbody>
            <tr><td>Always approve</td><td className="num">0.500</td><td className="num">{data.rows.decline_rate.toFixed(3)}</td><td className="num">{pct(naive.accuracy)}</td><td className="num">0%</td><td className="num">0%</td><td className="num">0</td></tr>
            <tr className="pick"><td>Hosted model (LightGBM)</td><td className="num">{t.auc.toFixed(3)}</td><td className="num">{t.ap.toFixed(3)}</td><td className="num">{pct(t.accuracy)}</td><td className="num">{pct(t.precision)}</td><td className="num">{pct(t.recall)}</td><td className="num">{t.f1.toFixed(3)}</td></tr>
            <tr><td>Localhost model (LightGBM and CatBoost)</td><td className="num">{local.auc.toFixed(3)}</td><td className="num">{local.ap.toFixed(3)}</td><td className="num">{pct(local.accuracy)}</td><td className="num">{pct(local.precision)}</td><td className="num">{pct(local.recall)}</td><td className="num">{local.f1.toFixed(3)}</td></tr>
          </tbody>
        </table>
      </div>

      <h2>Why eight questions</h2>
      <p>
        We removed one question at a time and measured what the model lost. The down payment and pincode carry most of
        the signal. The vehicle answer fills four columns and the pincode fills two, so a 15 column form becomes a short chat.
      </p>
      <QuestionChart points={data.search.path} />
      <p>
        Six questions would score almost the same. We still ask about employment and income because the brief asks for
        them and a lender needs them to judge affordability, even though they barely change past-decision accuracy.
        Gender was left out on purpose. It would add only {(data.search.with_gender[0] - data.search.all_questions[0]).toFixed(4)} AUC.
      </p>

      <h2>Where the cut-off sits</h2>
      <p>
        We assumed approving a bad case costs about {data.cost_ratio} times as much as turning away a good one, then
        picked the cut-off with the lowest total cost on out-of-fold predictions. That gives {pct(data.threshold.hosted)}.
      </p>
      <ThresholdChart curve={data.threshold_curve} cut={data.threshold.hosted} />
      <div className="table-wrap">
        <table>
          <thead><tr><th>If a bad approval costs</th><th className="num">Cut-off</th><th className="num">Applicants declined</th><th className="num">Recall</th><th className="num">Precision</th><th className="num">Accuracy</th></tr></thead>
          <tbody>
            {data.threshold_sensitivity.map((r) => (
              <tr key={r.ratio} className={r.ratio === data.cost_ratio ? "pick" : ""}>
                <td>{r.ratio} times a lost customer</td><td className="num">{pct(r.threshold)}</td><td className="num">{pct(r.declined_share)}</td><td className="num">{pct(r.recall)}</td><td className="num">{pct(r.precision)}</td><td className="num">{pct(r.accuracy)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Problems we found in the data</h2>
      <div className="table-wrap">
        <table>
          <thead><tr><th>What we found</th><th>What we did</th></tr></thead>
          <tbody>{fixes.map(([a, b]) => <tr key={a}><td>{a}</td><td>{b}</td></tr>)}</tbody>
        </table>
      </div>

      <h2>Who it treats differently</h2>
      <p>
        These are error rates by group on the locked rows. Gender isn&apos;t an input, yet it&apos;s still worth checking.
        Location is an input, and past decisions differ by area, so the model repeats some of that gap.
      </p>
      {Object.entries(groups).filter(([k]) => k !== "employment").map(([name, rows]) => (
        <div className="table-wrap" key={name}>
          <table>
            <thead><tr><th>By {name}</th><th className="num">Rows</th><th className="num">Declined in the past</th><th className="num">Declined by the model</th><th className="num">Recall</th></tr></thead>
            <tbody>
              {Object.entries(rows).map(([k, r]) => (
                <tr key={k}><td>{nice(k)}</td><td className="num">{n(r.n)}</td><td className="num">{pct(r.actual_decline)}</td><td className="num">{pct(r.flagged)}</td><td className="num">{pct(r.recall)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}

      <h2>What this can&apos;t tell you</h2>
      <p>
        The labels record what the lender decided, not whether anybody repaid. So this model learns to repeat past
        decisions, including any bias in them, and it can&apos;t claim to predict default. The factors shown to an applicant
        are real contributions from the model, but they describe patterns, not causes.
      </p>
    </main>
  );
}
