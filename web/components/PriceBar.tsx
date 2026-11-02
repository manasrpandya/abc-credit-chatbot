import { rupees } from "@/lib/api";

export default function PriceBar({ price, down, opening }: { price: number; down: number | null; opening?: boolean }) {
  const paid = down ?? 0;
  const loan = price - paid;
  const share = down === null ? 100 : (loan / price) * 100;
  return (
    <div className="bar-block">
      <div className="bar-numbers">
        <div>
          <strong>{down === null ? "?" : rupees(paid)}</strong>
          your money
        </div>
        <div>
          <strong>{down === null ? rupees(price) : rupees(loan)}</strong>
          {down === null ? "on-road price" : `on loan, ${Math.round(share)}% of the price`}
        </div>
      </div>
      <div className={opening ? "bar intro-bar" : "bar"} role="img"
           aria-label={down === null ? `On-road price ${rupees(price)}` : `${rupees(paid)} paid upfront, ${rupees(loan)} on loan`}>
        <div className="bar-loan" style={{ width: `${share}%` }} />
      </div>
    </div>
  );
}
