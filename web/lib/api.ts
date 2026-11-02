export async function post<T = any>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api/chat/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const info = await res.json().catch(() => null);
    throw new Error(info?.detail || "The server didn't respond. Please try again.");
  }
  return res.json();
}

export function rupees(n: number) {
  return "₹" + Math.round(n).toLocaleString("en-IN");
}

export type Chip = { label: string; value: string };

export type Question = {
  key: string;
  kind: string;
  text: string;
  chips: Chip[];
  placeholder: string;
  why: string;
};

export type AnswerRow = { key: string; label: string; value: string; derived?: boolean };

export type Turn = {
  state: string;
  messages: string[];
  answers: AnswerRow[];
  price: number | null;
  down: number | null;
  progress: { done: number; total: number };
  question: Question | null;
  stage: "asking" | "review" | "ineligible";
};

export type Reason = { factor: string; label: string; text: string };

export type Result = {
  state: string;
  decision: "approved" | "declined";
  probability: number;
  threshold: number;
  model_kind: string;
  model_version: string;
  reasons: Reason[];
  against: Reason[];
  shap: { factor: string; label: string; effect: number }[];
  what_if: { extra_down_payment: number; new_loan: number; new_ltv: number } | null;
  next_steps: string[];
  disclaimer: string;
  timing: Record<string, number>;
  llm_available: boolean;
};
