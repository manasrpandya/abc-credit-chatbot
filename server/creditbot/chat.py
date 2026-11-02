"""The application flow. A plain state machine owns the order and the rules. The llm can help
read a messy message or answer a side question, but everything it returns goes through the
same validators as a typed answer, and it never touches the decision."""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import uuid

from . import audit, nim, parsing, privacy
from .scorer import load_scorer

order = ["vehicle", "on_road_price", "down_payment", "pincode", "age", "employment", "income", "active_loans"]

employment_options = [("Salaried", "SAL"), ("Self-employed", "SEP"), ("Farming", "AGR"),
                      ("Irregular income", "NREGI"), ("On probation", "NPP"), ("Student", "STU"),
                      ("Pensioner", "PEN"), ("Not earning", "NONEARNMEM")]
employment_labels = {code: label for label, code in employment_options}
no_income_needed = {"STU", "NONEARNMEM"}

income_wording = {
    "SAL": "What's your monthly take-home salary?",
    "SEP": "What does your business bring in on average each month?",
    "AGR": "Averaged across the year, what's your monthly income?",
    "PEN": "What's your monthly pension?",
    "NREGI": "In a typical month, how much do you earn?",
    "NPP": "What's your monthly pay at the moment?",
}

why_we_ask = {
    "vehicle": "The model sets the usual price and how similar loans have gone.",
    "on_road_price": "The price and down payment together decide how much of the vehicle gets financed.",
    "down_payment": "A bigger down payment means a smaller loan. It's the strongest single signal we use.",
    "pincode": "It tells us your market area. Only the first three digits are kept in our logs.",
    "age": "Applicants need to be at least 18. Age is also one of the signals in the model.",
    "employment": "It decides how we ask about income, and it's a signal in the model.",
    "income": "We compare your income with the size of the loan.",
    "active_loans": "An existing loan affects how past applications like yours turned out.",
}

factor_labels = {"financing": "Down payment and loan size", "price": "Vehicle price", "income": "Income",
                 "location": "Your area", "vehicle": "Vehicle model", "age": "Age",
                 "employment": "Employment type", "active_loans": "Existing loans",
                 "education": "Education", "residence": "Residence"}

toward_decline = {
    "financing": "Most of the vehicle price would be on loan ({ltv:.0f}%)",
    "price": "The price of this vehicle is on the higher side",
    "income": "The loan is large compared with your income",
    "location": "Applications from your area were declined more often",
    "vehicle": "This model was declined more often in past applications",
    "age": "Applicants around your age were declined more often",
    "employment": "Your employment type was declined more often",
    "active_loans": "Applicants with your loan history were declined more often",
}
toward_approval = {
    "financing": "A good share of the price is covered upfront ({ltv:.0f}% on loan)",
    "price": "The price of this vehicle is on the lower side",
    "income": "Your income is comfortable for this loan size",
    "location": "Applications from your area were approved more often",
    "vehicle": "This model was approved more often in past applications",
    "age": "Applicants around your age were approved more often",
    "employment": "Your employment type was approved more often",
    "active_loans": "Applicants with your loan history were approved more often",
}

disclaimer = ("The approve or decline result comes from a statistical model trained on past applications. "
              "The factors above are patterns found in those past decisions, not a statement about you and "
              "not the formal reason for this result. They don't show cause and effect, and they aren't "
              "financial advice. Any final loan is subject to document and identity checks.")

max_llm_calls = 12
max_text = 500


def _secret():
    s = os.environ.get("STATE_SECRET")
    if not s:
        s = secrets.token_hex(32)
        os.environ["STATE_SECRET"] = s
    return s.encode()


def seal(state):
    raw = base64.urlsafe_b64encode(json.dumps(state, separators=(",", ":")).encode()).decode()
    return raw + "." + hmac.new(_secret(), raw.encode(), hashlib.sha256).hexdigest()


def unseal(token):
    try:
        raw, sig = str(token).rsplit(".", 1)
    except ValueError:
        return None
    good = hmac.new(_secret(), raw.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, good):
        return None
    return json.loads(base64.urlsafe_b64decode(raw.encode()))


def rupees(n):
    n = int(round(n))
    s = str(n)
    if len(s) <= 3:
        return "₹" + s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return "₹" + ",".join(parts) + "," + tail


def catalogue():
    look = load_scorer_for_mode().look
    names = sorted(look["model_price"])
    return {n: parsing.clean_model_name(n) for n in names}


def families():
    look = load_scorer_for_mode().look
    out = {}
    for model, make in look["model_make"].items():
        out.setdefault(make, []).append(model)
    return out


def load_scorer_for_mode():
    return load_scorer(os.environ.get("CREDITBOT_MODEL", "hosted"))


family_names = {"JUPITER": "Jupiter", "APACHE": "Apache", "RAIDER": "Raider", "MOPEDS": "XL 100",
                "NTORQ": "Ntorq", "SPORT": "Sport", "RADEON": "Radeon", "ZEST": "Zest", "TVS": "iQube",
                "CITY PLUS": "Star City", "RONIN": "Ronin", "SCOOTY": "Pep Plus"}


def _family_size(family):
    sizes = load_scorer_for_mode().look["model_size"]
    return sum(sizes.get(m, 0) for m in families()[family])


def new_state():
    return {"sid": uuid.uuid4().hex[:16], "t0": int(time.time() * 1000), "a": {}, "pick": None,
            "llm": 0, "out": None, "review": False}


def is_needed(key, a):
    if key == "income":
        return a.get("employment") not in no_income_needed
    return True


def next_key(state):
    a = state["a"]
    for key in order:
        if is_needed(key, a) and key not in a:
            return key
    return None


def question_for(state, key):
    a = state["a"]
    q = {"key": key, "kind": "text", "chips": [], "placeholder": "", "why": why_we_ask[key]}
    if key == "vehicle":
        if state.get("pick"):
            cat = catalogue()
            q.update(text="Which one is closest?", kind="chips",
                     chips=[{"label": cat[m], "value": m} for m in state["pick"]])
            q["chips"].append({"label": "Something else", "value": "other"})
        else:
            q.update(text="Which two-wheeler are you buying?", placeholder="Type a model, like Jupiter 125",
                     chips=[{"label": family_names.get(f, f.title()), "value": "family:" + f}
                            for f in sorted(families(), key=lambda f: -_family_size(f))
                            if f in family_names and f != "SCOOTY"] + [{"label": "Something else", "value": "other"}])
    elif key == "on_road_price":
        typical = load_scorer_for_mode().look["model_price"].get(a.get("vehicle"))
        if typical:
            usual = int(round(typical / 500.0) * 500)
            q.update(text=f"This model usually costs about {rupees(usual)} on the road. Is that close to your quote?",
                     placeholder="Or type your quoted price",
                     chips=[{"label": f"Yes, about {rupees(usual)}", "value": str(usual)}])
        else:
            q.update(text="What's the on-road price you've been quoted?", placeholder="For example 95,000")
    elif key == "down_payment":
        price = a["on_road_price"]
        chips = [{"label": f"{rupees(round(price * s / 500) * 500)} ({int(s * 100)}%)",
                  "value": str(int(round(price * s / 500) * 500))} for s in (0.1, 0.2, 0.3)]
        q.update(text="How much can you pay upfront?", placeholder="An amount, or a percentage like 20%", chips=chips)
    elif key == "pincode":
        q.update(text="What's your pincode?", placeholder="6 digits")
    elif key == "age":
        q.update(text="How old are you?", placeholder="Your age in years")
    elif key == "employment":
        q.update(text="How do you earn a living?", kind="chips",
                 chips=[{"label": label, "value": code} for label, code in employment_options])
    elif key == "income":
        q.update(text=income_wording.get(a.get("employment"), "What's your monthly income?"),
                 placeholder="For example 35,000 or 35k")
    elif key == "active_loans":
        q.update(text="Do you have any loans running at the moment?", kind="chips",
                 chips=[{"label": "Yes", "value": "yes"}, {"label": "No", "value": "no"}])
    return q


def validate(key, value, a):
    """(clean value, error message). Used for typed answers and llm suggestions alike."""
    if key == "vehicle":
        if value == "other":
            return "", None
        if value in load_scorer_for_mode().look["model_price"]:
            return value, None
        return None, "I couldn't find that model."
    if key == "on_road_price":
        v = parsing.parse_amount(value)
        if v is None:
            return None, "I need the price as an amount, like 95,000."
        if not 30000 <= v <= 600000:
            return None, "That price looks off for a two-wheeler. It should be between ₹30,000 and ₹6,00,000."
        return int(v), None
    if key == "down_payment":
        price = a.get("on_road_price", 0)
        pct = parsing.parse_percent(value)
        v = price * pct / 100 if pct is not None else parsing.parse_amount(value)
        if v is None:
            return None, "I need an amount, like 20,000, or a percentage, like 20%."
        if v < 0 or v >= price:
            return None, f"The down payment has to be less than the price of {rupees(price)}."
        if (price - v) / price < 0.2:
            return None, "At that level you'd be financing under 20% of the price, which we don't offer. Try a smaller down payment."
        if price - v < 10000:
            return None, "The smallest loan we offer is ₹10,000."
        return int(v), None
    if key == "pincode":
        v = parsing.parse_pincode(value)
        if not v or v[0] in "09":
            return None, "A pincode here has 6 digits and starts with 1 to 8."
        return v, None
    if key == "age":
        v = parsing.parse_age(value)
        if v is None or not 1 <= v <= 100:
            return None, "I need your age as a number."
        return v, None
    if key == "employment":
        v = parsing.parse_employment(value)
        if not v:
            return None, "Please pick the option closest to how you earn."
        return v, None
    if key == "income":
        v = parsing.parse_amount(value)
        if v is None:
            return None, "I need an amount per month, like 35,000."
        if 0 < v < 1000:
            return None, "Please give your monthly income in rupees, ₹1,000 or more. If you don't earn, enter 0."
        if v > 1000000:
            return None, "That's above what we can take here. Please check it's a monthly figure."
        return int(v), None
    if key == "active_loans":
        v = value if isinstance(value, bool) else parsing.parse_yes_no(value)
        if v is None:
            return None, "A simple yes or no works here."
        return bool(v), None
    return None, "Unknown question."


def set_answer(state, key, value):
    a = state["a"]
    old = a.get(key)
    a[key] = value
    if key == "vehicle" and old != value:
        a.pop("on_road_price", None)
        a.pop("down_payment", None)
    if key == "on_road_price" and old != value:
        a.pop("down_payment", None)
    if key == "employment":
        if value in no_income_needed:
            a["income"] = 0
        elif old in no_income_needed:
            a.pop("income", None)
    state["pick"] = None
    state["out"] = None


def _ack(key, value, a):
    if key == "vehicle":
        return parsing.clean_model_name(value) if value else "No problem, we'll work without a catalogue model."
    if key == "down_payment":
        price = a["on_road_price"]
        return f"That leaves a loan of {rupees(price - value)}."
    return None


def to_application(a):
    price, down = a["on_road_price"], a["down_payment"]
    loan = price - down
    return {"Model_Description": a.get("vehicle") or "", "Loan_Amount": loan,
            "LTV": round(loan / price * 100, 2), "Pincode": a["pincode"], "Age": a["age"],
            "Employment_Type": a["employment"], "Net_salary": a.get("income", 0),
            "PAST_LOANS_ACTIVE": "PAST_LOANS_ACTIVE" if a["active_loans"] else "NO_PAST_LOANS"}


def summary_rows(a):
    out = []
    cat = catalogue()
    labels = [("vehicle", "Vehicle"), ("on_road_price", "On-road price"), ("down_payment", "Down payment"),
              ("pincode", "Pincode"), ("age", "Age"), ("employment", "Employment"),
              ("income", "Monthly income"), ("active_loans", "Loans running")]
    for key, label in labels:
        if key not in a or (key == "income" and not is_needed("income", a)):
            continue
        v = a[key]
        if key == "vehicle":
            shown = cat.get(v, "Not listed")
        elif key in ("on_road_price", "down_payment", "income"):
            shown = rupees(v)
        elif key == "employment":
            shown = employment_labels.get(v, v)
        elif key == "active_loans":
            shown = "Yes" if v else "No"
        else:
            shown = str(v)
        out.append({"key": key, "label": label, "value": shown})
    if "on_road_price" in a and "down_payment" in a:
        out.insert(3, {"key": "down_payment", "label": "Loan amount", "derived": True,
                       "value": rupees(a["on_road_price"] - a["down_payment"])})
    return out


def _masked_answers(a):
    return {k: privacy.mask_answer(k, v) for k, v in a.items()}


def _reply(state, messages, events):
    key = next_key(state)
    payload = {"messages": messages, "answers": summary_rows(state["a"]),
               "price": state["a"].get("on_road_price"), "down": state["a"].get("down_payment"),
               "progress": {"done": sum(1 for k in order if k in state["a"] and is_needed(k, state["a"])),
                            "total": sum(1 for k in order if is_needed(k, state["a"]))}}
    if state.get("ineligible"):
        payload.update(question=None, stage="ineligible")
    elif key is None:
        payload.update(question=None, stage="review")
    else:
        q = question_for(state, key)
        payload.update(question=q, stage="asking")
        events.append(("question_asked", {"key": key, "text": q["text"]}))
    audit.write_many(state["sid"], events)
    payload["state"] = seal(state)
    return payload


def start():
    state = new_state()
    events = [("session_started", {"consent": True, "model_version": load_scorer_for_mode().version,
                                   "model_kind": load_scorer_for_mode().kind, "llm_available": nim.available(),
                                   "log_store": audit.backend_name()})]
    return _reply(state, ["Hi, I'll help you apply for a two-wheeler loan. A few short questions, and you'll have an answer straight away."], events)


def _accept(state, key, value, source, raw, events):
    """Returns an error message or None. Handles the age gate too."""
    clean, err = validate(key, value, state["a"])
    if err:
        events.append(("answer_rejected", {"key": key, "text": privacy.mask_text(str(raw)),
                                           "source": source, "reason": err}))
        return err
    set_answer(state, key, clean)
    events.append(("answer", {"key": key, "text": privacy.mask_text(str(raw)),
                              "value": privacy.mask_answer(key, clean), "source": source}))
    if key == "age":
        if clean < 18:
            state["ineligible"] = True
            events.append(("policy_stop", {"rule": "under 18"}))
        elif clean > 60:
            events.append(("policy_flag", {"rule": "age above the stated 18 to 60 policy, scored anyway"}))
    return None


def _llm_help(state, key, text, events):
    """Ask nim to read a message the plain parsers couldn't. Returns (applied keys, reply text, errors)."""
    if not nim.available() or state["llm"] >= max_llm_calls:
        return [], None, []
    masked = privacy.mask_for_llm(text)
    state["llm"] += 1
    start_time = time.perf_counter()
    data = nim.extract_answers(masked, question_for(state, key)["text"])
    events.append(("llm_call", {"purpose": "read_message", "ok": data is not None,
                                "ms": round((time.perf_counter() - start_time) * 1000)}))
    if not data:
        return [], None, []

    applied, errors = [], []
    suggestion = data.pop("question", None)
    # go in question order, the down payment check needs the price first
    for k in order:
        if k not in data or data[k] in (None, ""):
            continue
        v = data[k]
        if k == "vehicle":
            hits = parsing.match_vehicle(v, catalogue())
            if len(hits) == 1:
                v = hits[0]
            else:
                if hits and k == key:
                    state["pick"] = hits
                continue
        err = _accept(state, k, v, "llm", text, events)
        if err:
            errors.append(err)
        else:
            applied.append(k)
    answer = None
    if suggestion and not applied and state["llm"] < max_llm_calls:
        state["llm"] += 1
        answer = nim.answer_question(privacy.mask_for_llm(str(suggestion)), question_for(state, key)["text"])
        events.append(("llm_call", {"purpose": "side_question", "ok": answer is not None}))
    return applied, answer, errors


question_starts = {"why", "what", "how", "who", "when", "where", "which", "should", "is", "do", "does",
                   "will", "can", "could", "are"}


def answer(token, text=None, value=None):
    state = unseal(token)
    if state is None:
        return None
    events = []
    key = next_key(state)
    messages = []
    if key is None:
        return _reply(state, ["That's everything. Check your details below."], events)

    raw = (value if value is not None else (text or "")).strip()[:max_text]
    if not raw:
        return _reply(state, ["I didn't catch that."], events)

    # chips and direct picks skip all parsing
    if key == "vehicle" and raw.startswith("family:"):
        fam = raw.split(":", 1)[1]
        models = families().get(fam, [])
        sizes = load_scorer_for_mode().look["model_size"]
        state["pick"] = sorted(models, key=lambda m: -sizes.get(m, 0))[:8]
        events.append(("answer_partial", {"key": key, "family": fam}))
        return _reply(state, [], events)

    if key == "vehicle" and value is None and raw.lower() not in ("other", "something else"):
        hits = parsing.match_vehicle(raw, catalogue())
        if len(hits) == 1:
            raw = hits[0]
        elif hits and len(raw.split()) <= 4:
            state["pick"] = hits
            events.append(("answer_partial", {"key": key, "text": privacy.mask_text(raw)}))
            return _reply(state, [], events)

    source = "button" if value is not None else "typed"
    if key == "vehicle" and raw.lower() in ("other", "something else"):
        raw = "other"

    words = raw.lower().split()
    has_digit = any(c.isdigit() for c in raw)
    is_question = value is None and ("?" in raw or (words[0] in question_starts and not has_digit))
    long_message = value is None and len(words) >= 6
    gone_to_llm = is_question or long_message

    if not gone_to_llm:
        err = _accept(state, key, raw, source, raw, events)
        if err is None:
            return _reply(state, _after_answer(state, [key]), events)

    applied, reply, errors = _llm_help(state, key, raw, events)
    if applied:
        return _reply(state, _after_answer(state, applied), events)
    if reply:
        return _reply(state, [reply], events)
    if is_question:
        return _reply(state, [why_we_ask[key]], events)
    if errors:
        return _reply(state, [errors[0]], events)
    if gone_to_llm:
        # no llm to hand, or it found nothing: a long plain answer can still be read
        err = _accept(state, key, raw, source, raw, events)
        if err is None:
            return _reply(state, _after_answer(state, [key]), events)
    return _reply(state, [err], events)


def _after_answer(state, keys):
    if state.get("ineligible"):
        return ["You need to be at least 18 to apply, so we can't take this further today. Thanks for stopping by."]
    if len(keys) > 1:
        return [f"Got it, I've noted {len(keys)} of those."]
    ack = _ack(keys[0], state["a"][keys[0]], state["a"])
    return [ack] if ack else []


def edit(token, key):
    state = unseal(token)
    if state is None or key not in order:
        return None
    state["a"].pop(key, None)
    if key == "vehicle":
        state["a"].pop("on_road_price", None)
        state["a"].pop("down_payment", None)
    if key == "on_road_price":
        state["a"].pop("down_payment", None)
    state["out"] = None
    return _reply(state, [], [("answer_cleared", {"key": key})])


def _reasons(ranked, decision, ltv):
    wanted = toward_decline if decision == "declined" else toward_approval
    sign = 1 if decision == "declined" else -1
    main = [r for r in ranked if r["effect"] * sign > 0 and r["factor"] in wanted][:3]
    other = wanted is toward_decline and toward_approval or toward_decline
    against = [r for r in ranked if r["effect"] * sign < 0 and r["factor"] in other][:1]
    return ([{"factor": r["factor"], "label": factor_labels[r["factor"]],
              "text": wanted[r["factor"]].format(ltv=ltv)} for r in main],
            [{"factor": r["factor"], "label": factor_labels[r["factor"]],
              "text": other[r["factor"]].format(ltv=ltv)} for r in against])


next_steps = {
    "approved": ["Keep your ID, address proof and recent income proof ready.",
                 "Take this result to your ABC dealer to finish the paperwork.",
                 "The final loan is confirmed after those checks."],
    "declined": ["You can apply again with a different down payment or vehicle.",
                 "You can ask for a callback to talk it through with our team."],
}


def _facts(state):
    scorer = load_scorer_for_mode()
    application = to_application(state["a"])
    result = scorer.decide(application)
    return scorer, application, result


def submit(token):
    state = unseal(token)
    if state is None or next_key(state) is not None or state.get("ineligible"):
        return None
    began = time.perf_counter()
    scorer, application, result = _facts(state)
    decision_ms = (time.perf_counter() - began) * 1000

    t = time.perf_counter()
    ranked = scorer.explain(application)
    shap_ms = (time.perf_counter() - t) * 1000
    reasons, against = _reasons(ranked, result["decision"], application["LTV"])

    t = time.perf_counter()
    what_if = None
    if result["decision"] == "declined":
        what_if = scorer.what_if_down_payment(application, state["a"]["on_road_price"])
    what_if_ms = (time.perf_counter() - t) * 1000

    rows = {k: [v] for k, v in application.items()}
    _, x = scorer.prepare(rows)
    feature_values = {n: (None if v != v else round(float(v), 4))
                      for n, v in zip(scorer.meta["feature_names"], x[0].tolist())}

    elapsed = int(time.time() * 1000) - state["t0"]
    t = time.perf_counter()
    audit.write_many(state["sid"], [
        ("application", {"answers": _masked_answers(state["a"]),
                         "derived": {"loan_amount": application["Loan_Amount"], "ltv": application["LTV"]}}),
        ("features", feature_values),
        ("decision", {**result, "seconds_to_complete": round(elapsed / 1000, 1)}),
        ("reasons", {"shap_by_answer": ranked, "shown": reasons, "against": against, "what_if": what_if}),
    ])
    log_ms = (time.perf_counter() - t) * 1000

    state["out"] = result["decision"]
    return {"state": seal(state), "decision": result["decision"], "probability": result["probability"],
            "threshold": result["threshold"], "model_kind": result["model_kind"],
            "model_version": result["model_version"], "reasons": reasons, "against": against,
            "shap": [{**r, "label": factor_labels.get(r["factor"], r["factor"])} for r in ranked],
            "what_if": what_if, "next_steps": next_steps[result["decision"]], "disclaimer": disclaimer,
            "timing": {"decision_ms": round(decision_ms, 1), "shap_ms": round(shap_ms, 1),
                       "what_if_ms": round(what_if_ms, 1), "log_ms": round(log_ms, 1),
                       "total_ms": round((time.perf_counter() - began) * 1000, 1),
                       "seconds_to_complete": round(elapsed / 1000, 1)},
            "llm_available": nim.available()}


banned = re.compile(r"\b(guarantee|definitely|promise|assured|score|model|shap|probability)\b", re.I)


def plain_summary(decision, what_if):
    head = "Your application was approved." if decision == "approved" else "Your application wasn't approved this time."
    tail = ""
    if what_if:
        tail = f" A down payment about {rupees(what_if['extra_down_payment'])} higher would likely change this."
    return head + tail


def llm_summary_ok(text, decision, what_if):
    """The llm may only rephrase. Drop it if it contradicts the decision or invents a number."""
    low = text.lower()
    if decision == "approved" and ("declin" in low or "not approved" in low or "wasn't approved" in low):
        return False
    if decision == "declined" and re.search(r"\b(congratulations|you('| a)re approved|has been approved|was approved)\b", low) \
            and "wasn't approved" not in low:
        return False
    if banned.search(text):
        return False
    allowed = set()
    if what_if:
        allowed.add(str(what_if["extra_down_payment"]))
    for number in re.findall(r"\d[\d,]*", text):
        if number.replace(",", "") not in allowed:
            return False
    return True


def summary(token):
    state = unseal(token)
    if state is None or next_key(state) is not None:
        return None
    scorer, application, result = _facts(state)
    ranked = scorer.explain(application)
    reasons, _ = _reasons(ranked, result["decision"], application["LTV"])
    what_if = scorer.what_if_down_payment(application, state["a"]["on_road_price"]) if result["decision"] == "declined" else None
    fallback = plain_summary(result["decision"], what_if)
    decision_word = "approved" if result["decision"] == "approved" else "not approved"
    facts = {"decision": decision_word, "main_factors": [r["text"] for r in reasons]}
    if what_if:
        facts["extra_down_payment_rupees"] = what_if["extra_down_payment"]
    started = time.perf_counter()
    text = nim.explain(facts)
    used = bool(text and llm_summary_ok(text, result["decision"], what_if))
    audit.write_many(state["sid"], [("llm_call", {"purpose": "explain", "ok": bool(text), "passed_check": used,
                                                  "ms": round((time.perf_counter() - started) * 1000),
                                                  "text": text if used else None})])
    return {"text": text if used else fallback, "from_llm": used}


def callback(token, phone):
    state = unseal(token)
    if state is None:
        return None
    digits = re.sub(r"\D", "", str(phone))
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    if not re.fullmatch(r"[6-9]\d{9}", digits):
        return {"ok": False, "message": "Please enter a 10 digit mobile number."}
    audit.write_many(state["sid"], [("callback_requested", {"phone": privacy.mask_phone(digits),
                                                            "phone_token": privacy.token(digits)})])
    return {"ok": True, "message": f"Noted. This demo keeps only a masked copy: {privacy.mask_phone(digits)}."}


def simulate(token, down):
    """Re-score with another down payment. Nothing is saved, the result page slider uses this."""
    state = unseal(token)
    if state is None or next_key(state) is not None:
        return None
    a = dict(state["a"])
    clean, err = validate("down_payment", str(down), a)
    if err:
        return {"decision": state.get("out") or "declined", "probability": 0, "error": err}
    a["down_payment"] = clean
    scorer = load_scorer_for_mode()
    p = scorer.probability(to_application(a))
    return {"decision": "declined" if p >= scorer.threshold else "approved", "probability": round(p, 5)}
