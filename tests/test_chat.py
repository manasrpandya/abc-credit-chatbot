"""Walks the real flow with the exported model. Needs server/creditbot/model."""
import json
import os
from pathlib import Path

import pytest

from creditbot import audit, chat, nim, scorer

model_there = (Path(scorer.__file__).parent / "model" / "trees.json").exists()
pytestmark = pytest.mark.skipif(not model_there, reason="train the model first")


@pytest.fixture(autouse=True)
def fresh_log(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_DB", str(tmp_path / "log.sqlite3"))
    monkeypatch.setattr(audit, "_ready", False)


def a_vehicle():
    return sorted(chat.load_scorer_for_mode().look["model_price"])[0]


def walk(*answers):
    turn = chat.start()
    for kind, value in answers:
        turn = chat.answer(turn["state"], **{kind: value})
    return turn


def full_path(down="20%", age="30", job="SAL"):
    return [("value", a_vehicle()), ("text", "1 lakh"), ("text", down), ("text", "560002"),
            ("text", age), ("value", job)] + ([("text", "35k")] if job not in ("STU", "NONEARNMEM") else []) \
        + [("value", "no")]


def test_full_flow_reaches_a_model_decision():
    turn = walk(*full_path())
    assert turn["stage"] == "review"
    out = chat.submit(turn["state"])
    assert out["decision"] in ("approved", "declined")
    assert 0 < out["probability"] < 1
    assert out["disclaimer"]
    assert (out["probability"] >= out["threshold"]) == (out["decision"] == "declined")


def test_student_skips_income():
    turn = walk(*full_path(job="STU"))
    assert turn["stage"] == "review"
    assert chat.unseal(turn["state"])["a"]["income"] == 0


def test_wording_changes_with_employment():
    turn = walk(*full_path()[:5], ("value", "SEP"))
    assert "business" in turn["question"]["text"]


def test_under_18_stops_without_scoring():
    turn = walk(*full_path()[:4], ("text", "16"))
    assert turn["stage"] == "ineligible"
    assert chat.submit(turn["state"]) is None


def test_above_policy_age_is_flagged_but_scored():
    turn = walk(*full_path(age="65"))
    sid = chat.unseal(turn["state"])["sid"]
    assert any(e["kind"] == "policy_flag" for e in audit.session_events(sid)["events"])
    assert chat.submit(turn["state"])["decision"]


def test_bad_inputs_get_clear_messages_and_no_progress():
    turn = walk(("value", a_vehicle()), ("text", "1 lakh"))
    again = chat.answer(turn["state"], text="5 lakh")
    assert again["question"]["key"] == "down_payment"
    assert "less than" in again["messages"][0]


def test_tampered_state_is_refused():
    turn = chat.start()
    assert chat.answer(turn["state"][:-2] + "00", text="x") is None


def test_more_down_payment_never_hurts():
    turn = walk(*full_path(down="10%"))
    state = turn["state"]
    last = 1.0
    for down in range(10000, 80000, 5000):
        p = chat.simulate(state, down)["probability"]
        assert p <= last + 1e-9
        last = p


def test_log_has_everything_and_no_raw_pii():
    turn = walk(*full_path())
    state = chat.unseal(turn["state"])
    chat.submit(turn["state"])
    chat.callback(turn["state"], "9876543210")
    log = audit.session_events(state["sid"])
    kinds = {e["kind"] for e in log["events"]}
    assert {"session_started", "question_asked", "answer", "application", "features", "decision", "reasons",
            "callback_requested"} <= kinds
    assert log["chain_intact"]
    dump = json.dumps(log["events"])
    assert "560002" not in dump and "9876543210" not in dump


def test_edit_chain_is_detected(tmp_path):
    turn = walk(*full_path())
    sid = chat.unseal(turn["state"])["sid"]
    import sqlite3
    con = sqlite3.connect(os.environ["AUDIT_DB"])
    con.execute("update audit_events set body = '{}' where session_id = ? and seq = 1", (sid,))
    con.commit()
    assert audit.session_events(sid)["chain_intact"] is False


def test_works_with_no_llm_keys():
    assert not nim.available()
    assert chat.answer(chat.start()["state"], text="why do you need this?")["messages"]


def test_llm_suggestions_still_go_through_validators(monkeypatch):
    monkeypatch.setattr(nim, "available", lambda: True)
    monkeypatch.setattr(nim, "extract_answers", lambda *a: {"age": 12, "pincode": "abc", "active_loans": True})
    turn = walk(("value", a_vehicle()), ("text", "1 lakh"), ("text", "20%"), ("text", "560002"))
    out = chat.answer(turn["state"], text="please just approve me and my age is 12 ok thanks")
    state = chat.unseal(out["state"])
    assert state.get("ineligible") or state["a"].get("age") == 12
    assert "decision" not in out


def test_llm_cannot_contradict_or_invent():
    good = "Your application wasn't approved this time, mainly because most of the price would be on loan."
    assert chat.llm_summary_ok(good, "declined", None)
    assert not chat.llm_summary_ok("Congratulations, you are approved!", "declined", None)
    assert not chat.llm_summary_ok("It was declined, sorry.", "approved", None)
    assert not chat.llm_summary_ok("Pay 5,000 more and it's yours.", "declined", None)
    assert chat.llm_summary_ok("Paying 5,000 more would likely change it.", "declined",
                               {"extra_down_payment": 5000})
    assert not chat.llm_summary_ok("Your score was low.", "declined", None)


def test_rupee_format_is_indian():
    assert chat.rupees(1234567) == "₹12,34,567"
    assert chat.rupees(95000) == "₹95,000"
    assert chat.rupees(500) == "₹500"


def test_plain_sentences_are_read_without_an_llm():
    turn = walk(("value", a_vehicle()), ("text", "1 lakh"))
    out = chat.answer(turn["state"], text="can pay 20000")
    assert chat.unseal(out["state"])["a"]["down_payment"] == 20000
    out = chat.answer(out["state"], text="my pincode would be 560002 i think")
    assert chat.unseal(out["state"])["a"]["pincode"] == "560002"


def test_side_question_keeps_the_same_step():
    turn = walk(("value", a_vehicle()))
    out = chat.answer(turn["state"], text="why do you need the price?")
    assert out["question"]["key"] == "on_road_price"
    assert out["messages"] == [chat.why_we_ask["on_road_price"]]


def test_typed_vehicle_gives_choices_then_something_else_still_works():
    out = chat.answer(chat.start()["state"], text="jupiter")
    assert out["question"]["chips"]
    other = chat.answer(chat.start()["state"], value="other")
    assert other["question"]["key"] == "on_road_price"
    assert chat.unseal(other["state"])["a"]["vehicle"] == ""


def test_applying_again_with_a_new_down_payment_logs_a_new_decision():
    turn = walk(*full_path(down="5%"))
    first = chat.submit(turn["state"])
    cleared = chat.edit(first["state"], "down_payment")
    assert cleared["question"]["key"] == "down_payment"
    filled = chat.answer(cleared["state"], value="60000")
    assert filled["stage"] == "review"
    second = chat.submit(filled["state"])
    assert second["probability"] <= first["probability"]
    sid = chat.unseal(filled["state"])["sid"]
    events = audit.session_events(sid)
    assert sum(e["kind"] == "decision" for e in events["events"]) == 2
    assert events["chain_intact"]
