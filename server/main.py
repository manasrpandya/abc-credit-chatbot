import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from creditbot import audit, chat, nim

app = FastAPI(title="abc credit chatbot", docs_url=None, redoc_url=None, openapi_url=None)


class Turn(BaseModel):
    state: str
    text: str | None = None
    value: str | None = None


class Edit(BaseModel):
    state: str
    key: str


class Sealed(BaseModel):
    state: str


class Simulate(BaseModel):
    state: str
    down: int


class Callback(BaseModel):
    state: str
    phone: str


def _or_gone(out):
    if out is None:
        raise HTTPException(400, "This session can't continue. Please start again.")
    return out


@app.get("/api/health")
def health():
    scorer = chat.load_scorer_for_mode()
    return {"ok": True, "model": scorer.kind, "version": scorer.version, "threshold": scorer.threshold,
            "llm": nim.available(), "log_store": audit.backend_name()}


@app.post("/api/chat/start")
def start():
    return chat.start()


@app.post("/api/chat/answer")
def answer(turn: Turn):
    return _or_gone(chat.answer(turn.state, turn.text, turn.value))


@app.post("/api/chat/edit")
def edit(e: Edit):
    return _or_gone(chat.edit(e.state, e.key))


@app.post("/api/chat/submit")
def submit(s: Sealed):
    return _or_gone(chat.submit(s.state))


@app.post("/api/chat/summary")
def summary(s: Sealed):
    return _or_gone(chat.summary(s.state))


@app.post("/api/chat/simulate")
def simulate(s: Simulate):
    return _or_gone(chat.simulate(s.state, s.down))


@app.post("/api/chat/callback")
def callback(c: Callback):
    return _or_gone(chat.callback(c.state, c.phone))


def _check_key(key):
    wanted = os.environ.get("AUDIT_KEY", "")
    if os.environ.get("CREDITBOT_LOCAL") == "1" and not wanted:
        return
    if not wanted or key != wanted:
        raise HTTPException(401, "Wrong or missing audit key.")


@app.get("/api/audit/sessions")
def audit_sessions(x_audit_key: str | None = Header(default=None)):
    _check_key(x_audit_key)
    return {"sessions": audit.recent_sessions(), "store": audit.backend_name()}


@app.get("/api/audit/sessions/{session_id}")
def audit_session(session_id: str, x_audit_key: str | None = Header(default=None)):
    _check_key(x_audit_key)
    return audit.session_events(session_id)
