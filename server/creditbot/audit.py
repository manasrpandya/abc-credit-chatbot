"""Append only log. Postgres when DATABASE_URL is set (hosted), sqlite otherwise (localhost).
Each row carries a hash of the previous one, so a quiet edit breaks the chain."""
import hashlib
import json
import os
import sqlite3
import ssl
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

_lock = threading.Lock()
_ready = False


def _db_url():
    return os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL") or ""


def _pg():
    import pg8000.native
    u = urlparse(_db_url())
    return pg8000.native.Connection(user=u.username, password=u.password, host=u.hostname,
                                    port=u.port or 5432, database=u.path.lstrip("/"),
                                    ssl_context=ssl.create_default_context(), timeout=6)


def _sqlite_path():
    return Path(os.environ.get("AUDIT_DB", Path.cwd() / "audit_log.sqlite3"))


def _sqlite():
    con = sqlite3.connect(_sqlite_path(), timeout=5)
    con.row_factory = sqlite3.Row
    return con


create_sql = """create table if not exists audit_events (
    id text primary key, session_id text not null, seq integer not null, at_ms bigint not null,
    kind text not null, body text not null, prev_hash text not null, row_hash text not null)"""


def _ensure():
    global _ready
    if _ready:
        return
    if _db_url():
        con = _pg()
        con.run(create_sql)
        con.run("create index if not exists audit_session on audit_events (session_id, seq)")
        con.close()
    else:
        with _sqlite() as con:
            con.execute(create_sql)
            con.execute("create index if not exists audit_session on audit_events (session_id, seq)")
    _ready = True


def _hash(prev, session_id, seq, at_ms, kind, body):
    raw = "|".join([prev, session_id, str(seq), str(at_ms), kind, body])
    return hashlib.sha256(raw.encode()).hexdigest()


def write_many(session_id, events):
    """events: list of (kind, body). A logging failure never breaks the chat but does get printed."""
    if not events:
        return True
    try:
        _ensure()
        with _lock:
            if _db_url():
                con = _pg()
                last = con.run("select seq, row_hash from audit_events where session_id = :s "
                               "order by seq desc limit 1", s=session_id)
                seq = last[0][0] if last else -1
                prev = last[0][1] if last else "start"
                for kind, body in events:
                    seq += 1
                    text = json.dumps(body, ensure_ascii=False, default=str, sort_keys=True)
                    at_ms = int(time.time() * 1000)
                    row = _hash(prev, session_id, seq, at_ms, kind, text)
                    con.run("insert into audit_events values (:i, :s, :q, :a, :k, :b, :p, :h)",
                            i=uuid.uuid4().hex, s=session_id, q=seq, a=at_ms, k=kind, b=text, p=prev, h=row)
                    prev = row
                con.close()
            else:
                with _sqlite() as con:
                    last = con.execute("select seq, row_hash from audit_events where session_id = ? "
                                       "order by seq desc limit 1", (session_id,)).fetchone()
                    seq = last["seq"] if last else -1
                    prev = last["row_hash"] if last else "start"
                    for kind, body in events:
                        seq += 1
                        text = json.dumps(body, ensure_ascii=False, default=str, sort_keys=True)
                        at_ms = int(time.time() * 1000)
                        row = _hash(prev, session_id, seq, at_ms, kind, text)
                        con.execute("insert into audit_events values (?,?,?,?,?,?,?,?)",
                                    (uuid.uuid4().hex, session_id, seq, at_ms, kind, text, prev, row))
                        prev = row
        return True
    except Exception as err:
        print("audit write failed:", type(err).__name__, flush=True)
        return False


def _rows(sql_pg, sql_lite, **params):
    _ensure()
    if _db_url():
        con = _pg()
        rows = con.run(sql_pg, **params)
        names = [c["name"] for c in con.columns]
        con.close()
        return [dict(zip(names, r)) for r in rows]
    with _sqlite() as con:
        return [dict(r) for r in con.execute(sql_lite, tuple(params.values())).fetchall()]


def recent_sessions(limit=50):
    q = ("select session_id, min(at_ms) as started, max(at_ms) as last_seen, count(*) as events, "
         "max(case when kind = 'decision' then body end) as decision "
         "from audit_events group by session_id order by last_seen desc limit ")
    rows = _rows(q + ":n", q + "?", n=limit)
    for r in rows:
        r["decision"] = json.loads(r["decision"]) if r["decision"] else None
    return rows


def session_events(session_id):
    q = "select seq, at_ms, kind, body, prev_hash, row_hash from audit_events where session_id = "
    rows = _rows(q + ":s order by seq", q + "? order by seq", s=session_id)
    prev = "start"
    intact = True
    for r in rows:
        if r["prev_hash"] != prev or _hash(prev, session_id, r["seq"], r["at_ms"], r["kind"], r["body"]) != r["row_hash"]:
            intact = False
        prev = r["row_hash"]
        r["body"] = json.loads(r["body"])
    return {"events": rows, "chain_intact": intact}


def backend_name():
    return "postgres" if _db_url() else "sqlite"
