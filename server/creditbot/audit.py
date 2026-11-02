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
_con = None


def _db_url():
    return os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL") or ""


def _connect():
    import pg8000.native
    u = urlparse(_db_url())
    return pg8000.native.Connection(user=u.username, password=u.password, host=u.hostname,
                                    port=u.port or 5432, database=u.path.lstrip("/"),
                                    ssl_context=ssl.create_default_context(), timeout=6)


def _with_pg(work):
    """Opening a tls connection is by far the slowest step, so keep one open between requests.
    It can go stale while the database sleeps, hence the one retry."""
    global _con
    with _lock:
        for attempt in (1, 2):
            try:
                if _con is None:
                    _con = _connect()
                return work(_con)
            except Exception:
                try:
                    if _con is not None:
                        _con.close()
                except Exception:
                    pass
                _con = None
                if attempt == 2:
                    raise


def _sqlite_path():
    return Path(os.environ.get("AUDIT_DB", Path.cwd() / "audit_log.sqlite3"))


def _sqlite():
    con = sqlite3.connect(_sqlite_path(), timeout=5)
    con.row_factory = sqlite3.Row
    return con


create_sql = """create table if not exists audit_events (
    id text primary key, session_id text not null, seq integer not null, at_ms bigint not null,
    kind text not null, body text not null, prev_hash text not null, row_hash text not null)"""
index_sql = "create unique index if not exists audit_session on audit_events (session_id, seq)"


def _ensure():
    global _ready
    if _ready:
        return
    if _db_url():
        def setup(con):
            con.run(create_sql)
            con.run(index_sql)
        _with_pg(setup)
    else:
        with _sqlite() as con:
            con.execute(create_sql)
            con.execute(index_sql)
    _ready = True


def _hash(prev, session_id, seq, at_ms, kind, body):
    raw = "|".join([prev, session_id, str(seq), str(at_ms), kind, body])
    return hashlib.sha256(raw.encode()).hexdigest()


def _chain(session_id, seq, prev, events):
    rows = []
    for kind, body in events:
        seq += 1
        text = json.dumps(body, ensure_ascii=False, default=str, sort_keys=True)
        at_ms = int(time.time() * 1000)
        row = _hash(prev, session_id, seq, at_ms, kind, text)
        rows.append((uuid.uuid4().hex, session_id, seq, at_ms, kind, text, prev, row))
        prev = row
    return rows


def _write_pg(session_id, events):
    def work(con):
        last = con.run("select seq, row_hash from audit_events where session_id = :s "
                       "order by seq desc limit 1", s=session_id)
        rows = _chain(session_id, last[0][0] if last else -1, last[0][1] if last else "start", events)
        slots, params = [], {}
        for n, row in enumerate(rows):
            names = [f"c{n}_{i}" for i in range(len(row))]
            slots.append("(" + ", ".join(":" + x for x in names) + ")")
            params.update(zip(names, row))
        con.run("insert into audit_events values " + ", ".join(slots), **params)
    _with_pg(work)


def _write_sqlite(session_id, events):
    with _lock, _sqlite() as con:
        last = con.execute("select seq, row_hash from audit_events where session_id = ? "
                           "order by seq desc limit 1", (session_id,)).fetchone()
        rows = _chain(session_id, last["seq"] if last else -1, last["row_hash"] if last else "start", events)
        con.executemany("insert into audit_events values (?,?,?,?,?,?,?,?)", rows)


def write_many(session_id, events):
    """events: list of (kind, body). A logging failure never breaks the chat but does get printed."""
    if not events:
        return True
    try:
        _ensure()
        (_write_pg if _db_url() else _write_sqlite)(session_id, events)
        return True
    except Exception as err:
        print("audit write failed:", type(err).__name__, flush=True)
        return False


def _rows(sql_pg, sql_lite, **params):
    _ensure()
    if _db_url():
        def work(con):
            rows = con.run(sql_pg, **params)
            names = [c["name"] for c in con.columns]
            return [dict(zip(names, r)) for r in rows]
        return _with_pg(work)
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
