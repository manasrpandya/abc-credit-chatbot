"""Starts the localhost version: python run_local.py

Serves the built site and the api together on one port, with the stronger blended model
if it's there, and keeps its audit log in a local sqlite file.
"""
import os
import secrets
import sys
import threading
import webbrowser
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "server"))


def read_env():
    env_file = root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            if value:
                os.environ.setdefault(name.strip(), value)


def main():
    read_env()
    os.environ["CREDITBOT_LOCAL"] = "1"
    os.environ.setdefault("STATE_SECRET", secrets.token_hex(32))
    os.environ.setdefault("AUDIT_DB", str(root / "audit_log.sqlite3"))
    # the hosted database is never used from here
    os.environ.pop("DATABASE_URL", None)
    os.environ.pop("POSTGRES_URL", None)

    better = (root / "server" / "creditbot" / "model_local" / "catboost.cbm").exists()
    if better and "--hosted-model" not in sys.argv:
        try:
            import catboost  # noqa: F401
            os.environ["CREDITBOT_MODEL"] = "local"
        except ImportError:
            print("catboost isn't installed, so using the lighter hosted model instead.")
    if os.environ.get("CREDITBOT_MODEL") != "local":
        print("running with the hosted model")

    import uvicorn
    from fastapi.staticfiles import StaticFiles
    from main import app

    site = root / "web" / "out"
    if not site.exists():
        sys.exit("web/out is missing. Use the release zip, or run: cd web && npm install && npm run build")
    app.mount("/", StaticFiles(directory=site, html=True), name="site")

    port = int(os.environ.get("PORT", 8000))
    print(f"\nopen http://127.0.0.1:{port}\npress ctrl+c to stop\n")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
