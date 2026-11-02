"""Builds the zip for the github release: python local/build_release.py"""
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / "dist" / "abc-credit-local.zip"
skip_parts = {"__pycache__", ".pytest_cache"}


def add_tree(z, folder, into):
    for f in sorted(folder.rglob("*")):
        if f.is_file() and not skip_parts & set(f.parts):
            z.write(f, f"{into}/{f.relative_to(folder).as_posix()}")


def main():
    site = root / "web" / "out"
    model = root / "server" / "creditbot" / "model_local" / "catboost.cbm"
    if not site.exists() or not model.exists():
        raise SystemExit("build the site and train the model first")
    out.parent.mkdir(exist_ok=True)
    name = "abc-credit-local"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(root / "run_local.py", f"{name}/run_local.py")
        z.write(root / ".env.example", f"{name}/.env.example")
        z.write(root / "local" / "requirements-local.txt", f"{name}/requirements-local.txt")
        z.write(root / "local" / "start_mac_linux.sh", f"{name}/start_mac_linux.sh")
        z.write(root / "local" / "start_windows.bat", f"{name}/start_windows.bat")
        z.write(root / "local" / "README.txt", f"{name}/README.txt")
        z.write(root / "server" / "main.py", f"{name}/server/main.py")
        add_tree(z, root / "server" / "creditbot", f"{name}/server/creditbot")
        z.write(root / "pipeline" / "__init__.py", f"{name}/pipeline/__init__.py")
        z.write(root / "pipeline" / "predict.py", f"{name}/pipeline/predict.py")
        add_tree(z, site, f"{name}/web/out")
    print(out, round(out.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
