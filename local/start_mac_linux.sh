#!/bin/sh
# run with: sh start_mac_linux.sh
cd "$(dirname "$0")" || exit 1

py=""
for name in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$name" >/dev/null 2>&1; then
        if "$name" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
            py="$name"
            break
        fi
    fi
done
if [ -z "$py" ]; then
    echo "This needs python 3.10 to 3.13. Get it from https://www.python.org/downloads/ and run this again."
    exit 1
fi

if [ ! -d .venv ]; then
    echo "First run: setting things up, this takes a few minutes."
    "$py" -m venv .venv || exit 1
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --quiet -r requirements-local.txt || exit 1
fi
exec .venv/bin/python run_local.py
