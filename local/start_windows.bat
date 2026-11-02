@echo off
rem double click this file, or run it from a command prompt
cd /d "%~dp0"

set PY=
for %%v in (3.12 3.13 3.11 3.10) do (
    if not defined PY (
        py -%%v -c "import sys" >nul 2>&1 && set PY=py -%%v
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1 && set PY=python
)
if not defined PY (
    echo This needs python 3.10 to 3.13. Get it from https://www.python.org/downloads/ and run this again.
    pause
    exit /b 1
)

if not exist .venv (
    echo First run: setting things up, this takes a few minutes.
    %PY% -m venv .venv || goto failed
    .venv\Scripts\python -m pip install --quiet --upgrade pip
    .venv\Scripts\python -m pip install --quiet -r requirements-local.txt || goto failed
)
.venv\Scripts\python run_local.py
exit /b 0

:failed
echo Setup failed. See the message above, then delete the .venv folder and try again.
pause
exit /b 1
