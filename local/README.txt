abc credit loan chatbot, localhost version

this runs the whole thing on your own machine, with the stronger model. nothing needs node, and
no data leaves the machine unless you add language model keys.

you need python 3.10, 3.11, 3.12 or 3.13 (python.org/downloads)
on windows, tick "add python to path" in the installer.

mac or linux
  open a terminal in this folder and run:
    sh start_mac_linux.sh

windows
  double click start_windows.bat

the first run installs what it needs into a .venv folder here and takes a few minutes. after that it
starts straight away and opens http://127.0.0.1:8000 in your browser. press ctrl+c to stop it.

optional: language model
  the chatbot works without this, using scripted wording. to let it read free-form messages and
  word the explanations, copy .env.example to .env and paste an nvidia nim key or up to three.

audit log
  stored in audit_log.sqlite3 in this folder. open the audit log page in the site, no key is
  needed on localhost.

scoring a csv (the hidden test set for example)
  mac or linux:  .venv/bin/python -m pipeline.predict test.csv predictions.csv --better
  windows:       .venv\Scripts\python -m pipeline.predict test.csv predictions.csv --better

if something goes wrong
  port taken:     set PORT=8001 (windows) or PORT=8001 sh start_mac_linux.sh (mac, linux)
  failed install: delete the .venv folder and run the start script again
  mac says it can't be opened: run it with sh as shown above rather than double clicking
