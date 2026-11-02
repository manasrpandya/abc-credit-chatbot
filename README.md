# ABC Credit loan chatbot

A chatbot that takes a two-wheeler loan application in a short conversation, scores it with a trained
model and returns approve or decline straight away, along with what influenced the result.

Mini project for Z6008.

**Live site:** https://abc-credit-loans.vercel.app

The training data isn't in this repo since it isn't mine to publish. Put `training_data.csv` in the repo root
if you want to rerun the pipeline. The site and the localhost version work without it.

## How the decision works

The model decides, the language model only talks.

- A LightGBM model gives a probability, and that gets compared to a fixed cut-off. No language model is involved.
- A plain state machine owns the questions and their rules.
- An NVIDIA NIM model helps read messy messages and words the explanation. Anything it suggests goes through the
  same validators as a typed answer, and its explanation is thrown out if it contradicts the decision or adds a number.
- If NIM is down or no keys are set, the bot falls back to scripted wording and still finishes.

## Results

On a 20% test split that was kept locked during model selection (21,686 rows):

| | AUC | Avg. precision | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Always approve | 0.500 | 0.053 | 94.7% | 0% | 0% | 0 |
| Hosted model (LightGBM) | 0.799 | 0.199 | 87.6% | 18.9% | 41.1% | 0.259 |
| Localhost model (LightGBM + CatBoost) | 0.801 | 0.200 | 85.0% | 17.5% | 49.7% | 0.259 |

Precision, recall and F1 are for the declined class. That split turned out to be a kind one: refitting on 8 other random
splits gave AUCs of 0.786 to 0.798, so about **0.79** is the fair number to expect on new data. With shuffled labels the
pipeline scores 0.50, so nothing is leaking.

The full reasoning is in [report/report.md](report/report.md), and the data exploration is in
[notebooks/01_eda.ipynb](notebooks/01_eda.ipynb).

## Run it on your own machine

This is the stronger model, and it needs python 3.10 to 3.13 but no node.

1. Download `abc-credit-local.zip` from the [releases page](https://github.com/manasrpandya/abc-credit-chatbot/releases/latest) and unzip it.
2. **Mac or Linux:** open a terminal in the folder and run `sh start_mac_linux.sh`.
   **Windows:** double click `start_windows.bat`.
3. It sets itself up on the first run (a few minutes), then opens http://127.0.0.1:8000.

To turn on the language model, copy `.env.example` to `.env` and paste up to three NIM keys. It works fine without them.

## Working from source

Mac and Linux:

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-train.txt
```

Windows:

```bat
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements-train.txt
```

On a Mac, training also needs `brew install libomp` because of LightGBM. The app itself doesn't, since it walks the exported
trees in plain python.

Then, with `.venv/bin/python` (or `.venv\Scripts\python`) as `python`:

```sh
python -m pytest tests                 # no csv needed
python -m pipeline.search_fields       # which questions are worth asking
python -m pipeline.train               # compare models, evaluate, export
python -m pipeline.stability           # repeat on other splits, shuffled label check
python -m pipeline.figures             # charts for the report
python -m pipeline.site_data           # numbers for the site
python notebooks/build_eda.py          # rerun the notebook
```

To build the site (needs node), run `npm install` and `npm run build` in `web`. After that `python run_local.py` serves
the whole thing, or use `npm run dev` in `web` next to `uvicorn main:app --app-dir server` while working on it.

## Scoring a new file

```sh
python -m pipeline.predict test.csv predictions.csv            # hosted model
python -m pipeline.predict test.csv predictions.csv --better   # blended model
```

The input has the same columns as the training file, without `Decision`. Blank cells and unseen categories are fine.

## What's where

```
pipeline/    field search, training, evaluation, scoring a csv
server/      the api, chat flow, masking, audit log, tree scorer and the exported model
web/         the site (next.js, exported as static files)
notebooks/   data exploration
report/      the report, figures and the raw result files
tests/       scorer parity, leakage checks, masking, the chat flow
local/       start scripts and the release builder
```

Keys and secrets stay in `.env`, which is ignored.
