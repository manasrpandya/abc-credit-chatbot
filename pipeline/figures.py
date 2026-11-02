"""Charts for the report, from report/results/*.json. Run: python -m pipeline.figures"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import root

results = root / "report" / "results"
figures = root / "report" / "figures"
ink, ember, soft, grid = "#1b1613", "#e8441a", "#8a8078", "#e6e1db"

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": soft, "axes.labelcolor": ink, "xtick.color": ink,
                     "ytick.color": ink, "axes.grid": True, "grid.color": grid, "axes.axisbelow": True,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def question_curve():
    s = json.loads((results / "field_search.json").read_text())
    pts = s["path"]
    xs = [p["asked"] for p in pts]
    ys = [p["auc"] for p in pts]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(xs, ys, color=ink, lw=2, marker="o", ms=5)
    ax.axvline(8, color=ember, lw=1.5)
    ax.annotate("what the chatbot asks", (8, 0.64), xytext=(6, 0), textcoords="offset points", color=ember)
    for p in pts[1:]:
        if p["asked"] <= 3:
            ax.annotate("without " + p["removed"].replace("_", " "), (p["asked"], p["auc"]),
                        xytext=(8, -4), textcoords="offset points", fontsize=9, color=soft)
    ax.set_xlabel("questions asked")
    ax.set_ylabel("cross validated auc")
    ax.invert_xaxis()
    ax.set_title("Most of the signal needs only a few questions", loc="left", fontsize=12)
    fig.savefig(figures / "questions_vs_auc.png")
    plt.close(fig)


def threshold_curve():
    m = json.loads((results / "metrics.json").read_text())
    c = m["threshold_curve"]
    xs = [r["threshold"] for r in c]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(xs, [r["recall"] for r in c], color=ember, lw=2)
    ax.plot(xs, [r["precision"] for r in c], color=ink, lw=2)
    ax.plot(xs, [r["declined_share"] for r in c], color=soft, lw=2)
    last = -1
    ax.annotate("recall on declined", (xs[last], c[last]["recall"]), xytext=(-110, 14), textcoords="offset points", color=ember)
    ax.annotate("precision on declined", (xs[last], c[last]["precision"]), xytext=(-140, 8), textcoords="offset points", color=ink)
    ax.annotate("share of applicants declined", (0.2, [r for r in c if r["threshold"] >= 0.2][0]["declined_share"]),
                xytext=(10, 12), textcoords="offset points", color=soft)
    ax.axvline(m["threshold"]["hosted"], color=ember, lw=1, ls="--")
    ax.set_xlabel("probability cut-off")
    ax.set_title("A stricter cut-off catches more declines and turns away more people", loc="left", fontsize=12)
    ax.set_xlim(0.01, 0.5)
    fig.savefig(figures / "threshold_tradeoff.png")
    plt.close(fig)


def confusion():
    m = json.loads((results / "metrics.json").read_text())["test"]["hosted"]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    grid_vals = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    ax.imshow([[0, 0.25], [0.25, 0.6]], cmap="Oranges", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{grid_vals[i][j]:,}", ha="center", va="center", fontsize=14, color=ink)
    ax.set_xticks([0, 1], ["said approve", "said decline"])
    ax.set_yticks([0, 1], ["was approved", "was declined"])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Locked test rows", loc="left", fontsize=12)
    fig.savefig(figures / "confusion_matrix.png")
    plt.close(fig)


if __name__ == "__main__":
    figures.mkdir(parents=True, exist_ok=True)
    question_curve()
    threshold_curve()
    confusion()
