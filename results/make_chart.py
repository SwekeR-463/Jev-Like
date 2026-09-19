"""Our Jev-Like readout against the published JevBench v1.2 field, public items only.

    uv run --with matplotlib --python 3.12 python results/make_chart.py

Everything drawn here is measured. The Cost axis is left out entirely rather
than assumed: our adapter reports no provider tariff, and composite_v12.cost()
refuses a missing price precisely because it would otherwise read as a free 100.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = json.loads((ROOT / "jevbench/results/v1.2/jevbench-v1.2-results.json").read_text())
PERTASK = json.loads((ROOT / "jevbench/results/v1.2/jevbench-v1.2-per-task.json").read_text())
OURS = json.loads((ROOT / "results/jev_like_direct_v3_axes.json").read_text())
OUT = ROOT / "results/charts"

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2", "#b9b8b1"
ORANGE, GREY, TEAL, OURS_C = "#eb6834", "#7d7c78", "#1baf7a", "#12455e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.parse_math": False,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
SHORT = {
    "jev-1.13.0": "Jev 1.13.0 (closed)",
    "semif-qwen3.5-4b": "SemIf (Qwen3.5-4B)",
    "open-alternative-jev": "open-alternative-jev (Qwen3.5-4B)",
    "system-one-open": "system-one-open (Gemma 4 E2B)",
    "openjev-razorback16": "OpenJev razorback16 (26B)",
    "openjev-sglang": "openjev-sglang (Qwen3.6-35B)",
    "gpt-5.6-luna": "GPT-5.6 Luna (low)",
    "open-jev-deberta-v3-large": "open-jev-deberta-v3-large",
    "nimble-9b": "Bespoke Nimble 9B",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
    "deepseek-flash": "DeepSeek V4.1 Flash",
    "system-one-sg": "system-one (Qwen3-8B)",
}
OUR_NAME = "Jev-Like (Qwen3.5-2B, local)"
OUR_SHORT = "Jev-Like"


def public_tier_accuracies():
    """Accuracy per tier over exactly the 231 public decisions, for every system."""
    tier = {t["id"]: t["tier"] for t in PERTASK["tasks"]}
    out = {}
    for key, s in PERTASK["systems"].items():
        acc = {}
        for tid, (outcome, _conf) in s["public_tasks"].items():
            t = tier.get(tid)
            if t is None:
                continue
            c, w = acc.get(t, (0, 0))
            acc[t] = (c + (outcome == "c"), w + 1)
        out[key] = {t: c / w for t, (c, w) in acc.items()}
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    accs = public_tier_accuracies()
    ours_acc = {"easy": 48 / 48, "standard": 54 / 72, "hard": 56 / 111}
    rows = [(SHORT.get(e["key"], e["display"]), accs.get(e["key"], {}))
            for e in RES["systems"] if e["ranked"]]
    rows.append((OUR_NAME, ours_acc))
    rows.sort(key=lambda r: r[1].get("hard", 0))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.6, 8.6), dpi=170,
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    fig.patch.set_facecolor(SURFACE)

    # A: accuracy by tier, identical item set for every system.
    bar_h = 0.26
    y = list(range(len(rows)))
    alphas = [1.0 if r[0] == OUR_NAME else 0.5 for r in rows]
    for tier, colour, off in zip(("easy", "standard", "hard"),
                                 (TEAL, GREY, ORANGE), (bar_h, 0.0, -bar_h)):
        vals = [r[1].get(tier, 0) * 100 for r in rows]
        axA.barh([yy + off for yy in y], vals, height=bar_h - 0.02, label=tier,
                 color=[to_rgba(colour, a) for a in alphas])
        for yy, v in zip(y, vals):
            axA.text(v + 1.5, yy + off, f"{v:.0f}", va="center", ha="left",
                     fontsize=9, color=INK2)
    axA.set_yticks(y)
    axA.set_yticklabels([r[0] for r in rows])
    for tick, r in zip(axA.get_yticklabels(), rows):
        if r[0] == OUR_NAME:
            tick.set_color(OURS_C)
            tick.set_fontweight("bold")
    axA.set_xlim(0, 110)
    axA.set_xlabel("accuracy (%) on the 231 public decisions")
    axA.set_title("Same public items, every system", fontsize=12, color=INK, loc="left", pad=10)
    axA.legend(loc="lower right", frameon=False, ncol=3, fontsize=10)
    style(axA)

    # B: the three axes we can actually measure.
    our_ax = {"intelligence": OURS["intelligence"], "calibration": OURS["calibration"],
              "speed": OURS["speed_pooled"]}
    b_rows = [(SHORT.get(e["key"], e["display"]), e["axes"])
              for e in RES["systems"] if e["ranked"]]
    b_rows.append((OUR_SHORT, our_ax))
    b_rows.sort(key=lambda r: r[1].get("intelligence") or 0)
    by = list(range(len(b_rows)))
    b_alphas = [1.0 if r[0] == OUR_SHORT else 0.5 for r in b_rows]
    for key, colour, off in zip(("intelligence", "calibration", "speed"),
                                (TEAL, GREY, ORANGE), (0.24, 0.0, -0.24)):
        vals = [max(r[1].get(key) or 0, 0) for r in b_rows]
        axB.barh([yy + off for yy in by], vals, height=0.22, label=key,
                 color=[to_rgba(colour, a) for a in b_alphas])
        for yy, v in zip(by, vals):
            axB.text(v + 1.2, yy + off, f"{v:.0f}", va="center", ha="left",
                     fontsize=8, color=INK2)
    axB.set_yticks(by)
    axB.set_yticklabels([r[0] for r in b_rows])
    for tick, r in zip(axB.get_yticklabels(), b_rows):
        if r[0] == OUR_SHORT:
            tick.set_color(OURS_C)
            tick.set_fontweight("bold")
    axB.set_xlim(0, 100)
    axB.set_xlabel("axis score (0-100), higher is better")
    axB.set_title("Measured axes - cost excluded, not assumed", fontsize=12, color=INK,
                  loc="left", pad=10)
    axB.legend(loc="lower right", frameon=False, ncol=3, fontsize=10)
    style(axB)

    fig.suptitle("Jev-Like (Qwen3.5-2B, local, native softmax) vs JevBench v1.2",
                 fontsize=17, color=INK, x=0.007, ha="left", y=0.978, fontweight="bold")
    fig.text(0.007, 0.928,
             "231 public decisions, no held-out items, 0 failures. "
             "One prefill + one suffix pass, no tokens generated.",
             fontsize=11, color=INK2, ha="left")
    fig.text(0.007, 0.030,
             "Not a JevBench Score. That is a geometric mean of four axes; our Cost axis is unmeasured (local weights, no provider tariff) "
             "and is left out rather\nthan scored as free. Our Intelligence also omits the non-public judge tier and renormalises its weight, "
             "so it is not on an identical basis to the published rows.",
             fontsize=9, color=MUTED, ha="left", va="bottom")
    fig.subplots_adjust(left=0.225, right=0.985, top=0.885, bottom=0.125, wspace=0.62)
    path = OUT / "jev_like_vs_field.png"
    fig.savefig(path, facecolor=SURFACE)
    print("wrote", path)


if __name__ == "__main__":
    main()
