#!/usr/bin/env python
"""Class-stratified version of the error-pattern figure (Fig. 8), addressing R3's
request to show which occupancy class drives one-stage vs two-stage differences.

Universe: the 240 matched parcels labeled in BOTH visits (same set as the original
Fig. 8). Unit here is the facade observation (240 parcels x 2 visits = 480), so each
observation carries a single ground-truth class. Bars show the share of observations
within each class falling into the four agreement categories; counts are annotated.
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib as mpl
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# Arial-compatible font (Arimo) for consistency across all charts
for _f in (Path.home() / ".local/share/fonts/arimo").glob("*.ttf"):
    fm.fontManager.addfont(str(_f))
mpl.rcParams["font.family"] = "Arimo"

OUT = "occupancy_figures"
os.makedirs(OUT, exist_ok=True)

d = pd.read_csv("spatial_matched_pairs_occupancy_analysis.csv").drop_duplicates(
    subset=["visit1_objectid", "june25_objectid"])
bk = d[d.Visit1_Ground_Truth.isin(["Occupied", "Not Occupied"])
       & d.June25_Ground_Truth.isin(["Occupied", "Not Occupied"])]

rows = []
for _, r in bk.iterrows():
    rows.append((r.Visit1_Ground_Truth, r.Visit1_Pred_wo_LLM, r.Visit1_Pred_LLM))
    rows.append((r.June25_Ground_Truth, r.June25_Pred_wo_LLM, r.June25_Pred_LLM))
f = pd.DataFrame(rows, columns=["gtc","one","two"])

CATS = ["Both\nCorrect", "Both\nWrong", "One-stage\nOnly Wrong", "Two-stage\nOnly Wrong"]
COLORS = ["#cd7e7e", "#e3bd84", "#8d5b5b", "#7fa882"]  # match original Fig. 8 palette
CLASSES = ["Occupied", "Not Occupied"]

counts = {}
for cls in CLASSES:
    s = f[f["gtc"]==cls]
    one_ok, two_ok = s.one == cls, s.two == cls
    counts[cls] = [int((one_ok & two_ok).sum()), int((~one_ok & ~two_ok).sum()),
                   int((~one_ok & two_ok).sum()), int((one_ok & ~two_ok).sum())]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, cls in zip(axes, CLASSES):
    n = sum(counts[cls])
    pct = [100.0 * c / n for c in counts[cls]]
    bars = ax.bar(range(4), pct, color=COLORS, edgecolor="black", linewidth=0.8, width=0.72)
    for i, (b, c) in enumerate(zip(bars, counts[cls])):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{c}\n({pct[i]:.1f}%)", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_title(f"{cls}  (n = {n})", fontsize=13, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(CATS, fontsize=10)
    ax.set_ylim(0, 112)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("Share of facades within class (%)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "error_analysis_by_class.png"), dpi=300, bbox_inches="tight")
print("saved", os.path.join(OUT, "error_analysis_by_class.png"))
print({c: counts[c] for c in CLASSES})
