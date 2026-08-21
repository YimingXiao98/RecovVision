#!/usr/bin/env python
"""Merged error-pattern figure: one panel for ALL facades plus one per ground-truth
class (Occupied, Not Occupied), at the facade-observation level (240 confident matched
parcels x 2 visits = 480). The Overall panel is the exact sum of the two class panels.
Replaces the former separate aggregate (Fig. 8) and class-stratified (Fig. 9) figures.
Run from Data_figures/.
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib as mpl
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

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
f = pd.DataFrame(rows, columns=["gtc", "one", "two"])

CATS = ["Both\nCorrect", "Both\nWrong", "One-stage\nOnly Wrong", "Two-stage\nOnly Wrong"]
COLORS = ["#cd7e7e", "#e3bd84", "#8d5b5b", "#7fa882"]

def counts(sub):
    one_ok, two_ok = sub.one == sub.gtc, sub.two == sub.gtc
    return [int((one_ok & two_ok).sum()), int((~one_ok & ~two_ok).sum()),
            int((~one_ok & two_ok).sum()), int((one_ok & ~two_ok).sum())]

panels = [("All facades", f),
          ("Occupied", f[f["gtc"] == "Occupied"]),
          ("Not Occupied", f[f["gtc"] == "Not Occupied"])]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
for ax, (title, sub) in zip(axes, panels):
    c = counts(sub); n = sum(c); pct = [100.0 * x / n for x in c]
    bars = ax.bar(range(4), pct, color=COLORS, edgecolor="black", linewidth=0.8, width=0.72)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{c[i]}\n({pct[i]:.1f}%)", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_title(f"{title}  (n = {n})", fontsize=14, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(CATS, fontsize=11.5)
    ax.set_ylim(0, 112)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", labelsize=12)
axes[0].set_ylabel("Share of facades (%)", fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUT}/error_analysis_merged.png", dpi=300, bbox_inches="tight")
print("saved", f"{OUT}/error_analysis_merged.png")
print({t: counts(s) for t, s in panels})
