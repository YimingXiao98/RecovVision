#!/usr/bin/env python
"""Grouped bar chart of minority-class F1 (Not Occupied positive) across the four
backends, one-stage vs two-stage. Values are taken verbatim from Table tab:open-baselines
in main.tex so the figure and table cannot disagree. Addresses R1-3 / R3-M4 by giving the
open-baselines section a single visual anchor for the capability ranking.
"""
import os
from pathlib import Path
import numpy as np
import matplotlib as mpl
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

# Arial-compatible font (Arimo) for consistency across all charts
for _f in (Path.home() / ".local/share/fonts/arimo").glob("*.ttf"):
    fm.fontManager.addfont(str(_f))
mpl.rcParams["font.family"] = "Arimo"

OUT = "occupancy_figures"
os.makedirs(OUT, exist_ok=True)

MODELS = ["GPT-4o", "Qwen3-VL-235B", "Qwen3.5-397B", "GLM-4.6V"]
F1_ONE = [0.804, 0.706, 0.619, 0.425]   # one-stage  (tab:open-baselines)
F1_TWO = [0.840, 0.714, 0.681, 0.546]   # two-stage

C_ONE, C_TWO = "#88b8e8", "#c898b8"      # change-analysis palette: blue (one-stage), mauve (two-stage)
x = np.arange(len(MODELS)); w = 0.38

fig, ax = plt.subplots(figsize=(4, 2))
b1 = ax.bar(x - w/2, F1_ONE, w, label="One-stage", color=C_ONE, edgecolor="black", linewidth=0.7)
b2 = ax.bar(x + w/2, F1_TWO, w, label="Two-stage", color=C_TWO, edgecolor="black", linewidth=0.7)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=14)
ax.set_ylabel("F1 (Not Occupied, positive class)", fontsize=15)
ax.set_ylim(0, 0.95)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", labelsize=13)
ax.legend(frameon=False, fontsize=14, loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/open_baselines_f1.png", dpi=300, bbox_inches="tight")
print("saved", f"{OUT}/open_baselines_f1.png")
