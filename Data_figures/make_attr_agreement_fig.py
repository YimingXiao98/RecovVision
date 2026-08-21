#!/usr/bin/env python
"""Grouped bar chart: per-attribute agreement of each open backend with GPT-4o at the
perception stage. Reads occupancy_figures/attr_agreement.json (written by attr_agreement.py).
Attributes sorted by mean agreement; vehicle_presence is the lone low outlier.
"""
import json
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
D = json.load(open(f"{OUT}/attr_agreement.json"))
MODELS = ["Qwen3-VL-235B", "Qwen3.5-397B", "GLM-4.6V"]
COLORS = ["#88b8e8", "#e89898", "#c898b8"]  # change-analysis palette: blue, salmon, mauve
SHORT = {"house_destruction": "House\ndestroyed", "structural_damage": "Structural\ndamage",
         "exterior_debris": "Debris", "open_doors_windows": "Open\ndoors/windows",
         "site_accessible": "Site\naccessible", "exterior_mud": "Mud",
         "emergency_markings": "Emergency\nmarkings", "major_repairs": "Major\nrepairs",
         "vehicle_presence": "Vehicle\npresence"}
ATTRS = list(SHORT.keys())
# sort by mean agreement across models, descending
mean_ag = {k: np.mean([D[m]["agree"][k] for m in MODELS]) for k in ATTRS}
ATTRS = sorted(ATTRS, key=lambda k: -mean_ag[k])

x = np.arange(len(ATTRS)); w = 0.26
fig, ax = plt.subplots(figsize=(11, 4.6))
for i, (m, c) in enumerate(zip(MODELS, COLORS)):
    vals = [D[m]["agree"][k] * 100 for k in ATTRS]
    ax.bar(x + (i - 1) * w, vals, w, label=m, color=c, edgecolor="black", linewidth=0.6)
ax.set_xticks(x); ax.set_xticklabels([SHORT[k] for k in ATTRS], fontsize=12.5)
ax.set_ylabel("Agreement with GPT-4o (%)", fontsize=15)
ax.set_ylim(0, 105)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.45, linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", labelsize=13)
ax.legend(frameon=False, fontsize=13, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.34))
fig.tight_layout()
fig.savefig(f"{OUT}/attr_agreement.png", dpi=300, bbox_inches="tight")
print("saved", f"{OUT}/attr_agreement.png")
