#!/usr/bin/env python
"""TP/TN/FP/FN example panel from the two-stage VLM outputs (addresses R3's request
to show representative true/false positive/negative cases and failure modes).
Positive class = Not Occupied. Green frame = correct, red frame = error. Each panel
lists the visual cues the VLM flagged as present for that facade.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = ".."
IMG = {1: f"{ROOT}/data/FacadeTrack/Visit1_processed_images"}
OUT = "occupancy_figures"
os.makedirs(OUT, exist_ok=True)

# (cell, file, GT, two-stage pred, flagged cues, correct?)
PANELS = [
    ("True Positive",  "6225.jpg", "Not Occupied", "Not Occupied",
     "damaged, debris, open windows, mud, repairs", True),
    ("True Negative",  "443.jpg",  "Occupied", "Occupied",
     "vehicle present", True),
    ("False Positive", "75286.jpg", "Occupied", "Not Occupied",
     "X/0 mark (storefront signage misread)", False),
    ("False Negative", "6671.jpg", "Not Occupied", "Occupied",
     "mud, vehicle present", False),
]

GREEN, RED = "#2e8b57", "#c0392b"
fig, axes = plt.subplots(2, 2, figsize=(11, 8.6),
                         gridspec_kw={"hspace": 0.42, "wspace": 0.06})
for ax, (cell, fn, gt, pred, cues, ok) in zip(axes.ravel(), PANELS):
    im = Image.open(f"{IMG[1]}/{fn}").convert("RGB")
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    col = GREEN if ok else RED
    for s in ax.spines.values():
        s.set_edgecolor(col); s.set_linewidth(4)
    ax.set_title(f"{cell}", fontsize=13, fontweight="bold", color=col, pad=6)
    txt = f"Ground truth: {gt}   |   Two-stage: {pred}\nVLM-flagged cues: {cues}"
    ax.set_xlabel(txt, fontsize=9, labelpad=6)
fig.suptitle("Two-stage prediction outcomes (positive class: Not Occupied)",
             fontsize=14, fontweight="bold", y=0.95)
fig.savefig(f"{OUT}/tp_fp_examples.png", dpi=300, bbox_inches="tight")
print("saved", f"{OUT}/tp_fp_examples.png")
