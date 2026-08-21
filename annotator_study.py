#!/usr/bin/env python
"""
Inter-annotator reliability study for Recov-Vision (addresses R1-6, R2-2).

The original annotation assigned DISJOINT subsets to each annotator (no overlap),
so no agreement statistic could be computed. This builds a small stratified,
double-annotation sample (oversampling the minority "Not Occupied" class) for
independent re-labeling, then scores Cohen's/Fleiss' kappa + percent agreement.

  python annotator_study.py --make-sample      # writes blind sample + answer key
  python annotator_study.py --score            # after annotators fill the sample
"""
import argparse
import pandas as pd
from pathlib import Path
from run_openrouter_baseline import RESULT_CSV, LABEL_COL, ID_COL, IMG_DIR, OUTDIR

SAMPLE = OUTDIR / "annotator_study_sample.csv"   # blind: annotators fill label_A/B/C
KEY    = OUTDIR / "annotator_study_key.csv"      # ground-truth reference (not shown to raters)
PER_CELL = 25  # per (visit x class); minority class is the binding constraint

def make_sample():
    rows = []
    for visit in (1, 2):
        df = pd.read_csv(RESULT_CSV[visit])
        lc, ic = LABEL_COL[visit], ID_COL[visit]
        df = df[df[lc].isin(["Occupied", "Not Occupied"])]
        def _has_img(r):
            stem = str(r[ic]).strip()
            stem = stem if stem.lower().endswith(".jpg") else stem + ".jpg"
            return (IMG_DIR[visit] / stem).exists()
        for cls in ("Not Occupied", "Occupied"):
            sub = df[df[lc] == cls]
            sub = sub[sub.apply(_has_img, axis=1)]
            take = sub.sample(min(PER_CELL, len(sub)), random_state=42)
            for _, r in take.iterrows():
                stem = str(r[ic]).strip()
                if not stem.lower().endswith(".jpg"):
                    stem += ".jpg"
                rows.append({"visit": visit, "objectid": stem,
                             "image_path": str(IMG_DIR[visit] / stem),
                             "ground_truth": cls})
    s = pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle so raters are blind to class
    s.insert(0, "sample_id", range(1, len(s) + 1))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    s[["sample_id", "objectid", "visit", "ground_truth"]].to_csv(KEY, index=False)
    blind = s[["sample_id", "visit", "image_path"]].copy()
    for col in ("label_A", "label_B", "label_C"):
        blind[col] = ""   # annotators enter: Occupied / Not Occupied / Uncertain
    blind.to_csv(SAMPLE, index=False)
    n_no = (s["ground_truth"] == "Not Occupied").sum()
    print(f"Wrote {len(s)} samples -> {SAMPLE}  (Not Occupied={n_no}, Occupied={len(s)-n_no})")
    print(f"Answer key -> {KEY}")
    print("Annotators: open each image_path, fill label_A/label_B/label_C independently "
          "(Occupied / Not Occupied / Uncertain), do not consult the key.")

MARK = {"label_A": "Occupied", "label_B": "Not Occupied", "label_C": "Uncertain"}

def _to_label(r):
    """Resolve a row to Occupied / Not Occupied / Uncertain, accepting either a full
    word written in any column, or a single mark in the column for that class."""
    cells = {c: str(r.get(c, "")).strip() for c in ("label_A", "label_B", "label_C")}
    for v in cells.values():
        lv = v.lower()
        if lv == "occupied": return "Occupied"
        if lv == "not occupied": return "Not Occupied"
        if lv == "uncertain": return "Uncertain"
    marked = [c for c, v in cells.items() if v and v.lower() != "nan"]
    return MARK[marked[0]] if len(marked) == 1 else None

def score(labels_path=str(SAMPLE)):
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    d = pd.read_csv(labels_path)
    key = pd.read_csv(KEY)[["sample_id", "ground_truth"]]
    d["label"] = d.apply(_to_label, axis=1)
    m = d.merge(key, on="sample_id")
    n_unc = (m["label"] == "Uncertain").sum(); n_none = m["label"].isna().sum()
    sub = m[m["label"].isin(["Occupied", "Not Occupied"])]
    k = cohen_kappa_score(sub["ground_truth"], sub["label"])
    pa = (sub["ground_truth"].values == sub["label"].values).mean()
    print(f"Reliability check vs adjudicated labels ({labels_path}):")
    print(f"  scored={len(sub)}  Uncertain={n_unc}  unparsed={n_none}")
    print(f"  Cohen's kappa = {k:.3f}   percent agreement = {pa*100:.1f}%")
    print("  confusion (rows=GT [Not Occupied, Occupied], cols=annotator):")
    print(confusion_matrix(sub["ground_truth"], sub["label"], labels=["Not Occupied", "Occupied"]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-sample", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--labels", default=str(SAMPLE), help="filled labels CSV to score")
    a = ap.parse_args()
    if a.make_sample:
        make_sample()
    elif a.score:
        score(a.labels)
    else:
        ap.print_help()
