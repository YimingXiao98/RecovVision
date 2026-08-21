#!/usr/bin/env python
"""Per-attribute agreement between each open-weight vision backend and GPT-4o, at the
perception stage (the 9 extracted boolean attributes), pooled over both visits.

Localizes where the backends diverge: high agreement = robust, model-independent cue;
low agreement = subjective / model-dependent cue. Supports the modularity claim and the
"is the schema a reliable bottleneck?" question. Run from Data_figures/.
"""
import json, re
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = ".."
CANON = ["house_destruction", "structural_damage", "exterior_debris", "open_doors_windows",
         "site_accessible", "exterior_mud", "emergency_markings", "major_repairs",
         "vehicle_presence"]
# GPT-4o verbose key -> canonical; site_accessible is the NEGATION of "building inaccessible"
G2C = {
    "house destroyed": ("house_destruction", False),
    "house damaged": ("structural_damage", False),
    "debris outside house": ("exterior_debris", False),
    "doors or windows open": ("open_doors_windows", False),
    "building inaccessible": ("site_accessible", True),   # invert
    "is there large area of mud outside,": ("exterior_mud", False),
    "is there a red cross or zero written on the building": ("emergency_markings", False),
    "major repair work going on around the house (Wall pannels or Roof)": ("major_repairs", False),
    "are there any cars parked right outside the concerned house (not the black car rooftop partially visible)": ("vehicle_presence", False),
}
MODELS = {"Qwen3-VL-235B": "qwen", "Qwen3.5-397B": "qwen35-nothink", "GLM-4.6V": "glm46v-nothink"}

def parse_open(s):
    try:
        j = json.loads(str(s))
        return {k: bool(j[k]) for k in CANON if k in j}
    except Exception:
        return None

def parse_gpt(s):
    try:
        j = json.loads(re.sub(r"```json|```", "", str(s)).strip())
        out = {}
        for vk, (ck, inv) in G2C.items():
            if vk in j:
                out[ck] = (not bool(j[vk])) if inv else bool(j[vk])
        return out
    except Exception:
        return None

def gpt_table():
    rows = {}
    for v, f, col, idc in [(1, "FieldTrip1_occupancy_wo_LLM_result.csv", "Vision Model Output", "objectid"),
                            (2, "June25_occupancy_wo_LLM_result.csv", "Vision Model Output", "objectid")]:
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            a = parse_gpt(r[col])
            if a:
                rows[(v, str(r[idc]).strip())] = a
    return rows

gpt = gpt_table()
results = {}
for name, tag in MODELS.items():
    per_attr_match = {k: [0, 0] for k in CANON}  # [matches, total]
    pairs = {k: ([], []) for k in CANON}
    for v in (1, 2):
        d = pd.read_csv(f"{ROOT}/rebuttal_out/{tag}_visit{v}.csv")
        for _, r in d.iterrows():
            oid = str(r["objectid"]).strip()
            ao = parse_open(r.get("attributes"))
            ag = gpt.get((v, oid))
            if not ao or not ag:
                continue
            for k in CANON:
                if k in ao and k in ag:
                    per_attr_match[k][1] += 1
                    if ao[k] == ag[k]:
                        per_attr_match[k][0] += 1
                    pairs[k][0].append(int(ag[k])); pairs[k][1].append(int(ao[k]))
    agree = {k: (per_attr_match[k][0] / per_attr_match[k][1] if per_attr_match[k][1] else float("nan"))
             for k in CANON}
    kappa = {}
    for k in CANON:
        g_, o_ = pairs[k]
        try:
            kappa[k] = cohen_kappa_score(g_, o_) if len(set(g_)) > 1 or len(set(o_)) > 1 else float("nan")
        except Exception:
            kappa[k] = float("nan")
    n = per_attr_match[CANON[0]][1]
    results[name] = {"agree": agree, "kappa": kappa, "n": n}

print(f"{'attribute':20s} " + "  ".join(f"{m:>14s}" for m in MODELS))
for k in CANON:
    line = f"{k:20s} "
    for m in MODELS:
        line += f"  {results[m]['agree'][k]*100:5.1f}% (k={results[m]['kappa'][k]:+.2f})"
    print(line)
print("\nn (paired facades) per model:", {m: results[m]["n"] for m in MODELS})
print("\nmean agreement per model:",
      {m: round(sum(results[m]["agree"].values())/9*100, 1) for m in MODELS})

import json as _j
_j.dump({m: {"agree": results[m]["agree"], "kappa": results[m]["kappa"], "n": results[m]["n"]}
         for m in MODELS}, open("occupancy_figures/attr_agreement.json", "w"), indent=2)
print("wrote occupancy_figures/attr_agreement.json")
