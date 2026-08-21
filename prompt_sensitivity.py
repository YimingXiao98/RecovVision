#!/usr/bin/env python
"""Decision-stage prompt-sensitivity experiment (addresses R3-M4) + a cost/latency
benchmark (addresses R1-8). Decision-stage only: reuses the STORED GPT-4o vision
attributes (never re-runs vision), and re-runs only the text-only GPT-4o decision under
prompt perturbations. The baseline config reproduces the paper's two-stage GPT-4o labels.

Modes:
  python prompt_sensitivity.py --sensitivity --visit both     # the experiment (~$9, ~25 min)
  python prompt_sensitivity.py --benchmark --n 40             # cost/latency on full pipeline (~$1)
  python prompt_sensitivity.py --summarize                    # metrics table from saved CSV (no API)
"""
import argparse, json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import run_openrouter_baseline as base   # reuse client, helpers, exemplars, metrics

ROOT = base.ROOT
NINE = base.NINE
MODEL = "openai/gpt-4o"
OUTDIR = ROOT / "rebuttal_out"; OUTDIR.mkdir(exist_ok=True)
SENS_CSV = OUTDIR / "prompt_sensitivity.csv"

# --- stored GPT-4o attributes: verbose key -> (canonical, invert?) ; site_accessible = NOT inaccessible
G2C = {
    "house destroyed": ("house_destruction", False),
    "house damaged": ("structural_damage", False),
    "debris outside house": ("exterior_debris", False),
    "doors or windows open": ("open_doors_windows", False),
    "building inaccessible": ("site_accessible", True),
    "is there large area of mud outside,": ("exterior_mud", False),
    "is there a red cross or zero written on the building": ("emergency_markings", False),
    "major repair work going on around the house (Wall pannels or Roof)": ("major_repairs", False),
    "are there any cars parked right outside the concerned house (not the black car rooftop partially visible)": ("vehicle_presence", False),
}

def parse_gpt_attrs(s):
    try:
        j = json.loads(re.sub(r"```json|```", "", str(s)).strip())
    except Exception:
        return None
    out = {}
    for vk, (ck, inv) in G2C.items():
        if vk in j:
            out[ck] = (not bool(j[vk])) if inv else bool(j[vk])
    return out if len(out) == 9 else None

# --- instruction heads (original vs faithful paraphrase) ---
FIELDS = json.dumps({k: "bool" for k in NINE}, indent=2)
HEAD_ORIG = (
    "You are an expert in post-disaster building occupancy assessment. Given a building's "
    "attributes in JSON (keys listed below), decide if it is 'Occupied' or 'Not Occupied'.\n\n"
    "Consider all evidence: for example, some exterior mud may coexist with occupancy; parked "
    "vehicles can indicate occupancy when other signs are mixed; extensive roof or wall repairs "
    "may indicate temporary non-occupancy. If the evidence is mixed or unclear, prefer 'Not "
    "Occupied'. Be conservative: if there is significant indication that a building might not be "
    "occupied (e.g., destruction, inaccessibility, visible abandonment, or multiple risk "
    "factors), classify it as 'Not Occupied'. Classify as 'Occupied' only if the building "
    "appears livable and there are no clear signs of uninhabitability.\n\n"
    "The JSON fields you will receive:\n" + FIELDS)
HEAD_PARA = (
    "You assess post-disaster building occupancy. From the building's attribute JSON below, "
    "output whether it is 'Occupied' or 'Not Occupied'.\n\n"
    "Weigh all of the evidence together: mud alone can coexist with occupancy, and a parked "
    "vehicle can suggest occupancy when signals are mixed, whereas major roof or wall repairs "
    "can signal temporary vacancy. When the evidence is ambiguous, default to 'Not Occupied'. "
    "Apply a conservative bias: any clear indication of non-occupancy (destruction, "
    "inaccessibility, abandonment, or several risk factors together) means 'Not Occupied'. "
    "Choose 'Occupied' only when the building looks usable with no clear signs that it cannot "
    "be lived in.\n\nThe JSON fields provided:\n" + FIELDS)

# fixed (deterministic) exemplar orderings over the 9 examples
PERMS = {
    "baseline": list(range(9)),
    "order_rev": list(range(8, -1, -1)),
    "order_a": [4, 0, 7, 2, 8, 1, 5, 3, 6],
    "order_b": [2, 5, 1, 8, 3, 6, 0, 4, 7],
}
CONFIGS = (["baseline", "order_rev", "order_a", "order_b"]   # exemplar-order perturbations
           + ["paraphrase"])                                  # instruction reword (baseline order)

def build_prompt(vision_json_str, order, head):
    ex = [base._FEWSHOT[i] for i in order]
    body = "".join(f"\nExample {i}:\n{json.dumps(base._full(a), indent=2)}\n{lab}\n"
                   for i, (a, lab) in enumerate(ex, 1))
    tail = f"\nNow, decide for this building:\n{vision_json_str}\n\nOutput only one token: 'Occupied' or 'Not Occupied'."
    return head + "\n\nHere are some examples:\n" + body + tail

def decide(client, prompt):
    resp = base.call_with_retry(client, model=MODEL, temperature=0, max_tokens=5,
        messages=[{"role": "system", "content": "You are an expert in post-disaster building "
                   "occupancy assessment. Output only one token: Occupied or Not Occupied."},
                  {"role": "user", "content": prompt}],
        extra_body=base._extra(None))
    out = (resp.choices[0].message.content or "").strip()
    lab = "Not Occupied" if "not occupied" in out.lower() else ("Occupied" if "occupied" in out.lower() else out)
    u = resp.usage
    return lab, (getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))

def load_facades():
    import pandas as pd
    rows = []
    for v in (1, 2):
        d = pd.read_csv(base.RESULT_CSV[v]); lc = base.LABEL_COL[v]
        d = d[d[lc].isin(["Occupied", "Not Occupied"])]
        for _, r in d.iterrows():
            a = parse_gpt_attrs(r["Vision Model Output"])
            if a:
                rows.append({"visit": v, "objectid": str(r[base.ID_COL[v]]).strip(),
                             "ground_truth": r[lc], "gpt4o_two_stage": r.get("Pred LLM"),
                             "attrs": json.dumps(base._full(a))})
    return rows

def run_sensitivity(visits, workers=6):
    import pandas as pd
    facades = [f for f in load_facades() if f["visit"] in visits]
    client = base.make_client()
    tok = [0, 0]; t0 = time.time()
    def work(f):
        rec = dict(f)
        for cfg in CONFIGS:
            order = PERMS["baseline"] if cfg == "paraphrase" else PERMS[cfg]
            head = HEAD_PARA if cfg == "paraphrase" else HEAD_ORIG
            try:
                lab, (pt, ct) = decide(client, build_prompt(f["attrs"], order, head))
                rec[cfg] = lab; tok[0] += pt; tok[1] += ct
            except Exception as e:
                rec[cfg] = None; rec.setdefault("error", str(e)[:120])
        return rec
    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, rec in enumerate(ex.map(work, facades), 1):
            out.append(rec)
            if i % 50 == 0:
                print(f"  {i}/{len(facades)} facades x {len(CONFIGS)} configs  ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(out).to_csv(SENS_CSV, index=False)
    print(f"wrote {len(out)} rows -> {SENS_CSV}  | decision tokens in/out = {tok[0]}/{tok[1]}  "
          f"| elapsed {time.time()-t0:.0f}s")
    summarize()

def summarize():
    import pandas as pd
    d = pd.read_csv(SENS_CSV)
    base_lab = d["baseline"]
    # reproduction check vs stored paper two-stage labels
    rep = (d["baseline"] == d["gpt4o_two_stage"]).mean()
    print(f"\nBaseline reproduces stored GPT-4o two-stage labels on {rep*100:.1f}% of {len(d)} facades")
    print(f"{'config':12s} {'agree_vs_base':>13s} {'NotOcc_recall':>13s} {'F1':>7s} {'kappa':>7s}")
    gt = d["ground_truth"]
    for cfg in CONFIGS:
        sub = d[d[cfg].notna()]
        agree = (sub[cfg] == sub["baseline"]).mean()
        m = base.metrics(sub["ground_truth"], sub[cfg])
        print(f"{cfg:12s} {agree*100:12.1f}% {m['rec']:13.3f} {m['f1']:7.3f} {m['kappa']:7.3f}")

def run_benchmark(n=40):
    """Full pipeline (GPT-4o vision + decision) on n facades; measure latency + token cost."""
    import pandas as pd
    facs = []
    for v in (1, 2):
        d = pd.read_csv(base.RESULT_CSV[v]); lc = base.LABEL_COL[v]
        d = d[d[lc].isin(["Occupied", "Not Occupied"])].head(n // 2)
        for _, r in d.iterrows():
            img = base.resolve_image(v, r[base.ID_COL[v]])
            if img: facs.append(img)
    client = base.make_client(); lat = []; tin = tout = 0
    for i, img in enumerate(facs, 1):
        t = time.time()
        attrs, _, _ = base.extract(client, MODEL, img, None, use_schema=False, max_tokens=200)
        _ = decide(client, build_prompt(json.dumps(base._full(attrs)), PERMS["baseline"], HEAD_ORIG))
        lat.append(time.time() - t)
        print(f"  {i}/{len(facs)}  {lat[-1]:.2f}s", flush=True)
    lat.sort(); med = lat[len(lat)//2]
    print(f"\nBENCHMARK on {len(facs)} facades (full vision+decision, GPT-4o via OpenRouter):")
    print(f"  per-parcel latency: median {med:.2f}s, mean {sum(lat)/len(lat):.2f}s, "
          f"range {lat[0]:.2f}-{lat[-1]:.2f}s")
    print(f"  (token totals printed by OpenRouter usage; check dashboard for $ at run date)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--visit", default="both", choices=["1", "2", "both"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.summarize: summarize()
    elif a.benchmark: run_benchmark(a.n)
    elif a.sensitivity:
        run_sensitivity([1, 2] if a.visit == "both" else [int(a.visit)], a.workers)
    else: ap.print_help()
