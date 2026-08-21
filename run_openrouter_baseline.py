#!/usr/bin/env python
"""
Open-source VLM baseline for Recov-Vision via OpenRouter (OpenAI-compatible API).

Reproduces the paper's two-stage / one-stage occupancy pipeline with the model
swapped from GPT-4o to an open-weights VLM (default: qwen/qwen3-vl-235b-a22b-instruct).
Everything else is held fixed: same rectified facades (FacadeTrack), the canonical
9-attribute snake_case schema, the appendix few-shot decision prompt, and the paper's
tau=2 one-stage rule.

Usage:
  python run_openrouter_baseline.py --visit 1 --limit 3 --smoke      # cheap smoke test
  python run_openrouter_baseline.py --visit both                     # full run
  python run_openrouter_baseline.py --eval-only                      # metrics from saved CSVs
"""
import argparse, base64, json, os, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMG_DIR = {1: ROOT / "data/FacadeTrack/Visit1_processed_images",
           2: ROOT / "data/FacadeTrack/Visit2_processed_images"}
RESULT_CSV = {1: ROOT / "Data_figures/FieldTrip1_occupancy_wo_LLM_result.csv",
              2: ROOT / "Data_figures/June25_occupancy_wo_LLM_result.csv"}
LABEL_COL = {1: "label", 2: "Ground Truth"}
ID_COL = {1: "objectid", 2: "objectid"}
OUTDIR = ROOT / "rebuttal_out"

def model_tag(model):
    s = model.lower()
    for t in ("qwen", "gemma", "internvl", "glm", "pixtral", "llama"):
        if t in s:
            return t
    return re.sub(r"[^a-z0-9]+", "-", s.split("/")[-1])[:24]

def out_path(tag, visit):
    return OUTDIR / f"{tag}_visit{visit}.csv"

NINE = ["house_destruction", "structural_damage", "exterior_debris", "open_doors_windows",
        "site_accessible", "exterior_mud", "emergency_markings", "major_repairs", "vehicle_presence"]

VISION_PROMPT = (
    "Analyze the image and answer with a JSON object using EXACTLY these keys and boolean "
    "values (true/false). Do not add, remove, or rename keys. Return only the JSON object, "
    "no prose.\n\n{\n" + ",\n".join(f'  "{k}": true/false' for k in NINE) + "\n}"
)

VISION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "facade_attributes", "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "properties": {k: {"type": "boolean"} for k in NINE},
                   "required": NINE}},
}

# Canonical few-shot decision examples (snake_case), matching appendix Listing lst:decision-fewshot.
_FEWSHOT = [
    ({"site_accessible": True, "vehicle_presence": True}, "Occupied"),                         # Ex1
    ({"house_destruction": True, "structural_damage": True, "exterior_debris": True,
      "site_accessible": True}, "Not Occupied"),                                               # Ex2
    ({"structural_damage": True, "exterior_debris": True, "open_doors_windows": True,
      "site_accessible": True, "vehicle_presence": True}, "Not Occupied"),                     # Ex3
    ({"structural_damage": True, "site_accessible": True, "major_repairs": True}, "Not Occupied"),  # Ex4
    ({"site_accessible": True, "emergency_markings": True}, "Not Occupied"),                    # Ex5
    ({"site_accessible": True}, "Occupied"),                                                    # Ex6
    ({"house_destruction": True, "site_accessible": True}, "Not Occupied"),                     # Ex7
    ({"site_accessible": False}, "Not Occupied"),                                               # Ex8
    ({"site_accessible": True, "exterior_mud": True, "vehicle_presence": True}, "Occupied"),    # Ex9
]

def _full(attrs):  # fill all 9 keys, default False
    return {k: bool(attrs.get(k, False)) for k in NINE}

def build_decision_prompt(vision_json_str):
    head = (
        "You are an expert in post-disaster building occupancy assessment. Given a building's "
        "attributes in JSON (keys listed below), decide if it is 'Occupied' or 'Not Occupied'.\n\n"
        "Consider all evidence: for example, some exterior mud may coexist with occupancy; parked "
        "vehicles can indicate occupancy when other signs are mixed; extensive roof or wall repairs "
        "may indicate temporary non-occupancy. If the evidence is mixed or unclear, prefer 'Not "
        "Occupied'. Be conservative: if there is significant indication that a building might not be "
        "occupied (e.g., destruction, inaccessibility, visible abandonment, or multiple risk "
        "factors), classify it as 'Not Occupied'. Classify as 'Occupied' only if the building "
        "appears livable and there are no clear signs of uninhabitability.\n\n"
        "The JSON fields you will receive:\n" + json.dumps({k: "bool" for k in NINE}, indent=2) +
        "\n\nHere are some examples:\n"
    )
    body = ""
    for i, (a, lab) in enumerate(_FEWSHOT, 1):
        body += f"\nExample {i}:\n{json.dumps(_full(a), indent=2)}\n{lab}\n"
    tail = f"\nNow, decide for this building:\n{vision_json_str}\n\nOutput only one token: 'Occupied' or 'Not Occupied'."
    return head + body + tail


# ---------- one-stage deterministic rule (paper Eq. / Table tab:baseline-threshold) ----------
RISK = ["house_destruction", "structural_damage", "exterior_debris", "open_doors_windows",
        "exterior_mud", "emergency_markings", "major_repairs"]  # + site_accessible==False

def one_stage(attrs, tau=2):
    a = _full(attrs)
    r = sum(a[k] for k in RISK) + (0 if a["site_accessible"] else 1)  # 8 risk indicators
    v = 1 if a["vehicle_presence"] else 0
    return "Not Occupied" if (r - v) >= tau else "Occupied"


# ---------- OpenRouter client ----------
def load_key():
    # Prefer the environment variable; fall back to a local .env if present.
    env = os.environ.get("OPENROUTER_API_KEY")
    if env:
        return env
    envfile = ROOT / ".env"
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

def make_client():
    from openai import OpenAI
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=load_key())

REQUIRE_PARAMS = True  # some models (e.g. GLM-4.6V) have no endpoint that advertises every param; disable via --no-require-params

def _provider_pref(provider):
    pref = {"require_parameters": True} if REQUIRE_PARAMS else {}
    if provider:
        pref.update({"order": [provider], "allow_fallbacks": False})
    return pref

def _extra(provider, reasoning_off=False):
    e = {"provider": _provider_pref(provider)}
    if reasoning_off:
        e["reasoning"] = {"enabled": False}
    return e

def run_tag(model, reasoning_off=False):
    return model_tag(model) + ("-nothink" if reasoning_off else "")

def _parse_json(txt):
    txt = txt.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt); txt = txt.rstrip("`").strip()
    if not txt.startswith("{"):
        s, e = txt.find("{"), txt.rfind("}")
        if s != -1 and e != -1:
            txt = txt[s:e + 1]
    return json.loads(txt)

def call_with_retry(client, **kw):
    last = None
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kw)
        except Exception as e:
            last = e
            if "403" in str(e) or "limit exceeded" in str(e).lower():
                raise  # billing/key-limit errors will not recover by retrying
            time.sleep(2 * (attempt + 1))
    raise last

def extract(client, model, image_path, provider, use_schema=True, max_tokens=200, reasoning_off=False, temp=0.0, seed=None):
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    kw = dict(model=model, temperature=temp, max_tokens=max_tokens,
              messages=[{"role": "system", "content": "You output only strict JSON."},
                        {"role": "user", "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
              extra_body=_extra(provider, reasoning_off))
    if use_schema:
        kw["response_format"] = VISION_SCHEMA
    if seed is not None:
        kw["seed"] = seed
    resp = call_with_retry(client, **kw)
    prov = getattr(resp, "provider", None) or (getattr(resp, "model_extra", {}) or {}).get("provider")
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError(f"empty content (finish={resp.choices[0].finish_reason})")
    return _parse_json(content), prov, content

def decide_two_stage(client, model, vision_json_str, provider, max_tokens=5, reasoning_off=False, temp=0.0, seed=None):
    kw = dict(model=model, temperature=temp, max_tokens=max_tokens,
        messages=[{"role": "system", "content": "You are an expert in post-disaster building occupancy "
                   "assessment. Output only one token: Occupied or Not Occupied."},
                  {"role": "user", "content": build_decision_prompt(vision_json_str)}],
        extra_body=_extra(provider, reasoning_off))
    if seed is not None:
        kw["seed"] = seed
    resp = call_with_retry(client, **kw)
    out = (resp.choices[0].message.content or "").strip()
    if not out:
        raise RuntimeError(f"empty decision (finish={resp.choices[0].finish_reason})")
    return "Not Occupied" if "not occupied" in out.lower() else ("Occupied" if "occupied" in out.lower() else out)


def resolve_image(visit, objid):
    stem = str(objid).strip()
    if not stem.lower().endswith(".jpg"):
        stem += ".jpg"
    p = IMG_DIR[visit] / stem
    return p if p.exists() else None


def run_visit(visit, limit=None, smoke=False, model="qwen/qwen3-vl-235b-a22b-instruct",
              provider=None, use_schema=True, workers=6, mtv=200, mtd=5, reasoning_off=False,
              temp=0.0, seed=None, tag=None):
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    tag = tag or run_tag(model, reasoning_off)
    c1, c2 = f"{tag}_one_stage", f"{tag}_two_stage"
    df = pd.read_csv(RESULT_CSV[visit])
    lc, ic = LABEL_COL[visit], ID_COL[visit]
    df = df[df[lc].notna() & df[lc].isin(["Occupied", "Not Occupied"])].copy()
    if limit:
        df = df.head(limit)
    client = make_client()
    t0 = time.time()

    def process(item):
        i, row = item
        objid = row[ic]
        img = resolve_image(visit, objid)
        rec = {"objectid": objid, "ground_truth": row[lc],
               "gpt4o_two_stage": row.get("Pred LLM"), "gpt4o_one_stage": row.get("Pred_wo_LLM")}
        if img is None:
            rec["error"] = "image_missing"; return i, rec
        try:
            attrs, prov, raw = extract(client, model, img, provider, use_schema, max_tokens=mtv, reasoning_off=reasoning_off, temp=temp, seed=seed)
            rec["attributes"] = json.dumps(_full(attrs))
            rec["provider"] = prov
            rec[c1] = one_stage(attrs)
            rec[c2] = decide_two_stage(client, model, json.dumps(_full(attrs)), provider, max_tokens=mtd, reasoning_off=reasoning_off, temp=temp, seed=seed)
        except Exception as e:
            rec["error"] = str(e)[:200]
        return i, rec

    items = list(enumerate([r for _, r in df.iterrows()]))
    results = {}
    if smoke or workers <= 1:
        for it in items:
            i, rec = process(it); results[i] = rec
            print(f"[{i+1}] obj={rec['objectid']} gt={rec['ground_truth']} prov={rec.get('provider')} "
                  f"1stage={rec.get(c1)} 2stage={rec.get(c2)} "
                  f"attrs={rec.get('attributes')} err={rec.get('error','')}")
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, rec in ex.map(process, items):
                results[i] = rec; done += 1
                if done % 25 == 0:
                    print(f"  visit{visit}: {done}/{len(items)}  ({time.time()-t0:.0f}s)", flush=True)
    rows = [results[i] for i in range(len(items))]
    out = pd.DataFrame(rows)
    outp = out_path(tag, visit)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False)
    print(f"visit{visit} [{tag}]: wrote {len(out)} rows -> {outp}  "
          f"(errors={out['error'].notna().sum() if 'error' in out else 0})")
    return out


def metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
    P = "Not Occupied"
    return dict(n=len(y_true),
                acc=accuracy_score(y_true, y_pred),
                prec=precision_score(y_true, y_pred, pos_label=P, zero_division=0),
                rec=recall_score(y_true, y_pred, pos_label=P, zero_division=0),
                f1=f1_score(y_true, y_pred, pos_label=P, zero_division=0),
                kappa=cohen_kappa_score(y_true, y_pred))

def evaluate(tag="qwen", label=None):
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score
    label = (label or tag)
    c1, c2 = f"{tag}_one_stage", f"{tag}_two_stage"
    for visit in (1, 2):
        p = out_path(tag, visit)
        if not p.exists():
            print(f"visit{visit} [{tag}]: no output yet"); continue
        d = pd.read_csv(p)
        if c2 in d.columns:
            d = d[d[c2].notna()]
        print(f"\n===== Visit {visit}  ({label}, n={len(d)}) =====")
        gt = d["ground_truth"]
        print(" GPT-4o   one-stage :", {k: round(v, 3) for k, v in metrics(gt, d["gpt4o_one_stage"]).items()})
        print(" GPT-4o   two-stage :", {k: round(v, 3) for k, v in metrics(gt, d["gpt4o_two_stage"]).items()})
        print(f" {label:8s} one-stage :", {k: round(v, 3) for k, v in metrics(gt, d[c1]).items()})
        print(f" {label:8s} two-stage :", {k: round(v, 3) for k, v in metrics(gt, d[c2]).items()})
        print(f" cross-model kappa (GPT-4o vs {label}, two-stage):",
              round(cohen_kappa_score(d["gpt4o_two_stage"], d[c2]), 3))


def aggregate_seeds(prefix):
    """Aggregate {prefix}-s*_visit{1,2}.csv across seeds: mean and 95% CI per metric."""
    import pandas as pd, glob, re
    from sklearn.metrics import cohen_kappa_score
    files = glob.glob(f"rebuttal_out/{prefix}-s*_visit1.csv")
    seeds = sorted({m.group(1) for p in files for m in [re.search(r"-s(\w+)_visit1\.csv$", p)] if m})
    if not seeds:
        print("no seed files for prefix", prefix); return
    per = {"one": [], "two": []}; xkappa = []; nlast = 0; used = []
    for s in seeds:
        tag = f"{prefix}-s{s}"; c1, c2 = f"{tag}_one_stage", f"{tag}_two_stage"
        d = pd.concat([pd.read_csv(out_path(tag, v)) for v in (1, 2)], ignore_index=True)
        if c2 not in d.columns or d[c2].notna().sum() < 50:
            print(f"  skip seed {s}: no usable predictions (run failed?)"); continue
        d = d[d[c2].notna() & d["gpt4o_two_stage"].notna()]; nlast = len(d)
        gt = d["ground_truth"]
        per["one"].append(metrics(gt, d[c1])); per["two"].append(metrics(gt, d[c2]))
        xkappa.append(cohen_kappa_score(d["gpt4o_two_stage"], d[c2])); used.append(s)
    if not per["two"]:
        print(f"no usable seeds for '{prefix}'"); return

    def summarize(rows):
        out = {}
        for k in [k for k in rows[0] if k != "n"]:
            vals = [r[k] for r in rows]; m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
            ci = 4.303 * sd / (len(vals) ** 0.5)  # t(2, 0.975) for n=3 seeds
            out[k] = f"{m:.3f}+/-{ci:.3f}"
        return out
    print(f"=== aggregate '{prefix}': usable seeds={used}, n={nlast} ===")
    print(" one-stage (mean +/- 95% CI):", summarize(per["one"]))
    print(" two-stage (mean +/- 95% CI):", summarize(per["two"]))
    print(f" cross-model kappa vs GPT-4o (two-stage): {sum(xkappa)/len(xkappa):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--visit", default="both", choices=["1", "2", "both"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct")
    ap.add_argument("--provider", default=None, help="pin a single OpenRouter provider slug")
    ap.add_argument("--no-schema", action="store_true", help="instruction-only JSON (no response_format)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--mt-vision", type=int, default=200, dest="mt_vision")
    ap.add_argument("--mt-decision", type=int, default=5, dest="mt_decision")
    ap.add_argument("--no-reasoning", action="store_true", dest="no_reasoning")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default=None, help="explicit output tag (overrides model-derived tag)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--aggregate", default=None, help="aggregate {PREFIX}-s*_visit*.csv across seeds")
    ap.add_argument("--no-require-params", action="store_true", dest="no_require_params",
                    help="drop provider require_parameters (needed for GLM-4.6V + json_schema)")
    a = ap.parse_args()
    if a.no_require_params:
        globals()["REQUIRE_PARAMS"] = False
    if a.aggregate:
        aggregate_seeds(a.aggregate); sys.exit(0)
    tag = a.tag or run_tag(a.model, a.no_reasoning)
    if a.eval_only:
        evaluate(tag); sys.exit(0)
    visits = [1, 2] if a.visit == "both" else [int(a.visit)]
    for v in visits:
        run_visit(v, limit=a.limit, smoke=a.smoke, model=a.model,
                  provider=a.provider, use_schema=not a.no_schema, workers=a.workers,
                  mtv=a.mt_vision, mtd=a.mt_decision, reasoning_off=a.no_reasoning,
                  temp=a.temp, seed=a.seed, tag=a.tag)
    if not a.smoke:
        evaluate(tag)
