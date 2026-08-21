# Recov-Vision — reproducibility release

Code and per-parcel data to reproduce the results in *Recov-Vision: Linking
Street View Imagery and Vision-Language Models for Post-Disaster Recovery*,
accepted at the *International Journal of Disaster Risk Reduction (IJDRR)*.

The pipeline links panoramic street-view video to parcels, rectifies each view
to a facade, extracts nine boolean attributes with a vision-language model, and
maps those attributes to an occupancy label with either a transparent one-stage
count rule or a two-stage text-only decision. This release lets you reproduce
the open-weight model comparison, the prompt-sensitivity analysis, the
inter-annotator reliability check, and the paper's figures — most of them
**offline**, from the included per-parcel outputs.

---

## What's included

```
run_openrouter_baseline.py   Open-weight reproduction of the full pipeline      -> Table 9
prompt_sensitivity.py        Decision-prompt perturbation experiment            -> Table 10
annotator_study.py           Independent-annotator reliability (Cohen's kappa)  -> Section 3.2
cli.py                       Entry point for the geometry+VLM pipeline
vrtoolkit/                   Pipeline package (matching, extraction, rectify, VLM)
prompts/                     The exact appendix prompts (extraction + few-shot decision)
Data_figures/                Figure/table generators + stored GPT-4o per-parcel CSVs
  make_open_baselines_f1.py     -> Figure 11 (minority-class F1 by backend)
  make_attr_agreement_fig.py    -> Figure 12 (per-attribute agreement)
  attr_agreement.py             -> Figure 12 data (per-attribute agreement vs GPT-4o)
  make_error_merged.py          -> Figure 8 (class-stratified error analysis)
  make_error_by_class.py        -> Figure 8 variant
  make_tp_fp_examples.py        -> Figure 7 (needs imagery)
rebuttal_out/                Per-parcel outputs for every open-weight run + sensitivity + IAA
data/FacadeTrack/            Where to place the (separately released) blurred imagery
samples/                     Tiny sample CSVs to smoke-test the pipeline
```

No API keys are committed. The imagery is released separately on HuggingFace
(`Ymx1025/FacadeTrack`); see `data/FacadeTrack/README.md`.

---

## Install

```bash
python -m venv .venv && . .venv/bin/activate      # Python 3.9+
pip install -r requirements.txt
# system: FFmpeg (only needed to extract frames from raw video)
#   apt-get install ffmpeg   |   brew install ffmpeg
cp .env.example .env         # then edit; or export the vars in your shell
```

Keys (only needed to *re-run* the models; the analyses below run offline from
the bundled CSVs):
- `OPENROUTER_API_KEY` — open-weight runs (`run_openrouter_baseline.py`, `prompt_sensitivity.py`)
- `OPENAI_API_KEY` — original GPT-4o pipeline (`vrtoolkit` VLM step)

---

## Reproduce the paper — offline (from included per-parcel CSVs)

These need **no API key and no imagery**:

```bash
# Table 9 — open-weight backends, one-stage vs two-stage, both visits pooled
python run_openrouter_baseline.py --eval-only --model qwen/qwen3-vl-235b-a22b-instruct
python run_openrouter_baseline.py --eval-only --model qwen/qwen3.5-397b-a17b   --tag qwen35-nothink
python run_openrouter_baseline.py --eval-only --model z-ai/glm-4.6v            --tag glm46v-nothink

# Table 10 — decision-prompt sensitivity (baseline vs 3 exemplar orders + 1 paraphrase)
python prompt_sensitivity.py --summarize

# Section 3.2 — inter-annotator reliability (Cohen's kappa = 0.70)
python annotator_study.py --score --labels rebuttal_out/labels_selina.csv

# Figures 8, 11, 12 (run from Data_figures/; PNGs land in Data_figures/occupancy_figures/)
cd Data_figures
python make_open_baselines_f1.py     # Figure 11
python attr_agreement.py             # Figure 12 data
python make_attr_agreement_fig.py    # Figure 12
python make_error_merged.py          # Figure 8
cd ..
```

(Figure 7, `make_tp_fp_examples.py`, needs the facade images — see below.)

## Reproduce from scratch — re-run the models (needs API + imagery)

```bash
# download the blurred imagery first (see data/FacadeTrack/README.md), then:
python run_openrouter_baseline.py --visit both --model qwen/qwen3-vl-235b-a22b-instruct
python run_openrouter_baseline.py --visit both --model z-ai/glm-4.6v --tag glm46v-nothink --no-reasoning
python prompt_sensitivity.py --sensitivity --visit both      # rewrites rebuttal_out/prompt_sensitivity.csv
```
Add `--limit 3 --smoke` for a cheap first check on any model.

## Full pipeline from raw video (GPT-4o path)

```bash
python -m cli match     footprints.csv GPS_FOLDER   matched.csv
python -m cli extract   matched.csv    VIDEO_ROOT   frames_output
python -m cli orient    matched.csv    GPS_FOLDER   matched_orientation.csv
python -m cli dewarp    matched_orientation.csv frames_output frames_dewarped \
       --h_fov 90 --pitch 0 --width 1920 --aspect 16:9 --yaw_offset -90
python -m cli vlm       matched_orientation.csv frames_dewarped occupancy.csv \
       --gt_csv ground_truth.csv          # add --no-llm for the one-stage rule
```
A self-contained demo on the bundled samples:
```bash
python -m scripts.generate_sample_images --csv samples/csv/sample_matches_orientation.csv \
       --id-col ObjectId --out samples/images/frames_dewarped
python -m cli vlm samples/csv/sample_matches_orientation.csv samples/images/frames_dewarped \
       out/sample_predictions.csv --gt_csv samples/csv/sample_ground_truth.csv --image_id_column ObjectId
```

---

## Prompts and decision rule

- `prompts/vision_extraction.txt` — the canonical 9-key attribute schema (Appendix Listing 1).
- `prompts/decision_fewshot.txt` — the two-stage decision prompt with 9 exemplars (Listing 3).
- One-stage rule (`vrtoolkit.VLMpipeline.decide_occupancy`, `run_openrouter_baseline.one_stage`):
  predict **Not Occupied** iff `r - v >= 2`, where `r` is the number of true risk
  indicators (the 7 damage/condition cues plus `site_accessible == False`) and
  `v = 1` if a vehicle is present (Table 1 / Eq. 3).

The open-weight backends and the `vrtoolkit` GPT-4o path both use this canonical
snake_case schema, so the comparison is prompt-matched. The **original** GPT-4o
run phrased each attribute as a natural-language question and excluded the survey
vehicle in its vehicle attribute; that single wording difference is documented in
Section 4.3 and quantified in Figure 12 (`vehicle_presence`).

---

## Notes

- **API drift & determinism.** All runs use temperature 0. Re-running the GPT-4o
  baseline against a current snapshot reproduces ~93% of the original stored
  labels; this API drift is exactly why the open-weight results are the
  reproducible reference. temp-0.6 reasoning runs use seeds (best-effort across
  providers) and are reported with across-seed variability.
- **Fonts.** The figure scripts request the Arimo font (an Arial-metric clone)
  for consistency; if it isn't installed, matplotlib falls back automatically —
  cosmetic only.
- **`rebuttal_out/` also contains** extra runs not in the main tables (Gemma,
  Kimi, and the temp-0.6 reasoning seeds `*-think-s{1,2,3}`); the `--aggregate
  PREFIX` flag of `run_openrouter_baseline.py` averages the seed runs.

## Security

Set keys via environment variables or a local `.env` (git-ignored). Do not commit
keys. If a key was ever exposed, rotate it at the provider console.

## Citation

If you use this code or data, please cite:

```bibtex
@misc{xiao2026recovvisionlinkingstreetview,
      title={Recov-Vision: Linking Street View Imagery and Vision-Language Models for Post-Disaster Recovery},
      author={Yiming Xiao and Archit Gupta and Miguel Esparza and Yu-Hsuan Ho and Antonia Sebastian and Hannah Weas and Rose Houck and Ali Mostafavi},
      year={2026},
      eprint={2509.20628},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.20628},
}
```
