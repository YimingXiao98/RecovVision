# FacadeTrack imagery

The rectified facade images are **not** bundled here (they are large and are
released separately). Download the blurred, public FacadeTrack dataset from
HuggingFace and place the two image folders here:

```
data/FacadeTrack/
├── Visit1_processed_images/   # <objectid>.jpg
└── Visit2_processed_images/   # <objectid>.jpg
```

HuggingFace dataset: `Ymx1025/FacadeTrack`
(all faces and license plates are blurred in the released images).

```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Ymx1025/FacadeTrack", repo_type="dataset",
                  local_dir="data/FacadeTrack")
PY
```

The image `<objectid>` matches the `objectid` column in
`Data_figures/FieldTrip1_occupancy_wo_LLM_result.csv` (Visit 1) and
`June25_occupancy_wo_LLM_result.csv` (Visit 2).

**Note:** the per-parcel vision outputs and predictions needed to regenerate the
paper's tables/figures are already included (`Data_figures/*.csv`,
`rebuttal_out/*.csv`), so you can reproduce those **without** downloading imagery.
You only need the images to (a) re-run the VLM extraction from scratch or (b)
regenerate the example-facade figure (Figure 7).
