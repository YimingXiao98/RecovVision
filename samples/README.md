# Samples

This folder provides minimal CSV examples and empty image directories to illustrate expected formats and directory structure for the pipeline.

## Folders

- `images/frames_output/`: Where raw frames (extracted from videos) would be stored as `<ObjectId>.jpg`.
- `images/frames_dewarped/`: Where dewarped frames would be written.

These are left empty here (place your own `.jpg` files). The pipeline expects images named exactly `<ObjectId>.jpg`.

## CSVs

- `csv/sample_footprints.csv`: Minimal building footprint inputs (used by spatial match).
- `csv/sample_matches.csv`: Example output of spatial matching (input to extraction/orientation).
- `csv/sample_matches_orientation.csv`: Example with an `orientation` column (input to dewarping and VLM).
- `csv/sample_ground_truth.csv`: Ground truth labels for the VLM pipeline evaluation.

These files use tiny, fake values solely to demonstrate schema.

