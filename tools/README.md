# tools/ — orientation alignment helper

`align_orientation.ipynb` re-orients a dataset so its heart segmentation matches a known-good
**anchor** volume (`template/acdc_anchor_label.nii.gz`), then writes the corrected NIfTI files in
MorphiNet's directory layout.

## Why this matters
MorphiNet encodes the segmentation as a gradient field that deforms the template mesh — this only
works when the segmentation is in the orientation the pipeline expects. The bundled anchor is already
in that orientation, so **matching a new dataset to the anchor is enough** (you don't need to know the
template frame). A mis-oriented dataset will silently produce wrong reconstructions.

## Requirements
- Run inside the `morphinet` conda env (uses `nibabel`, `torch`, `numpy`, `ipywidgets`, `plotly`).
- Data in MorphiNet label convention — `{1: lv, 2: lv-myo, 3: rv, 4: rv-myo}` — laid out as
  `<dir>/imagesTs/<stem>_0000.nii.gz` and `<dir>/labelsTs/<stem>.nii.gz`.

## How to use
Open the notebook from the repository root and run the cells top to bottom.

1. **Configure** (the *config* cell):
   - `SOURCE_DIR` — your unchanged data to re-orient (defaults to the bundled ACDC example).
   - `IMAGES_SUBDIR` / `LABELS_SUBDIR` / `IMAGE_SUFFIX` — file layout (default `imagesTs` / `labelsTs`,
     `_0000.nii.gz`).
   - `OUTPUT_DIR` — where corrected files are written (defaults to `MORPHINET_ACDC_DATA_DIR` from
     `config.env`).
   - `MYO_LABELS` / `RV_CAVITY` / `LV_CAVITY` — label values (defaults match the convention above).
2. **Run the setup cells** (imports, core functions, overlay). The overlay cell shows a standalone
   interactive 3D view — **drag to rotate, scroll to zoom** — to confirm rendering works.
3. **Align** (the *Align the source to the anchor* cell): an interactive view with controls.
   - Pick a subject from the dropdown.
   - Click flip/swap buttons (`f:x f:y f:z`, `s:xy s:xz s:yz`) to build a sequence; `undo` / `clear`
     to edit, or type directly into the sequence box. **Order matters.**
   - The view re-renders on each apply. Goal: land the **source** markers on the **anchor** markers —
     blue ✕ on red ◆ (RV), cyan ✕ on green ◆ (LV) — with the orange myo cloud over the gray one.
   - The plot's `x / y / z` axes use the same notation as the sequence (`x`↔`D`, `y`↔`W`, `z`↔`H` array
     dims), so `f:x` flips along the plot's x-axis. Spot-check a few subjects — one sequence fits all.
4. **Lock & validate** (the *Lock the sequence* cell): set `FINAL_SEQUENCE` to the string you settled
   on (copy it from the box). The cell checks invertibility + affine consistency and shows a final
   overlay.
5. **Bake** (the *Bake* cell): run `process_dataset(SOURCE_DIR, OUTPUT_DIR, FINAL_SEQUENCE, limit=2)`
   to try two files, then drop `limit` for the full dataset. It writes the re-oriented
   `imagesTs` / `labelsTs` (compensated affines, dtypes and label values preserved) plus an
   `orientation_preprocessing_report.json`.
6. **Wire into the pipeline** (the *Make it pipeline-ready* cell): point `MORPHINET_ACDC_DATA_DIR`
   (`config.env`) at `OUTPUT_DIR`. File names are preserved, so the existing `test` split of
   `dataset/dataset_task21_f0.json` keeps working; the cell also regenerates a split
   (`dataset_task21_f0.generated.json`) for inspection.
7. **Validate a baked subject** (the *Validate* cell): `validate_baked(OUTPUT_DIR, "<stem>")` overlays
   a saved file on the anchor with identity — it should already match.

## Verified example (ACDC)
For the bundled ACDC archive → anchor, the sequence is **`s:xz f:x f:y`** (reproduces the processed
dataset exactly across subjects of differing in-plane size). Re-derive your own with the tool rather
than reusing a string from another dataset.

## How it works
- Transforms are flip/swap signed-permutations reused from `data/utils/geometry.py` (the same
  vocabulary `data.components.SequentialTransformd` uses) — `x`↔`D`, `y`↔`W`, `z`↔`H`.
- The anchor is the orientation reference: source and anchor are compared in a shared per-axis-
  normalized `x / y / z` frame, so only orientation differences are visible.
- Baking applies the sequence to image + label and compensates the affine (`A → A · P⁻¹`) so world
  coordinates are preserved; label values and dtypes are unchanged.
- The tool only matches a source to the anchor's on-disk orientation; it does not touch any
  pipeline-side processing.
