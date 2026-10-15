# Data preparation

For environment setup, see the [project README](../readme.md). Run the commands below from the repository root.

## Data and storage

Use the same dataset and fixed split file across the team. The shared development sample belongs in `dataset/rsna_dev_100/`. The full dataset on the lab server belongs in `dataset/rsna_full/`, under `/amax/liming/projects/RBC_Letssolveit_LungStoryShort/`.

Both use the official RSNA adjudicated pneumonia dataset's **Calculated** labels. The 100-image sample is for development and debugging; do not add it to the full dataset as extra training images. See the [sample README](rsna_dev_100/README.md) for source references, terms, and attribution.

```text
dataset/
    rsna_dev_100/
        images/                   # Original DICOM files
        sample_ids.csv            # Image IDs, patient IDs, paths, and checksums
        labels.csv                # Original targets and boxes
        detailed_class_info.csv   # Original class labels
        splits.csv                # Shared fixed splits
    rsna_full/
        images/                   # 30,000 extracted original DICOM files
        source_metadata/          # Official annotation and mapping JSON files
        metadata/                 # Converted CSVs and fixed splits.csv
        prepared/                 # Generated training inputs and review reports
```

The full conversion includes 29,684 labelled images and 11,390 boxes. It excludes 316 images without Calculated annotations; these are not negative examples. Classes are 9,790 Normal, 12,788 No Lung Opacity / Not Normal, and 7,106 Lung Opacity. `Target=0` does not necessarily mean normal. Membership in the original Kaggle Stage 2 splits has not been verified.

## One preparation entry point

Run `dataset/scripts/prepare_dataset.py` for splitting, image and box processing, reports, and representative preview generation. It reuses existing splits unchanged, or creates missing splits with the fixed procedure below.

The output directory must be new or empty. Choose a different output directory for another run. There is no resume mode. Existing images, annotations, split assignments, and previous results are not overwritten.

### Development sample

The sample already contains its CSV inputs and fixed splits:

```sh
python dataset/scripts/prepare_dataset.py \
  --input dataset/rsna_dev_100 \
  --manifest-file sample_ids.csv \
  --output dataset/rsna_dev_100_prepared
```

### Full dataset with existing metadata

Use this when `dataset/rsna_full/metadata/` already contains the converted CSVs and `splits.csv`. Do not repeat conversion:

```sh
python dataset/scripts/prepare_dataset.py \
  --input dataset/rsna_full/metadata \
  --output dataset/rsna_full/prepared \
  --visual-limit 25
```

### First preparation from official JSON files

After downloading and extracting the original dataset, the same entry point can also create the metadata. This requires a new or empty `metadata/` directory:

```sh
python dataset/scripts/prepare_dataset.py \
  --input dataset/rsna_full/metadata \
  --annotations dataset/rsna_full/source_metadata/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json \
  --mappings dataset/rsna_full/source_metadata/pneumonia-challenge-dataset-mappings_2018.json \
  --images-dir dataset/rsna_full/images \
  --output dataset/rsna_full/prepared \
  --visual-limit 25
```

Conversion writes `image_manifest.csv`, `labels.csv`, `detailed_class_info.csv`, `excluded_images.csv`, and `conversion_summary.json`. The workflow then creates `splits.csv` and processes the labelled images. `--images-dir` is only used for JSON conversion; existing metadata supplies image paths through its manifest.

Converted manifests contain absolute source image paths. After moving the raw data to another machine, update the manifest paths before reprocessing. Prepared PNG paths are relative to the prepared directory.

## Fixed patient-level splits

`splits.csv` has columns `image_id,nih_patient_id,split`, with `train`, `val`, or `test`. Read IDs as strings to preserve leading zeros. The original label column `patientId` identifies an image; use the manifest's `nih_patient_id` to identify a person.

The procedure uses seed 42: split patients 70/30, then divide the remaining patients equally into validation and test sets. Stratify by each patient's maximum target and keep all images of each patient together.

| Dataset | Split | Patients | Images |
| --- | --- | ---: | ---: |
| Development | train | 67 | 70 |
| Development | val | 14 | 14 |
| Development | test | 15 | 16 |
| Full | train | 8,563 | 20,818 |
| Full | val | 1,835 | 4,271 |
| Full | test | 1,835 | 4,595 |

Patient grouping means image proportions are not exactly 70/15/15. Preparation checks complete assignments, matching patient IDs, and no patient overlap. Existing valid assignments are reused rather than regenerated.

Verify a saved split against the fixed seed without changing files:

```sh
python dataset/scripts/prepare_dataset.py \
  --input dataset/rsna_full/metadata \
  --check-splits
```

For the development sample, use `--input dataset/rsna_dev_100 --manifest-file sample_ids.csv --check-splits`.

## Shared image and box formats

- Apply DICOM rescale slope and intercept, then invert MONOCHROME1 images.
- Clip per image at the 0.5th and 99.5th intensity percentiles and normalize to float32 values in `[0, 1]`.
- Resize with bilinear interpolation, preserve aspect ratio, and pad with black pixels to **512 by 512**.
- Save rounded 8-bit RGB PNGs with three identical grayscale channels, in `[0, 255]`.
- Apply the same horizontal/vertical scale and padding to boxes. No crop, rotation, or random augmentation is applied.

Use `--target-size` only when the team agrees on a different size. Model-specific normalization and augmentation belong in the training data loader.

Box coordinates are floating-point pixel edges, with origin at the top left and exclusive maximum coordinates. Original and processed coordinates are separate; detection training uses `processed_x_min`, `processed_y_min`, `processed_x_max`, and `processed_y_max`.

## Outputs and training inputs

| Output inside the prepared directory | Purpose |
| --- | --- |
| `processed_images/` | PNG inputs for classification and detection |
| `labels.csv` | One row per image: patient, target, class, split, paths, dimensions, and status |
| `boxes.csv` | One row per positive box; join to labels using `image_id` |
| `summary.json` | Counts, settings, and issue totals |
| `issue_summary.csv` / `issue_summary.md` | Issues requiring inspection |
| `visual_checks.html` / `visual_checks/` | Before/after previews with drawn boxes |
| `review_examples.jpg` | Automatic contact sheet with one preview per class/box-count group |

For classification, load PNGs using `labels.csv`'s `processed_path` and select the saved `split`. Detection also uses the processed coordinates from `boxes.csv`; keep negative images with empty box lists. Preview images contain drawn annotations and are only for review.

Resolve `processed_path` relative to the prepared directory. The source `image_path` points back to the raw DICOM and is not needed to load the training PNG. The input metadata `labels.csv` and prepared `labels.csv` have different schemas.

Previews are limited per class/box-count group, not across the whole dataset. Use `--visual-limit 0` to disable previews and the contact sheet, or `-1` for all images.

Final CSVs and reports are written after all images have been processed. Before training:

1. Confirm the command finished successfully and inspect `summary.json` and issue reports.
2. For the sample, expect 100 decoded images, 82 boxes, 50 positive images, and 96 patients. For the full dataset, expect 29,684 decoded images, 11,390 boxes, 7,106 positive images, and 12,233 patients.
3. Confirm split counts match the table above and resolve reported errors.
4. Open `visual_checks.html` or `review_examples.jpg` to inspect orientation, intensity, and box alignment. Automated coordinate checks do not replace visual review.

## Script responsibilities

| File | Responsibility |
| --- | --- |
| `prepare_dataset.py` | Command entry point, fixed splits, image/box workflow, and review outputs |
| `image_processing.py` | Shared DICOM loading, intensity processing, and resizing functions |
| `convert_rsna_annotations.py` | Official JSON-to-CSV conversion called by the entry point |
