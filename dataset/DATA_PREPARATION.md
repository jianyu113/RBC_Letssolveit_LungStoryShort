# Data preparation

For environment setup and activation, see the [project README](readme.md). Run all dataset commands below from the repository root.

## Shared dataset

The shared development sample is stored at `dataset/rsna_dev_100/`; generated files go into the separate `dataset/rsna_dev_100_prepared/` directory.

Each teammate should obtain the same sample folder from the team's shared Drive and place it at the same repository-relative location. Keep the images, CSV files, and fixed split file together:

```text
dataset/rsna_dev_100/
    images/                   # 100 original DICOM files
    sample_ids.csv            # Image IDs, patient IDs, paths, and checksums
    labels.csv                # Original image targets and bounding boxes
    detailed_class_info.csv   # Original Calculated class labels
    splits.csv                # Fixed patient-level train/val/test assignments
    provenance.json           # Source URLs and sampling procedure
    README.md                 # Sample description and attribution
    source_metadata/          # Original RSNA annotation and mapping JSON files
```

This sample comes from the RSNA official adjudicated pneumonia dataset, using its **Calculated** label group. It contains 100 images from 96 NIH patients: 50 Lung Opacity images, 19 Normal images, and 31 No Lung Opacity / Not Normal images. The positive images have 82 bounding boxes. `Target=0` does not necessarily mean a normal image.

See the [sample README](dataset/rsna_dev_100/README.md) for the NIH and RSNA source references, terms, and attribution. Retain those attributions when sharing the sample. The CSV files were converted from official JSON annotations; membership in the original Kaggle Stage 2 splits has not been verified. This small sample is for developing the preparation workflow, not final model performance claims.

## Fixed patient-level splits

Use the existing `splits.csv` unchanged across the team. Its columns are `image_id,nih_patient_id,split`; `split` is `train`, `val`, or `test`. Read both ID columns as strings to preserve leading zeros.

The source `labels.csv` column named `patientId` identifies an **image**. Use `sample_ids.csv`'s `nih_patient_id` to identify a person and prevent patient overlap between sets.

| Split | Patients | Images | Positive images | Negative images |
| --- | ---: | ---: | ---: | ---: |
| train | 67 | 70 | 35 | 35 |
| val | 14 | 14 | 7 | 7 |
| test | 15 | 16 | 8 | 8 |

The split uses seed 42, a 70/30 patient split, and an equal division of the remaining patients into validation and test sets. Stratification uses each patient's maximum target; all images from one patient stay together. Patient grouping and rounding explain why image counts are not exactly 70/15/15.

Verify the saved assignments without changing them:

```sh
python dataset/scripts/splits.py --check
```

## Complete preparation workflow

Run the single preparation entry point:

```sh
python dataset/scripts/prepare_dataset.py \
  --input dataset/rsna_dev_100 \
  --output dataset/rsna_dev_100_prepared
```

The output directory must be new or empty. To run again while keeping an earlier result, choose a different output directory. The original DICOM images, annotations, and fixed split file stay unchanged.

`prepare_dataset.py` reads the image manifest, labels, classes, and fixed splits. It checks unique image assignments, patient identity agreement, and patient overlap before processing. It uses the functions in `image_processing.py` for image processing and joins each image to its fixed split by `image_id`.

The shared image format is:

- Apply DICOM rescale slope and intercept, then invert MONOCHROME1 images.
- Clip each image at its 0.5th and 99.5th intensity percentiles and normalize to float32 values in `[0, 1]`.
- Resize with bilinear interpolation, preserving aspect ratio, and pad with black pixels to **512 by 512**. Use `--target-size` to explicitly change this size for an agreed experiment.
- Round the processed values to 8-bit PNG pixels. Files use RGB with three identical grayscale channels; the stored range is `[0, 255]`.
- Apply no random augmentation, cropping, or rotation. Model-specific normalization belongs in the later training data loader.

Bounding boxes use floating-point pixel-edge coordinates: origin at the top left, `x` increasing rightward, `y` increasing downward, and exclusive maximum coordinates. The output keeps the original `x`, `y`, `width`, and `height` values and separate original and processed `x_min`, `y_min`, `x_max`, and `y_max` coordinates. Processed boxes use the same actual horizontal/vertical resize factors and padding as their images. Do not normalize these coordinates again unless the model's input format requires it.

## Outputs and completion checks

| Output | Purpose |
| --- | --- |
| `labels.csv` | One row per image, including patient ID, target, class, fixed split, paths, dimensions, and processing status |
| `boxes.csv` | One row per positive bounding box, with original and processed coordinates; join to labels using `image_id` |
| `processed_images/` | Processed PNG images referenced by `labels.csv` |
| `summary.json` | Counts, preparation settings, and issue summary |
| `issue_summary.csv` / `issue_summary.md` | Issues requiring inspection |
| `visual_checks.html` / `visual_checks/` | Original and processed images with boxes for manual review |

Paths in the output `labels.csv` are relative to the output directory. Resolve them from that directory, including the source `image_path` and the generated `processed_path`. Moving the whole `dataset/` folder preserves the standard layout.

Before calling preparation complete:

1. Confirm 100 images decoded, 100 PNG images were produced, and all 82 boxes remain present.
2. Confirm 96 distinct patients, 50 positive and 50 negative images, and the split counts above.
3. Inspect the issue reports and resolve any reported errors.
4. Open `visual_checks.html` and review image orientation, intensity, and box alignment. A numerical coordinate check alone does not establish that a box is correctly placed.

Optionally create a representative contact sheet after preparation:

```sh
python dataset/scripts/make_review_examples.py \
  --results dataset/rsna_dev_100_prepared \
  --output dataset/rsna_dev_100_prepared/review_examples.jpg
```

## Using converted annotations later

The current 100-image sample already contains the required CSV files; do not rerun JSON conversion for this workflow. `convert_rsna_annotations.py` is available when preparing a different subset from the official source JSON files.

That converter names its manifest `image_manifest.csv`. Pass `--manifest-file image_manifest.csv` to the preparation entry point when using that output. `--splits-file` selects a different fixed split CSV when needed; its assignments must cover the input images and preserve patient separation. The current `splits.py` expects `sample_ids.csv`, so a new dataset also requires an explicitly prepared, consistent manifest and split file before running this workflow.
