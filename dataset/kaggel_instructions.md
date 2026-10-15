# Full dataset preparation on Kaggle

Create a new **Kaggle Notebook** with **Internet On** and **Accelerator None**. Paste each code block below into a separate code cell, keeping `%%bash` as the first line.

These instructions use the same official RSNA adjudicated dataset and Calculated annotations as our server run. Use the downloads below to reproduce our results; the current workflow has not verified equivalence with Kaggle's competition `stage_2_*` files.

Start with a fresh session. The pipeline requires new or empty output folders and does not resume an interrupted preparation run.

## 1. Get the code and install dependencies

The separate environment uses Python 3.12.15 and the project requirements. Use its absolute Python path in every cell; shell activation does not carry over between cells.

```bash
%%bash
set -e

git clone --depth 1 \
  https://github.com/jianyu113/RBC_Letssolveit_LungStoryShort.git \
  /tmp/lungstoryshort-code

python -m pip install -q uv
uv venv --python 3.12.15 --seed /tmp/lungstoryshort

/tmp/lungstoryshort/bin/python -m pip install \
  torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cpu

/tmp/lungstoryshort/bin/python -m pip install \
  -r /tmp/lungstoryshort-code/requirements.txt

/tmp/lungstoryshort/bin/python -m pip check
```

## 2. Download and extract the data

```bash
%%bash
set -e

mkdir -p /tmp/rsna_full/source_metadata
cd /tmp/rsna_full

RSNA_SOURCE="https://s3.amazonaws.com/east1.public.rsna.org/AI/2018"

curl -fL -C - --retry 5 \
  "$RSNA_SOURCE/pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip" \
  -o rsna_images.zip

curl -fL -C - --retry 5 \
  "$RSNA_SOURCE/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json" \
  -o source_metadata/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json

curl -fL -C - --retry 5 \
  "$RSNA_SOURCE/pneumonia-challenge-dataset-mappings_2018.json" \
  -o source_metadata/pneumonia-challenge-dataset-mappings_2018.json

unzip -tq rsna_images.zip
unzip -q -n rsna_images.zip -d images

find images -type f -iname '*.dcm' | wc -l
```

Expected count: **30,000 DICOM files**.

## 3. Run the complete pipeline

This converts annotations, creates the fixed patient-level splits, processes images and boxes, and generates review previews. Progress appears directly below the cell.

```bash
%%bash
set -e

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/tmp/lungstoryshort/bin/python -u \
  /tmp/lungstoryshort-code/dataset/scripts/prepare_dataset.py \
  --input /kaggle/working/rsna_full/metadata \
  --annotations /tmp/rsna_full/source_metadata/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json \
  --mappings /tmp/rsna_full/source_metadata/pneumonia-challenge-dataset-mappings_2018.json \
  --images-dir /tmp/rsna_full/images \
  --output /kaggle/working/rsna_full/prepared \
  --target-size 512 \
  --visual-limit 25
```

## 4. Check the results

```bash
%%bash
set -e

/tmp/lungstoryshort/bin/python \
  /tmp/lungstoryshort-code/dataset/scripts/prepare_dataset.py \
  --input /kaggle/working/rsna_full/metadata \
  --check-splits

/tmp/lungstoryshort/bin/python -m json.tool \
  /kaggle/working/rsna_full/prepared/summary.json
```

Expected results:

| Item | Expected |
| --- | ---: |
| Decoded images | 29,684 |
| Positive / negative images | 7,106 / 22,578 |
| Bounding boxes | 11,390 |
| Patients | 12,233 |
| Train / validation / test images | 20,818 / 4,271 / 4,595 |
| Train / validation / test patients | 8,563 / 1,835 / 1,835 |
| Issues | `{}` |

The 316 images without Calculated annotations are excluded. Review `review_examples.jpg` and the images in `visual_checks/` for orientation, intensity, and box alignment before training. Automated checks do not replace visual review.

## 5. Save and share the results

After adding all four cells, choose **Save Version → Save & Run All**. Kaggle runs them in a fresh session and saves outputs under `/kaggle/working` when the version completes. If you already ran the cells interactively, this option runs them again.

Keep these outputs:

```text
/kaggle/working/rsna_full/
    metadata/
        splits.csv             # Fixed patient-level assignments
        image_manifest.csv     # Source image and patient mapping
        conversion_summary.json
        ...
    prepared/
        processed_images/      # 512 x 512 PNGs
        labels.csv             # Image labels, paths, and saved split
        boxes.csv              # Original and processed box coordinates
        summary.json
        review_examples.jpg
        ...
```

Training uses the prepared PNGs and CSVs. Keep `splits.csv` unchanged across the team. The raw downloads and environment under `/tmp` are temporary; the saved manifest's raw image paths expire with that session. Prepared PNG paths are relative to the prepared directory and remain usable when the whole directory is shared.

See [Data preparation](DATA_PREPARATION.md) for the image and bounding-box formats. Platform references: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) and [uv Python environments](https://docs.astral.sh/uv/pip/environments/).
