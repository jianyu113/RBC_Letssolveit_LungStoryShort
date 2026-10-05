# RSNA Development Sample: 100 Images

A fixed sample for developing and checking the LungStoryShort preprocessing pipeline.

## Contents

| File or folder | Contents |
| --- | --- |
| `images/` | 100 original DICOM images, unchanged from the RSNA archive |
| `labels.csv` | Image targets and all corresponding bounding boxes |
| `detailed_class_info.csv` | One original Calculated class per image |
| `sample_ids.csv` | Fixed image list, NIH identifiers, file paths, sizes, and SHA-256 checksums |
| `provenance.json` | Source URLs, metadata checksums, and exact sampling procedure |
| `source_metadata/` | Complete original RSNA annotation and mapping JSON files |

The source JSON files describe the larger dataset. Use the three CSV files for this 100-image sample.

## Sample and labels

- 50 images: Lung Opacity (`Target=1`), with 82 bounding boxes.
- 19 images: Normal (`Target=0`).
- 31 images: No Lung Opacity / Not Normal (`Target=0`).
- Sampling seed: 42. The source population contains 29,684 images with Calculated labels; 316 images without those labels were excluded.

`labels.csv` uses columns `patientId,x,y,width,height,Target`. Here, `patientId` is the image identifier, not a person identifier. An image with multiple boxes has multiple rows. Negative images have one row with empty box coordinates. Coordinates are unchanged floating-point pixel values on the original 1024-by-1024 image, with the origin at the top left.

The annotations come exclusively from the official **Calculated** label group. Target indicates the Lung Opacity annotation; negative images can still have other abnormalities.

This is the **RSNA official adjudicated dataset**. The CSV files were converted from its JSON annotations and are not the original Kaggle Stage 2 CSV files. Membership in the Kaggle Stage 2 training/test splits has not been verified. Use this sample for preprocessing development, not final model evaluation. There are 96 distinct NIH patient identifiers; future dataset splits should keep each patient's images together.

## Team use

Upload the accompanying ZIP to the shared Drive. Each teammate should extract the same ZIP and use `sample_ids.csv` as the common sample list. Keep original images unchanged and save processed outputs separately.

All 100 files passed ZIP CRC and SHA-256 checks and decoded to 1024-by-1024 arrays with pydicom 3.0.2 and Pillow 12.3.0 in the `lungstoryshort` Conda environment. The DICOM transfer syntax is JPEG Baseline (Process 1).

## Sources and attribution

The NIH Clinical Center is the original image data provider.

- [NIH Chest X-ray dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. IEEE CVPR, pp. 3462-3471, 2017.
- [RSNA Pneumonia Detection Challenge dataset](https://www.rsna.org/artificial-intelligence/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018)
- Shih G, et al. [Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia](https://doi.org/10.1148/ryai.2019180041). Radiology: Artificial Intelligence, 2019.
- [RSNA Terms of Use and Attribution](https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.pdf)

Retain these attributions when sharing the data. Do not attempt to identify or contact individuals represented in the dataset.
