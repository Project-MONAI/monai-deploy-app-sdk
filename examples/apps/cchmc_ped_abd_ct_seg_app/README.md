# MONAI Application Package (MAP) for CCHMC Pediatric Abdominal CT Segmentation MONAI Bundle

This MAP is based on the [CCHMC Pediatric Abdominal CT Segmentation MONAI Bundle](https://github.com/cchmc-dll/pediatric_abdominal_segmentation_bundle/tree/original). This model was developed at Cincinnati Children's Hospital Medical Center by the Department of Radiology.

The PyTorch and TorchScript DynUNet models can be downloaded from the [MONAI Bundle Repository](https://github.com/cchmc-dll/pediatric_abdominal_segmentation_bundle/tree/original/models).

For questions, please feel free to contact Elan Somasundaram (Elanchezhian.Somasundaram@cchmc.org) and Bryan Luna (Bryan.Luna@cchmc.org).

## Pipeline Overview

The application executes the following processing DAG:

1. **DICOM Data Loader** — loads all DICOM instances from the input folder
2. **DICOM Series Selector** — selects qualifying CT series using JSON-based rules (see [Series Selection](#series-selection))
3. **DICOM Series to Volume** — converts the selected series to a 3D image volume; registers [nvImageCodec](https://github.com/NVIDIA/nvImageCodec) GPU-accelerated decoder plugin (`decoder_nvimgcodec.py`) with `pydicom` at startup for compressed pixel data (JPEG, JPEG 2000, HTJ2K). **Note:** CUDA 12-compatible `nvJPEG` lossless decoder silently produces zero-filled buffers for JPEG Lossless streams where `9 <= BitsStored <= 15` or `DHT` is before `SOF3`. A custom version of `decoder_nvimgcodec.py` has been created that detects this condition and raises `NotImplementedError` so pydicom falls through to the next capable decoder (e.g. `GDCM`). This issue is resolved in  `nvJPEG>= 13.0.2` (see [nvImageCodec issue](https://github.com/NVIDIA/nvImageCodec/pull/51#issuecomment-4407066179))
4. **Abdomen Seg Operator** — runs DynUNet inference to produce Liver / Spleen / Pancreas segmentation masks (labels 1 / 2 / 3)
5. **Segmentation Metrics Operator** — computes volume, slice count, pixel count, and intensity statistics per organ
6. **Segmentation Z-Score Operator** — compares organ metrics to age/sex-specific normative CSV data; generates z-scores, percentiles, and an optional PDF report
7. **Segmentation Contour Operator** — extracts boundary contours from the segmentation mask
8. **Segmentation Overlay Operator** — blends contours onto the input scan to produce an RGB overlay image; applies VOI LUT windowing from the source series when available; GPU-accelerated
9. **DICOM SEG Writer** — writes organ masks as a DICOM Segmentation object → `output/SEG/`
10. **DICOM SR Writer** — writes organ volumes (Liver, Spleen) and z-scores as a DICOM Enhanced SR → `output/SR/`
11. **DICOM SC Writer** — writes the contour overlay as a DICOM Secondary Capture → `output/SC/`

## Custom Operators

| Operator | File | Description |
|---|---|---|
| `AbdomenSegOperator` | `abdomen_seg_operator.py` | Inference with DynUNet; supports PyTorch (`.pt` state dict) and TorchScript model loading |
| `DICOMTextSRWriterOperator` | `dicom_text_sr_writer_operator.py` | Writes DICOM Enhanced SR with a structured content sequence; supports field filtering and custom Concept Name codes |
| `DICOMSCWriterOperator` | `dicom_sc_writer_operator.py` | Writes multi-frame DICOM Secondary Capture with source series metadata copied |
| `SegmentationMetricsOperator` | `segmentation_metrics_operator.py` | Computes volume, slice count, pixel count, and intensity stats; GPU-accelerated via CuPy when available |
| `SegmentationZScoreOperator` | `segmentation_zscore_operator.py` | Computes z-scores and percentiles against normative CSV data; generates `matplotlib` PDF visualization |
| `SegmentationContourOperator` | `segmentation_contour_operator.py` | Extracts label boundaries using MONAI `LabelToContour` transform |
| `SegmentationOverlayOperator` | `segmentation_overlay_operator.py` | Generates RGB overlay via alpha blending; handles CT VOI LUT windowing and per-slice windowing variations |

## DICOM Outputs

| Output | Subfolder | Contents |
|---|---|---|
| DICOM SEG | `output/SEG/` | Organ segmentation masks (Liver, Spleen, Pancreas) |
| DICOM SR | `output/SR/` | Liver and Spleen volumes and z-scores (CT Abdomen Report, LOINC 41806-1) |
| DICOM SC | `output/SC/` | Organ contour overlay images |
| DICOM Encapsulated PDF | `output/PDF/` | Z-score quantile curve report |

Output visibility is controlled by the `labels_dict` parameters in `app.py`. By default, the model segments Liver (1), Spleen (2), and Pancreas (3); SEG and SC outputs have all organs, but only Liver and Spleen are included in the SR and Encapsulated PDF outputs.

## Z-Score Analysis

The `SegmentationZScoreOperator` compares computed organ metrics against sex-stratified, age-specific normative reference data to produce clinically interpretable z-scores and percentiles.

### How It Works

1. Patient demographics (age and sex) are extracted directly from the DICOM series tags (`PatientAge`, `PatientSex`).
2. For each organ metric (e.g., liver volume, spleen volume, liver HU), the operator locates the matching normative dataset in the `assets/` folder.
3. Quantile regression curves (5th–95th percentile, in 5-percentile steps) are interpolated at the patient's exact age using linear interpolation with extrapolation at the boundaries.
4. The patient's measured value is placed within those quantile curves to estimate its percentile, which is then converted to a z-score via the inverse normal CDF (`scipy.stats.norm.ppf`).
5. If `generate_plots=True`, a multi-panel PDF is rendered with quantile curves and the patient's value annotated for each organ, then passed downstream via the `pdf_bytes` output port.

### PDF Report

The `SegmentationZScoreOperator` outputs an optional **PDF visualization** (in-memory `bytes`) containing one subplot per organ. Each panel shows:
- Age-specific quantile curves (5th, 25th, 50th, 75th, 95th percentile) for the patient's sex
- The patient's measured value plotted as a marker at the patient's age
- An annotation box displaying the raw value, percentile, and z-score

The PDF is passed to `DICOMEncapsulatedPDFWriterOperator`.

## Assets Folder

The `assets/` folder contains the sex-stratified normative reference data used by `SegmentationZScoreOperator`. Each subfolder corresponds to one biomarker and must contain two CSV files — one for males (`results_m_fine.csv`) and one for females (`results_f_fine.csv`).

### Structure

```text
assets/
├── liver/               # Liver volume normative data
│   ├── results_m_fine.csv
│   ├── results_f_fine.csv
│   ├── results_df.csv   # Raw cohort data
│   ├── outlier_df.csv   # Identified outliers
│   ├── stats.json       # Cohort summary statistics
│   └── figure.html      # Interactive quantile figure
├── liver_hu/            # Liver mean HU normative data
│   └── (same structure)
└── spleen/              # Spleen volume normative data
    └── (same structure)
```

### CSV Format

Each `results_{m,f}_fine.csv` contains pre-computed quantile regression curves with the following columns:

| Column | Description |
|---|---|
| `Age` | Age in years (2.0–19.0, 0.5-year steps) |
| `0.05` – `0.95` | Predicted biomarker value at that quantile level (5th–95th percentile, 5-point steps) |

### Cohort Summary

| Biomarker | Males (n) | Females (n) | Age Range |
|---|---|---|---|
| Liver volume (mL) | 1,025 | 1,107 | 2–19 years |
| Liver mean HU | 1,025 | 1,107 | 2–19 years |
| Spleen volume (mL) | 1,013 | 1,089 | 2–19 years |

### Adding a New Biomarker

To add normative data for a new organ or metric:
1. Create a subfolder under `assets/` (e.g., `assets/pancreas/`).
2. Add `results_m_fine.csv` and `results_f_fine.csv` with the same column format described above.
3. In `app.py`, add the organ to `labels_dict` and update `organ_name_mapping` in `SegmentationZScoreOperator` if the metric key differs from the folder name.

## Series Selection

Series selection criteria are defined in JSON within `app.py` and evaluated by `DICOMSeriesSelectorOperator`. The default rules select **Standard Axial CT** series meeting all of the following:

- **Modality:** `CT` (case-insensitive)
- **ImageOrientationPatient:** Axial orientation (determined programmatically)
- **ImageType:** contains `PRIMARY` (excludes secondary and reformatted series)
- **SliceThickness:** between 2.0 and 5.0 mm (inclusive)
- **SeriesDescription:** does not contain `cor`, `sag`, or `lung` (case-insensitive)

All series matching the criteria are selected (`all_matched=True`) and sorted by SOP instance count. Downstream operators perform inference and write outputs for the first selected series only.

## Model Information

- **Architecture:** DynUNet (3D, instance normalization, residual blocks, deep supervision disabled)
- **Labels:** background (0), liver (1), spleen (2), pancreas (3)
- **Algorithm Name:** CCHMC Pediatric CT Liver-Spleen Segmentation
- **Algorithm Version:** 0.4.3
- **MAP Version:** 1.10.0

## Resource Requirements

| Resource | Requirement |
|---|---|
| CPU | 1 |
| GPU | 1 |
| System Memory | 1 Gi |
| GPU Memory | 11 Gi |

With a NVIDIA GeForce RTX 3090 (24 GB), inference for a 204-instance input series takes approximately 21 seconds.

## Scripts

The `scripts/` folder contains shell scripts for common tasks (e.g. running the app code locally with Python interpreter, MAP packaging, MAP execution). All scripts expect a `.env` file in the working directory that sets `HOLOSCAN_INPUT_PATH`, `HOLOSCAN_OUTPUT_PATH`, and `HOLOSCAN_MODEL_PATH`. See example below:

```env
HOLOSCAN_INPUT_PATH=${PWD}/input
HOLOSCAN_MODEL_PATH=${PWD}/model/dynunet_FT.ts
HOLOSCAN_OUTPUT_PATH=${PWD}/output
```

| Script | Arguments | Description |
|---|---|---|
| `model_run.sh` | — | Runs the app locally with the Python interpreter |
| `map_build.sh` | `<tag_prefix> <image_version> <sdk_version> <cuda_version>` | Packages the app as a MAP using `monai-deploy package` |
| `map_run.sh` | `<tag_prefix> <image_version>` | Runs the MAP locally using `monai-deploy run` |
| `map_run_interactive.sh` | — | Runs the MAP container interactively for debugging |
| `map_extract.sh` | — | Extracts the MAP container filesystem for inspection | 
