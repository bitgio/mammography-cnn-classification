# Data Directory

This directory is intended to document the structure of the datasets used in this project based on the CBIS-DDSM collection from The Cancer Imaging Archive.

## Dataset Components

The project relies on three main types of data:

- **Imaging data**: mammography DICOM files, including full-field images and ROI/cropped images used for training, testing, and interpretability analysis (Grad-CAM).
- **Clinical annotation data**: CSV files containing diagnostic and lesion-related information such as pathology (benign/malignant), mass shape, assessment, and breast density.
- **Technical metadata**: CSV files used to link imaging data with clinical annotations, including image paths, ROI references, and acquisition-related information.

## Data Availability

The dataset is not included in this repository due to the size of the imaging data and in compliance with the TCIA data usage policy. Clinical annotation and technical metadata CSV files are also excluded to maintain a clean and focused project structure.

All data can be obtained from the original source [The Cancer Imaging Archive - CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/).

## Data Citation

Users must abide by the TCIA Data Usage Policy and provide proper attribution when using this dataset:

Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016).
Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) [Data set].
The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
