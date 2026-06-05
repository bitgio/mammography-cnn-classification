# Deep Learning-Based Prediction of Malignancy in Mammography

## Overview

This project explores the use of Convolutional Neural Networks (CNNs) for breast cancer detection from mammography images using publicly available medical datasets.

In addition to deep learning model development, the project includes a structured exploratory data analysis (EDA) workflow based on Pandas and SQL to validate data quality, analyze clinical metadata, and support reproducible machine learning experimentation.

The repository is organized as an end-to-end workflow including data preprocessing, exploratory analysis, CNN training, evaluation, and Grad-CAM-based model explainability.

Early detection of breast cancer is critical for improving patient outcomes. This project aims to investigate deep learning techniques for automated classification while also improving transparency through medical imaging visualization methods.

## Dataset

This project is based on publicly available medical imaging data from the CBIS-DDSM dataset hosted by The Cancer Imaging Archive.
Specifically, the dataset includes:

- Mass-Training Full Mammogram Images (DICOM) used for training and validation
- Mass-Test Full Mammogram Images (DICOM) used for final testing
- Mass-Test ROI, cropped images, and lesion masks (DICOM) used for interpretability (Grad-CAM analysis against ground-truth lesion annotations)
- Clinical annotation CSV files describing pathology and lesion characteristics (e.g., mass shape, assessment, breast density)
- Technical metadata CSV files used to link imaging data with clinical records (e.g., image paths, ROI references)

Note: The imaging dataset is not included in this repository due to its size and in compliance with the TCIA data usage policy. Clinical annotation and technical metadata CSV files are intentionally excluded to maintain a clean and focused project structure, avoiding redundancy since they are directly available from the original dataset.

The full dataset is publicly available and can be downloaded from [The Cancer Imaging Archive - CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/).

Data Citation (required by TCIA):
Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016).
Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) [Data set].
The Cancer Imaging Archive. [DOI: 10.7937/K9/TCIA.2016.7O02S9CY](https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY)

## Objective

- Binary classification of mammograms (benign vs malignant)
- Evaluate CNN performance using accuracy and ROC-AUC
- Provide model interpretability using Grad-CAM
- Perform exploratory data analysis and metadata validation using Pandas and SQL
- Assess dataset consistency, patient-level leakage, and class distribution prior to model training

## Methodology

### Exploratory Data Analysis

Before model development, a structured exploratory data analysis workflow was conducted on mammography clinical data and imaging metadata.

The analysis included:

- integration of clinical annotation data and technical metadata
- composite key construction for reliable merging
- missing and duplicated values assessment
- pathology class distribution analysis
- breast density and mass shape exploration
- patient-level redundancy and leakage analysis
- SQL-based analytical validation using SQLite
- training/test schema consistency checks

The EDA pipeline was implemented using:

- Pandas
- SQLite
- Matplotlib
- Seaborn

This phase improved dataset transparency, reproducibility, and overall data quality prior to CNN model training.

### Model Development

Multiple Convolutional Neural Network (CNN) architectures were designed and implemented using:

- TensorFlow
- Keras

Different architectures were explored to identify the best-performing configuration.

### Training Strategy

To improve generalization and robustness, several techniques were applied:

- Hyperparameter tuning
  (learning rate, batch size, input size, layer number, filter number)
- Regularization
  Dropout layers were introduced to mitigate overfitting
- Data augmentation
  Applied to increase dataset variability and improve model robustness
 
 ### Evaluation Protocol

- Separate training, validation, and test sets were used.
- Model performance was evaluated using:
  - Accuracy
  - ROC-AUC

To further assess model reliability:

- Cross-validation was performed, but the training and validation performances varied considerably across the folds. This resulted in an unsteady degree of overfitting, which was influenced by the composition of the datasets used.

### Model Selection

Different CNN architectures were systematically compared based on validation performance, leading to the selection of the best-performing model.  

## Results

The models show the ability to distinguish between benign and malignant cases, with performance evaluated through:

- Training and validation accuracy
- ROC curve analysis

### Performance Metrics

| Metric | Value |
|--------|-------|
| AUC (Validation Set) |	0.798 |
| AUC (Test Set) |	0.653 |

The observed performance gap between the validation and test sets indicates a **generalization issue**, suggesting that the model might be **overfitting** the training distribution. Specifically, the AUC being higher on the validation set compared to the test set is a typical sign of overfitting, where the model performs well on the data it was trained on but struggles to generalize to unseen data. These values are visualized in the following figures.

---

### Figure 1. ROC Curves on Validation and Test Sets

![Figure 1 - ROC Curves](results/roc_validation_vs_test.png)

**Fig. 1:** Receiver Operating Characteristic (ROC) curves for validation and test sets. The decrease in performance on the test set highlights a degradation in model generalization, consistent with overfitting observed during training.

### Figure 2. Learning Curves on Training and Validation Sets

![Figure 2 - Learning Curves](results/learning_training_vs_validation.png)

**Fig. 2:** Learning curves for training and validation sets. The discrepancy in accuracy and loss between the training and validation sets is an evident sign of overfitting. Specifically, there is a considerable gap between the training and validation accuracy as both sets reach their respective plateaus.

### Figure 3. ROC Curves - Cross-Validation

![Figure 3 - ROC Curves Cross-Validation](results/roc_cross_validation.png)

**Fig. 3:** The ROC curves from the cross-validation folds illustrate variability in model performance, with fluctuations depending on the data split. These fluctuations suggest potential issues with generalization, indicating that the model may be overfitting to certain subsets of the data.

## Key Insights

- A significant gap between validation and test performance indicates overfitting behavior
- Model performance does not fully generalize to unseen data, highlighting the importance of robust regularization and data augmentation strategies
- Benchmarking multiple CNN architectures was essential to identify performance limitations
- Evaluation results emphasize the importance of reliable validation strategies in medical imaging tasks

## Interpretability (Grad-CAM)

To improve model transparency, Grad-CAM was used to visualize the regions of mammograms influencing the model's predictions across different cases. Representative examples were selected from different prediction outcomes (True Positive, False Positive, True Negative, False Negative) to better understand model behavior.

### Figure 4. Correct prediction with aligned attention

![Figure 4 - Grad-CAM Good Case](results/gradcam_roi_aligned.png)

**Fig. 4:** In this example, the model focuses on the region corresponding to the ground-truth lesion mask. This suggests that it is learning clinically meaningful features, with minor spatial discrepancies.

### Figure 5. Misaligned attention (failure case)

![Figure 5 - Grad-CAM Bad Case](results/gradcam_attention_misaligned.png)

**Fig. 5:** In this case, the model prediction is incorrect and the attention is not aligned with the lesion region. This highlights limitations in generalization and suggests potential reliance on non-clinical features.

### Findings

A qualitative Grad-CAM analysis was conducted on the selected test cases.

- In some instances, the model attends to clinically relevant regions corresponding to annotated ROIs
- In other cases, attention is partially misaligned with lesion regions, indicating limitations in spatial generalization and localization consistency

These results highlight both the strengths and limitations of the model in terms of interpretability and clinical reliability.

## Limitations

- Limited dataset size, which may affect model generalization to unseen clinical data
- Use of full mammograms without advanced image preprocessing techniques (e.g., segmentation or lesion-focused cropping) for model training
- No external validation dataset for assessing real-world performance
- Model generalization to clinical environments remains uncertain without further validation

## Future Work

- Apply transfer learning techniques (e.g., ResNet, EfficientNet) to improve performance and generalization
- Explore more advanced preprocessing and data augmentation strategies
- Investigate segmentation-based approaches for lesion localization
- Improve model robustness and reduce overfitting through better regularization techniques
- Validate the model on external datasets to assess generalization in real-world scenarios
- Incorporate clinically relevant metrics such as sensitivity and recall
- Enhance interpretability by integrating more quantitative analysis methods alongside Grad-CAM

## How to Run

This project was developed using Python in Google Colab and Kaggle environments.

The workflow is organized as a sequence of notebooks covering:
data preprocessing, exploratory analysis, model training, evaluation, and interpretability.

To reproduce the workflow, install dependencies and run the notebooks in sequential order.

Alternatively, the project can be run directly in Google Colab or Kaggle.

## Repository Structure

- notebooks/ → exploratory analysis and model development (in progress)
- results/ → performance metrics and visual outputs (ROC curves, training curves, Grad-CAM)
- data/ → tabular datasets used during analysis and model development (to be defined)
- future work includes modularizing reusable code and environment setup

## Technologies Used

- Python
- TensorFlow / Keras – CNN model development and training
- Pandas / NumPy – data manipulation and preprocessing
- SQLite – SQL-based exploratory data analysis and validation
- Matplotlib / Seaborn – data visualization and training metrics analysis
- Google Colab / Kaggle – development and experimentation environments

## About the Author

Master’s Degree in Physics – University of Pisa  
Focus on Machine Learning, Data Analysis, and Scientific Programming

## Key Takeaways

- Built and trained CNN models for medical imaging classification using a real-world dataset
- Developed a structured exploratory data analysis pipeline using Pandas and SQL, including dataset validation and leakage analysis
- Applied deep learning techniques, including regularization and data augmentation, to improve model generalization
- Evaluated model performance using clinically relevant metrics such as ROC-AUC
- Investigated model interpretability using Grad-CAM, analyzing both correct predictions and failure cases
- Gained hands-on experience with medical imaging data, including preprocessing and domain-specific challenges
