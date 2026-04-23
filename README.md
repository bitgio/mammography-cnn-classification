# Deep Learning-Based Prediction of Malignancy in Mammography

## Overview

This project focuses on the classification of mammography images (benign vs malignant) using deep learning techniques.
The goal is to develop and evaluate convolutional neural networks (CNNs) trained on medical imaging data, with particular attention to model generalization and explainability.

The work is based on a Master’s thesis in Physics and follows a research-oriented approach, including model benchmarking, performance evaluation, and interpretation of results.

## Dataset

- Mammography images in DICOM format
- Binary classification task: benign vs malignant lesions
- Data preprocessing steps:
  - image normalization
  - resizing
  - dataset splitting (training / validation / test)
  - data augmentation techniques to improve robustness

Note: Dataset not included due to privacy and usage restrictions.

## Methodology

- Designed multiple Convolutional Neural Network (CNN) architectures using TensorFlow and Keras
- Optimized:
  - hyperparameter tuning (learning rate, batch size, etc.)
  - regularization techniques (dropout layers)
  - data augmentation
- Investigated different models to identify the best-performing architectures
- Evaluated performance on separate validation and test sets to assess generalization
- Assessed model stability using cross-validation, observing consistent performance across folds  

## Results

Model performance was evaluated using the Area Under the ROC Curve (AUC) on both validation and test sets.

---

| Metric | Value |
|--------|-------|
| AUC (Validation Set) |	0.798 |
| AUC (Test Set) |	0.653 |

The observed performance gap indicates a generalization issue, suggesting overfitting on the training distribution.

---

### Figure 3. ROC curve on validation and test sets

![Figure 3 - ROC Curve](results/fig3_roc_validation_vs_test.png)

**Fig. 3:** Receiver Operating Characteristic (ROC) curves for validation and test sets. The decrease in performance on the test set highlights a degradation in model generalization, consistent with overfitting observed during training.

## Key Insights

- A significant gap between validation and test performance indicates overfitting behavior
- Model performance does not fully generalize to unseen data
- The analysis emphasizes the importance of robust evaluation strategies in medical imaging tasks
- Benchmarking multiple models was essential to identify performance limitations

## Model Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) was used to highlight regions of the input images most influential in the model's predictions.

### Figure 1. Correct prediction with aligned attention

![Figure 1 - Grad-CAM Good Case](results/fig1_gradcam_roi_aligned.png)

**Fig. 1:** The model correctly focuses on clinically relevant regions that largely overlap with the annotated ROI, indicating meaningful feature learning with minor spatial discrepancies.

### Figure 2. Misaligned attention (failure case)

![Figure 2 - Grad-CAM Bad Case](results/fig2_gradcam_attention_misaligned.png)

**Fig. 2:** The model highlights regions considerably outside the annotated ROI, suggesting limitations in spatial generalization and potential overfitting to non-clinical features.

### Findings:

- A qualitative Grad-CAM analysis on a small subset of test images shows that, in some cases, the model focuses on clinically relevant regions
- However, in other cases, attention is partially misaligned with respect to the annotated ROIs, suggesting potential limitations in generalization.

These results provide insight into both model reliability and limitations, which are critical in medical AI applications.

## Repository Structure

- notebooks/ → Jupyter notebooks for model development and experimentation
- src/ → (optional) modularized code for preprocessing and model definition
- results/ → evaluation metrics and generated plots (ROC curves, training curves, Grad-CAM examples)
- requirements.txt → project dependencies

## How to Run

pip install -r requirements.txt
jupyter notebook notebooks/model_development.ipynb

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib 

## Future Improvements

- Reduce overfitting through improved regularization strategies
- Explore alternative architectures and transfer learning
- Increase dataset size and diversity
- Explore advanced data augmentation and preprocessing techniques
- Improve model calibration and robustness

## Author

Master’s Degree in Physics – University of Pisa

Focus on Machine Learning, Data Analysis, and Scientific Programming

## Conclusion

This project highlights the importance of model generalization and interpretability in medical deep learning applications. The observed performance gap suggests that improving robustness remains a key challenge for real-world deployment.
