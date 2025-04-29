Stacking Ensemble Classifier for Multi-Modal Breast Cancer Diagnosis
This repository contains code, data, and supplementary materials for the manuscript "A Stacking Ensemble Classifier for Breast Cancer Detection Using Multi-Modal Data" submitted to the Journal of Biomedical Informatics (2025). The study integrates Fine Needle Aspiration (FNA) data from the Wisconsin Breast Cancer Dataset (WBCD) and mammography data from CBIS-DDSM to develop a stacking ensemble classifier with high recall (0.944) and interpretability.

Model Training and Evaluation Pipeline
The flowchart below illustrates the final model training and evaluation pipeline, using a classification threshold of 0.3, as described in the manuscript’s supplementary materials (Flowchart of Final Model Training and Evaluation Pipeline).

Pipeline Overview[Flowchart of Final Model Training and Evaluation Pipeline.pdf](https://github.com/user-attachments/files/19967472/Flowchart.of.Final.Model.Training.and.Evaluation.Pipeline.pdf)

Input: Merged WBCD and CBIS-DDSM dataset (3549 samples, six shared features: radius_mean, texture_mean, perimeter_mean, smoothness_mean, compactness_mean, symmetry_mean).
Process:
Preprocess and align features from WBCD and CBIS-DDSM datasets (e.g., Gaussian blur, VGG16, PCA, GLCM, Canny edge detection).
Train a stacking ensemble classifier (RandomForest, XGBoost, GradientBoosting, LightGBM, CatBoost, with Logistic Regression meta-model) using 5-fold cross-validation.
Evaluate on the test set (1028 samples) with a 0.3 threshold, achieving 87.6% accuracy and 0.944 recall.
Perform LIME analysis for interpretability on 10 instances (5 benign, 5 malignant).

Outputs:
aligned_breast_cancer_dataset.csv: Processed dataset with aligned features.
stacking_classifier_merge_0.3.pkl: Trained stacking classifier model.
lime_analysis_report.html: LIME analysis report detailing feature importance.
lime_explanation.png: LIME visualizations for 10 instances (5 benign, 5 malignant).
figure3_merged_confusion.png: Merged dataset confusion matrix.
figure4_pr_curve.png: Merged dataset 10 folds Precision-Recall curve.
figure5_roc_curve.png:Merged dataset 10 folds ROC curve.
figure6_lime_importance.png: LIME feature importance.

Repository Structure:
/codes/ python scripts and codes 
/datas/ data files 
/lime analysis report html/ LIME Analysis Report of 10 Individual Instance Explanations
/lime_explanation.jpg/
/figures/ all the figures and flow charts 
requirements.txt: Python dependencies.
README.md: This file.

Datasets
Wisconsin Breast Cancer Dataset (WBCD): Available at https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29 [3].
CBIS-DDSM: Available at https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM [4].
Processed data is provided in data/aligned_breast_cancer_dataset.csv (merged dataset) created by: Pouria Bodaghi

