# Sampling_102103365
This repository contains the code for a sampling assignment that involves handling imbalanced datasets, creating balanced class datasets, applying various sampling techniques, and evaluating their impact on machine learning models.

# Code Overview:
### 1. Import Libraries:
Import necessary libraries, including pandas for data manipulation, scikit-learn for machine learning models, imbalanced-learn for sampling techniques, and others.

### 2. Generate Imbalanced Dataset:
Use make_classification from scikit-learn to generate an imbalanced dataset with 772 samples.

### 3.Create samples:
Divide the dataset into five samples using the sample size detection formula. Each sample is split into training and testing sets.

### 4.Define Models:
Specify machine learning models to be used: Decision Tree, Random Forest, Support Vector Classifier (SVC), Gradient Boosting, and k-Nearest Neighbors.

### 5.Define Sampling Techniques:
Including RandomOverSampler, SMOTE, EasyEnsemble, ClusterCentroids, EditedNearestNeighbours, and InstanceHardnessThreshold. Note that EasyEnsembleClassifier is initialized with specific parameters.

### 6.Model Training and Evaluation:
Iterate through the samples and apply each sampling technique to train the models.For EasyEnsembleClassifier, train the model directly, as it doesn't use fit_resample.For other sampling techniques, use fit_resample to oversample/undersample the training data.

# Discussion:
## DecisionTreeClassifier:
- Best Technique: RandomOverSampler yields the highest accuracy.
- EasyEnsembleClassifier: Slightly lower accuracy but still competitive.

## RandomForestClassifier:
- Best Technique: RandomOverSampler and SMOTE are effective, while EasyEnsembleClassifier provides a balance.
- ClusterCentroids: Significant drop in accuracy.

## SVC:
- Best Technique: RandomOverSampler and SMOTE.
- EditedNearestNeighbours: Surprisingly high accuracy compared to other models.

## GradientBoostingClassifier:
- Best Technique: RandomOverSampler and ENN.
- EasyEnsembleClassifier: Substantially lower accuracy compared to other models.

## KNeighborsClassifier:
- Best Technique: InstanceHardness Threshold works best.
- EasyEnsembleClassifier: Substantially lower accuracy compared to other models.
