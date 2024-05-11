# Breast-Cancer-Prediction-Model
This repository contains code for a Breast Cancer Prediction Model developed using machine learning techniques. The model predicts whether breast cancer is benign or malignant based on various features.

# Breast Cancer Prediction Model

This repository contains a breast cancer prediction model built using Artificial Neural Networks (ANN) with the Keras library. The model predicts whether a breast tumor is benign or malignant based on various features extracted from diagnostic images.

## Dataset

The dataset used in this project is obtained from the UCI Machine Learning Repository. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The dataset includes the following columns:

- id
- diagnosis
- radius_mean
- texture_mean
- perimeter_mean
- area_mean
- smoothness_mean
- compactness_mean
- concavity_mean
- concave_points_mean
- symmetry_mean
- fractal_dimension_mean
- radius_se
- texture_se
- perimeter_se
- area_se
- smoothness_se
- compactness_se
- concavity_se
- concave_points_se
- symmetry_se
- fractal_dimension_se
- radius_worst
- texture_worst
- perimeter_worst
- area_worst
- smoothness_worst
- compactness_worst
- concavity_worst
- concave_points_worst
- symmetry_worst
- fractal_dimension_worst

# Usage

1. Enter the required features in the submission form on the web application.
2. Click the "PREDICT" button to get the prediction result.
3. The predicted diagnosis (Benign or Malignant) will be displayed in the footer.

# Models
Two models were developed and evaluated:

Model 1: ANN with 15 input features (selected based on EDA)
Model 2: ANN with all 30 input features
Model 2 provided the highest accuracy and was chosen for deployment.

# Data Preprocessing

The dataset was checked for missing values and duplicates. No missing values or duplicates were found.
Outliers were identified using the Interquartile Range (IQR) method and removed to ensure the model's robustness.

# Exploratory Data Analysis (EDA)

Visualizations such as box plots and heatmaps were used to gain insights into the dataset.
Features with high correlation were identified and redundant features were removed to improve model performance.

# Web Application
The breast cancer prediction model is deployed as a web application using Flask. Users can input the required features through a user-friendly interface and obtain predictions instantly.

# Files

app.py: Flask application for the web interface.
Breast_Cancer_Prediction.ipynb: Script to build, train, and evaluate the ANN models.
data_train.csv: Training dataset used to fit the scaler.
your_model.h5: Pre-trained ANN model.
templates/index.html: HTML template for the web interface.
templates/result.html: HTML template to display prediction results.
static/style.css: CSS stylesheet for styling the web interface.
