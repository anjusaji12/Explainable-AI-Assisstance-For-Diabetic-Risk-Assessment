# Explainable-AI-Assisstance-For-Diabetic-Risk-Assessment

##Project Overview

This project presents a clinical decision support system designed to predict diabetic risk categories (Healthy, Pre-Diabetic, or Diabetic) using the BRFSS 2015 dataset. Unlike traditional "black-box" models, this system integrates Explainable AI (XAI) to provide transparency for clinicians and actionable recourse for patients.

##Key Features

Predictive Modeling: Multi-class classification using XGBoost with SMOTE to handle significant class imbalance.

Global/Local Attribution: Utilizes SHAP (SHapley Additive exPlanations) to visualize feature contributions for specific diagnostic outcomes.

Counterfactual Recourse: Implements DiCE (Diverse Counterfactual Explanations) to generate actionable lifestyle changes (e.g., target BMI or blood pressure) required to transition a patient from a high-risk state to a "Healthy" state.

##Methodology

1. Data Preprocessing
 
Source: BRFSS 2015 Health Indicators dataset.

Feature Engineering: Dropped non-clinical features (Education, Income, etc.) to focus on physiological and lifestyle markers.

Sampling: Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the target classes, ensuring the model performs well on minority "Pre-Diabetic" cases.

2. The XAI Stack

SHAP: Used to answer "Why did the model classify this patient as Diabetic?" by breaking down the impact of BMI, Age, and Blood Pressure on the model's output.

DiCE: Used to answer "What is the minimum change needed for this patient to become healthy?" This provides Counterfactual Explanations, which are essential for prescriptive healthcare.

##User Interface

The Streamlit dashboard is divided into three sections:

Patient Input: A sidebar for entering clinical data (BMI, High BP, etc.).

Diagnostic Prediction: Real-time risk probability for Healthy, Pre-Diabetic, and Diabetic status.

Explainability Suite: SHAP Waterfall Plots for local feature importance.


