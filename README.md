# Heart Disease Classification using Machine Learning

## 1. Problem Statement
The objective of this project is to design, implement and compare multiple machine
learning classification models to predict the presence of heart disease in patients
based on clinical attributes. The project also demonstrates end-to-end deployment
of trained machine learning models using a Streamlit web application for interactive
evaluation.

## 2. Dataset Description
The dataset used in this project is the Heart Disease dataset obtained from a public
repository (Kaggle). It contains medical and diagnostic attributes of patients,
which are commonly used for cardiovascular risk assessment.

- Number of instances: 918  
- Number of features: 12 (after preprocessing and feature engineering)  
- Target variable: HeartDisease  
  - 0 -> No heart disease  
  - 1 -> Presence of heart disease  

Basic feature engineering was performed to enhance feature representation and satisfy
the minimum feature size requirement.

## 3. Machine Learning Models Used
The following six classification models were implemented using the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes Classifier  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

## 4. Model Performance Comparison
