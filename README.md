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
| Model                    | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
| ------------------------ | -------- | ----- | --------- | ------ | ----- | ----- |
| Logistic Regression      | 0.887    | 0.932 | 0.891     | 0.906  | 0.898 | 0.771 |
| Decision Tree            | 0.804    | 0.803 | 0.825     | 0.819  | 0.822 | 0.605 |
| KNN                      | 0.891    | 0.935 | 0.898     | 0.906  | 0.902 | 0.780 |
| Naive Bayes              | 0.909    | 0.947 | 0.920     | 0.913  | 0.917 | 0.816 |
| Random Forest (Ensemble) | 0.891    | 0.944 | 0.898     | 0.906  | 0.902 | 0.780 |
| XGBoost (Ensemble)       | 0.874    | 0.931 | 0.889     | 0.882  | 0.885 | 0.745 |

## 5. Model Performance Observations
| ML Model Name            | Observation about Model Performance                                                                                                                                                                                               |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Logistic Regression provided strong and consistent baseline performance with high recall and F1-score, indicating its effectiveness in capturing linear relationships within the dataset while maintaining good interpretability. |
| Decision Tree            | Decision Tree showed comparatively lower performance due to overfitting, as it tends to learn noise from the training data, resulting in reduced generalization capability on unseen data.                                        |
| KNN                      | KNN achieved high accuracy and F1-score when combined with proper feature scaling, demonstrating its ability to capture local patterns in the data but with increased sensitivity to feature distribution.                        |
| Naive Bayes              | Naive Bayes achieved the highest overall performance across multiple metrics, including accuracy, AUC and MCC indicating strong probabilistic modeling and robustness to feature independence assumptions.                      |
| Random Forest (Ensemble) | Random Forest delivered stable and reliable performance by combining multiple decision trees, effectively reducing overfitting and improving generalization compared to a single decision tree.                                   |
| XGBoost (Ensemble)       | XGBoost demonstrated competitive performance through gradient boosting, efficiently handling complex relationships, though its performance was slightly lower than Random Forest due to dataset size and feature characteristics. |

