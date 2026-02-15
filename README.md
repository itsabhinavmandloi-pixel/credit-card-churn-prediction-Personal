# 💳 Credit Card Customer Churn Prediction

A comprehensive machine learning solution for predicting customer churn in credit card services using multiple classification algorithms and an interactive Streamlit web application.

# 💳 Credit Card Customer Churn Prediction

A comprehensive machine learning solution for predicting customer churn in credit card services using multiple classification algorithms and an interactive Streamlit web application.

---

## 🚀 Live Deployment

**Streamlit Web Application:** [https://itsabhinavmandloi-pixel-credit-card-churn-prediction-app-w8aev8.streamlit.app/](https://itsabhinavmandloi-pixel-credit-card-churn-prediction-app-w8aev8.streamlit.app/)

**GitHub Repository:** [https://github.com/itsabhinavmandloi-pixel/credit-card-churn-prediction-Personal](https://github.com/itsabhinavmandloi-pixel/credit-card-churn-prediction-Personal)

---

## 📋 Table of Contents

- [Live Deployment](#live-deployment) 
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Models Used](#models-used)
- [Performance Comparison](#performance-comparison)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Author](#author)

---

## 🚀 Live Deployment

**🌐 Streamlit Web Application:** [https://itsabhinavmandloi-pixel-credit-card-churn-prediction-app-w8aev8.streamlit.app/](https://itsabhinavmandloi-pixel-credit-card-churn-prediction-app-w8aev8.streamlit.app/)

**📂 GitHub Repository:** [https://github.com/itsabhinavmandloi-pixel/credit-card-churn-prediction-Personal](https://github.com/itsabhinavmandloi-pixel/credit-card-churn-prediction-Personal)

> ⭐ **Try the live demo!** Upload customer data and get real-time churn predictions across 6 different ML models.

---


## 🎯 Problem Statement

A bank is experiencing increasing customer churn in their credit card services. The business objective is to build a machine learning model that can predict which customers are likely to churn, enabling the bank to proactively engage with at-risk customers and improve retention through targeted interventions.

---

## 📊 Dataset Description

**Source:** [Kaggle - Credit Card Customers Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)

### Dataset Characteristics

- **Total Instances:** 10,127 customers
- **Total Features:** 20 features (18 used after removing Naive Bayes columns and CLIENTNUM)
- **Target Variable:** Attrition_Flag (Binary Classification)
  - Attrited Customer (Churned): **16.07%**
  - Existing Customer (Retained): **83.93%**

### Key Features

#### Customer Demographics
- `Customer_Age` - Age of the customer
- `Gender` - Male or Female
- `Dependent_count` - Number of dependents
- `Education_Level` - Educational qualification
- `Marital_Status` - Marital status

#### Account Information
- `Card_Category` - Type of card (Blue, Silver, Gold, Platinum)
- `Months_on_book` - Period of relationship with bank
- `Credit_Limit` - Credit limit on the card
- `Income_Category` - Annual income category

#### Transaction Behavior
- `Total_Trans_Amt` - Total transaction amount (Last 12 months)
- `Total_Trans_Ct` - Total transaction count (Last 12 months)
- `Avg_Utilization_Ratio` - Average card utilization ratio

#### Relationship Metrics
- `Contacts_Count_12_mon` - Number of contacts (Last 12 months)
- `Months_Inactive_12_mon` - Number of months inactive (Last 12 months)
- `Total_Relationship_Count` - Total number of products held by customer

> **Note:** The dataset presents a class imbalance challenge with only 16% churned customers, making it important to use appropriate evaluation metrics beyond accuracy.

---

## 🤖 Models Used

This project implements and compares **6 different machine learning algorithms**:

1. **Logistic Regression** - Linear classification model
2. **Decision Tree** - Tree-based classifier
3. **K-Nearest Neighbors (KNN)** - Instance-based learning
4. **Naive Bayes** - Probabilistic classifier
5. **Random Forest** - Ensemble method (Bagging)
6. **XGBoost** - Ensemble method (Gradient Boosting)

---

## 📈 Performance Comparison

### Model Metrics Table

| Rank | ML Model Name           | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|:----:|:------------------------|:--------:|:------:|:---------:|:------:|:--------:|:------:|
| 6    | Logistic Regression     | 84.90%   | 0.9165 | 0.5188    | 0.8062 | 0.6313   | 0.5627 |
| 3    | Decision Tree           | 94.03%   | 0.9183 | 0.7898    | 0.8554 | 0.8213   | 0.7864 |
| 4    | K-Nearest Neighbors     | 90.62%   | 0.8790 | 0.8261    | 0.5262 | 0.6429   | 0.6119 |
| 5    | Naive Bayes             | 88.06%   | 0.8415 | 0.6361    | 0.5969 | 0.6159   | 0.5456 |
| 2    | Random Forest           | 95.16%   | 0.9832 | 0.8514    | 0.8462 | 0.8488   | 0.8200 |
| **🏆 1** | **XGBoost (Best)**  | **97.14%** | **0.9922** | **0.9211** | **0.8985** | **0.9097** | **0.8927** |

> **📊 Metrics Legend:**
> - **Accuracy**: Overall correctness of predictions
> - **AUC**: Area Under ROC Curve - discriminative ability
> - **Precision**: Ratio of correct positive predictions
> - **Recall**: Ratio of actual positives correctly identified
> - **F1 Score**: Harmonic mean of Precision and Recall
> - **MCC**: Matthews Correlation Coefficient - quality measure for imbalanced data

### Model Performance Analysis

#### 🟢 Logistic Regression
Achieved **84.90% accuracy** with strong AUC of 0.9165, demonstrating good discriminative ability. The model shows high recall (0.8062) indicating effective identification of churning customers, though precision (0.5188) suggests moderate false positives. Suitable as a baseline model with interpretable coefficients for business insights.

#### 🟢 Decision Tree
Strong performance with **94.03% accuracy** and balanced F1 score of 0.8213. The model exhibits excellent recall (0.8554) for detecting churn cases while maintaining good precision (0.7898). MCC of 0.7864 confirms robust performance on imbalanced data. Tree structure provides interpretable decision rules.

#### 🟢 K-Nearest Neighbors
Achieved **90.62% accuracy** with the highest precision (0.8261) among non-ensemble models, indicating minimal false alarms. However, lower recall (0.5262) suggests some churning customers are missed. Distance-based approach effectively captures local patterns in customer behavior space.

#### 🟢 Naive Bayes
Moderate performance with **88.06% accuracy** and AUC of 0.8415. Balanced precision (0.6361) and recall (0.5969) result in F1 score of 0.6159. Despite the feature independence assumption, the probabilistic model provides reasonable baseline performance with fast training and prediction times.

#### 🟢 Random Forest (Ensemble)
Outstanding ensemble performance with **95.16% accuracy** and excellent AUC of 0.9832. Well-balanced precision (0.8514) and recall (0.8462) yield F1 score of 0.8488. MCC of 0.8200 demonstrates strong correlation between predictions and actual outcomes. Random Forest effectively handles feature interactions and provides robust predictions through bootstrap aggregation.

#### 🏆 XGBoost (Ensemble) - **BEST MODEL**
Best overall performance with **97.14% accuracy** and exceptional AUC of 0.9922, indicating near-perfect ranking ability. Outstanding precision (0.9211) and recall (0.8985) result in F1 score of 0.9097. Highest MCC of 0.8927 confirms superior predictive power. Gradient boosting with scale_pos_weight effectively handles class imbalance, making it the **optimal choice for production deployment**.

---

## 📁 Repository Structure

```
credit-card-churn-prediction/
│
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── runtime.txt                         # Streamlit Cloud Python runtime hint
├── README.md                           # Project documentation
├── model/                              # Trained model files directory
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── k_nearest_neighbors_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl                      # StandardScaler for preprocessing
│   ├── label_encoders.pkl              # Label encoders for categorical variables
│   └── train_models.py                 # Training + evaluation source code
└── test_data_sample.csv                # Sample test data for app demo
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/itsabhinavmandloi-pixel/credit-card-churn-prediction-Personal.git
   cd credit-card-churn-prediction-Personal
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### Running the Streamlit Application

1. **Start the application:**
   ```bash
   streamlit run app.py
   ```

2. **Access the application:**
   - Open your web browser and navigate to `http://localhost:8501`
   - The application will automatically open in your default browser

### Using the Application

1. **Select a Model:** Choose from 6 available ML models in the dropdown
2. **Upload Data:** Upload a CSV file containing customer data
3. **View Predictions:** See churn predictions and probabilities
4. **Analyze Results:** Review confusion matrix, metrics, and classification report (if ground truth labels are provided)

### Expected CSV Format

Your CSV file should contain the following columns:

```
Customer_Age, Gender, Dependent_count, Education_Level, Marital_Status, 
Income_Category, Card_Category, Months_on_book, Total_Relationship_Count, 
Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit, 
Total_Revolving_Bal, Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1, 
Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio
```

Optionally include `Attrition_Flag` for model evaluation.

---

## ✨ Features

- ✅ **Multi-Model Comparison** - Compare 6 different ML algorithms
- ✅ **Interactive Web Interface** - User-friendly Streamlit dashboard
- ✅ **Real-time Predictions** - Instant churn predictions on uploaded data
- ✅ **Comprehensive Metrics** - Accuracy, AUC, Precision, Recall, F1, MCC
- ✅ **Visual Analytics** - Confusion matrix and data visualization
- ✅ **Model Evaluation** - Detailed classification reports
- ✅ **Easy Deployment** - Simple setup and execution

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|:-----------|:-------:|:--------|
| Python | 3.8+ | Programming language |
| Streamlit | 1.41.1 | Web application framework |
| Scikit-learn | 1.6.1 | Machine learning models |
| XGBoost | 2.0.3 | Gradient boosting model |
| Pandas | 2.2.3 | Data manipulation |
| NumPy | 2.2.3 | Numerical operations |
| Matplotlib | 3.10.0 | Data visualization |
| Seaborn | 0.13.2 | Statistical visualization |

---

## 👨‍💻 Author

**Abhinav Mandloi**  
M.Tech in Artificial Intelligence & Machine Learning  
BITS Pilani - Work Integrated Learning Programmes  
**BITS ID:** 2025aa05473

**Project:** ML Assignment 2 - Credit Card Customer Churn Prediction  
**Submission Date:** February 15, 2026

---

## 📄 License

This project is developed for academic purposes as part of M.Tech ML Assignment.

---

## 🤝 Contributing

This is an academic project. For suggestions or improvements, please feel free to reach out.

---

<div align="center">
  <b>⭐ If you find this project useful, please consider giving it a star! ⭐</b>
</div>
