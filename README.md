# 📊 Customer Churn Prediction Project

---
### 🚀 Executive Summary
This project presents an **end-to-end solution** for **predicting customer churn** in a telecommunications company. Leveraging advanced **machine learning algorithms**, **deep learning**, and **explainability techniques (SHAP)**, we deliver an actionable framework to identify potential churners and support **data-driven retention strategies**.

Our final deliverable is a **high-performance XGBoost model**, fine-tuned and deployed with a **REST API**, accompanied by a **detailed cost-benefit analysis** ensuring **strategic decision-making** to optimize customer lifetime value (CLV).

---

## 📌 Problem Statement
**Customer churn** is a critical challenge in the telecom industry. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project aims to:
- Identify **at-risk customers**.
- Provide **actionable insights** for the business to **mitigate churn**.
- Maximize **retention campaign ROI** through **precision targeting**.

---

## 🌟 Project Objectives
1. **EDA & Feature Engineering**: Understand customer behavior and engineer impactful features.
2. **Model Development**: Build robust predictive models (Logistic Regression, XGBoost, TensorFlow NN).
3. **Class Imbalance Handling**: Ensure fairness by using **class weights** and **threshold tuning**.
4. **Explainability with SHAP**: Deliver **transparent insights** for **C-suite stakeholders**.
5. **Cost-Benefit Analysis**: Justify the retention strategy from a **financial perspective**.
6. **Model Deployment**: Provide a **scalable REST API** for real-time churn prediction.

---

## 📂 Project Structure
```plaintext
customer-churn-prediction-deployment/
🔹🔹 notebooks/
🔹🔹 🔹 customer-churn-check.ipynb       # Final cleaned and structured notebook

🔹 README.md                            # Project overview and instructions
🔹 requirements.txt                     # Python libraries for environment setup
🔹 .gitignore                           # Ignored files and directories
```

---

## 📊 Dataset Overview
- **Source**: IBM Sample Telco Customer Churn Dataset
- **Records**: 7,043 customers
- **Features**: Demographics, Services subscribed, Account Information, Contract Details, Payment Method

---

## 🔍 Exploratory Data Analysis (EDA)
- Clear **class imbalance**: 26% churners vs 74% non-churners.
- Senior citizens, month-to-month contract holders, and those using electronic checks have **higher churn rates**.
- Class imbalance was **quantified and addressed** with **class weights** and **threshold tuning**.

---

## 🔧 Feature Engineering
- **Security Bundle** (Online Security + Backup + Tech Support)
- **Streaming Bundle** (TV + Movies)
- **Average Monthly Cost**
- **High-Risk Customer Tagging**

---

## 🧠 Machine Learning Models
### 1. Logistic Regression (Baseline)
- Precision: 0.52
- Recall: 0.79
- ROC AUC: 0.8442
- Insights: **High recall**, good for catching churners early but increased false positives.

### 2. XGBoost (Final Model)
- Tuned hyperparameters via RandomizedSearchCV.
- Precision: 0.53
- Recall: 0.77
- ROC AUC: 0.8457
- **Cost-Benefit Analysis** identified it as the **optimal model**.

### 3. TensorFlow Neural Network
- Precision: 0.53
- Recall: 0.75
- ROC AUC: 0.8419
- Competitive, but with **higher deployment and maintenance costs**.

---

## ⚖️ Threshold Tuning & Class Imbalance Handling
- Adjusted thresholds to optimize **Recall ≥ 80%**, ensuring **early detection** of churners.
- Precision-Recall tradeoffs analyzed for **strategic business alignment**.

---

## 💸 Cost-Benefit Analysis

| **Model**         | **False Positive Cost** | **False Negative Cost** | **Total Cost** |
|-------------------|-------------------------|-------------------------|----------------|
| Logistic Regression | ₹138,500               | ₹390,000               | ₹528,500       |
| TensorFlow Neural Network | ₹125,000               | ₹460,000               | ₹585,000       |
| **XGBoost (Selected)**   | ₹128,000               | ₹425,000               | **₹553,000**   |

---

## ✅ Why We Selected **XGBoost** as the Final Model

While **Logistic Regression** demonstrated a marginally lower **Total Cost**, our recommendation favors **XGBoost** for the following reasons:

### 🔥 Superior Model Performance
- XGBoost consistently delivers **high recall**, which is critical for reducing customer churn risk.
- **Balanced precision-recall trade-off** at optimized thresholds, ensuring both retention efficiency and cost management.

### 🚀 Scalability and Efficiency
- **Optimized for speed and scalability**, XGBoost efficiently handles **large-scale datasets** common in **telco industry** scenarios.
- **Parallel processing** and optimized computation make it ideal for **real-time deployment** and **high-volume predictions**.

### 📈 Handles Complexity
- Captures **non-linear feature interactions** and **complex data patterns** that simpler models like Logistic Regression may miss.
- Robust against **imbalanced datasets**, supported by **built-in handling of missing values** and **custom class weights**.

### 🔎 Interpretability with SHAP
- **SHAP (SHapley Additive exPlanations)** enables transparent decision-making by **explaining feature contributions** at both global and individual levels.
- Ensures compliance with **regulatory** and **stakeholder transparency** requirements.

### 🌐 Industry Proven
- Extensively used in **real-world telecom deployments** by leaders such as **Vodafone**, **Jio**, and **Verizon**.
- Validated across multiple **industries and applications** for **customer retention** and **churn prediction**.

---

## 🔎 Explainability with SHAP
- **Contract type**, **tenure**, and **InternetService** are the top drivers of churn.
- SHAP summary plots provided **C-suite explainability** and **trust in AI decisions**.

---

## 🚀 Deployment: Flask REST API (demonstration of how the model could be deployed)
A **Flask API** can be built to serve real-time predictions:
```bash
# Run the API server
python app.py

# Endpoint
POST /predict
Payload (JSON): [{...customer data...}]
```

---

## 📝 Requirements
```bash
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
xgboost
tensorflow
shap
joblib
flask
```

---

## ✅ Key Takeaways
1. **XGBoost** is recommended for **production deployment**, offering a **balanced performance** and **lowest churn cost**.
2. **SHAP explanations** help align **technical outputs** with **business expectations**.
3. Framework is **scalable**, **interpretable**, and ready for **production deployment**.

---

## 🌟 Recommendations to Stakeholders
- Prioritize **Month-to-Month** customers for retention campaigns.
- Focus on customers with **Fiber Optic** internet and **Electronic Check** payments.
- Incentivize **long-term contracts** to reduce churn.

## 💡 Final Recommendation
➡️ **Deploy XGBoost as the production-grade churn prediction model**, coupled with **SHAP explainability** for actionable insights.
➡️ Ensure **ongoing monitoring** and **periodic retraining** to maintain model relevance as customer behavior evolves.

---

## 🏆 Consultant's Note
This solution not only delivers **technical rigor**, but also ensures **strategic alignment** with **business objectives**, making it a **consulting-grade framework** ready for **stakeholder presentations** and **implementation**.

---

📝 What's Next?
1. Model Retraining every quarter
2. Cost-benefit reassessment based on updated churn behavior
3. Integration with CRM for automated targeting

---

📒 Additional Details
All step-by-step methodologies, analyses, and recommendations are thoroughly documented in the notebook markdown.
This provides clarity, reproducibility, and business relevance, making it a highly valuable reference for decision-makers.

---

## 👨‍💻 Author
**Pratham Sharma**
- MBA (SPJIMR) | BSc (IIT Madras) | B.Tech (CSE)
- AI Consultant and Data Scientist  
- [LinkedIn](https://www.linkedin.com/in/pratham-sharma-spjimr)  
- [Kaggle](https://www.kaggle.com/theprathamsharma1)
