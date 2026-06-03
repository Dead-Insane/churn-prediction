# 📊 Customer Churn Prediction & Business Impact Dashboard

A complete end-to-end data science project that predicts telecom customer churn, 
quantifies business revenue at risk, and delivers insights through an interactive 
Streamlit dashboard.

---

## 🎯 Problem Statement

Telecom companies lose significant revenue when customers cancel their subscriptions.
The goal of this project is to:
- Identify customers likely to churn **before** they leave
- Understand **why** they churn
- Quantify the **business impact** in dollar terms
- Provide a deployable tool for business and risk teams

---

## 📁 Project Structure

churn-prediction/
│
├── data/                        # Charts and processed outputs
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Cleaning & Feature Engineering
│   └── 03_modeling.ipynb       # Model Training & Evaluation
├── src/
│   ├── model.pkl                # Saved best model
│   ├── scaler.pkl               # Saved scaler
│   └── feature_columns.pkl     # Saved feature alignment
├── app/
│   └── streamlit_app.py        # Interactive web application
├── requirements.txt
└── README.md

---

## 🔍 Key Findings from EDA

- **Churn rate is ~26%** — dataset is moderately imbalanced
- Customers with **month-to-month contracts** churn significantly more
- **Low tenure customers** (< 12 months) are highest risk
- **Fiber optic** subscribers show higher churn despite premium pricing
- **Higher monthly charges** correlate with increased churn probability

---

## ⚙️ Methodology

### Data Preprocessing
- Converted `TotalCharges` from object to numeric
- Dropped 11 rows with blank TotalCharges values
- Label encoded binary categorical columns
- One-hot encoded multi-class categorical columns
- Standard scaled numerical features (tenure, MonthlyCharges, TotalCharges)

### Models Trained & Compared

| Model | Accuracy | Recall (Churn) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.726 | **0.794** | **0.835** |
| Random Forest | 0.789 | 0.495 | 0.821 |
| XGBoost | 0.753 | 0.684 | 0.815 |

**Best Model: Logistic Regression**
- Chosen based on highest Recall (0.794) and ROC-AUC (0.835)
- Catching churners matters more than raw accuracy in a business context
- A simpler, interpretable model that outperformed ensemble methods

---

## 💰 Business Impact

| Metric | Value |
|---|---|
| High-risk customers identified | 606 |
| Average monthly charge | $64.76 |
| Annual revenue at risk | $470,947 |
| Estimated retainable revenue | **$141,284** |

> Assumes 30% customer retention success rate on flagged high-risk customers

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Dead-Insane/churn-prediction.git
cd churn-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download [IBM Telco Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
and save as `data/telco_churn.csv`

**4. Run notebooks in order**

notebooks/01_eda.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_modeling.ipynb

**5. Launch the Streamlit app**
```bash
cd app
streamlit run streamlit_app.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Deployment | Streamlit |
| Version Control | Git, GitHub |

---

## 👤 Author

**Divya Prakash**  
[LinkedIn](https://linkedin.com/in/divya-prakash-09w) | 
[GitHub](https://github.com/Dead-Insane)
