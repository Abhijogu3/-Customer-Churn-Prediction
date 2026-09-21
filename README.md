# 📊 Customer Churn Prediction

An end-to-end **Machine Learning project** for predicting telecom customer churn and identifying customers who are at higher risk of leaving the business.

The project builds a reusable preprocessing and modeling pipeline, compares multiple classification algorithms, addresses class imbalance, tunes a Random Forest model, and applies **SHAP-based model explainability**.

---

## 🎯 Business Problem

Customer churn directly affects recurring revenue and customer lifetime value.

The objective of this project is to use customer demographics, subscribed services, account information, contract details, and billing behavior to predict whether a customer is likely to churn.

The resulting churn probability can help retention teams:

- Identify customers at high risk of churn
- Prioritize customers for proactive intervention
- Design targeted retention campaigns
- Reduce potential revenue loss
- Improve customer lifetime value

---

## 📁 Dataset

The project uses the **Telco Customer Churn dataset**:

`Telco_Customer_Churn.csv`

| Metric | Value |
|---|---:|
| Original Records | 7,043 |
| Original Columns | 21 |
| Target Variable | Churn |
| Non-Churn Customers | 5,163 |
| Churn Customers | 1,869 |
| Churn Rate | ~27% |
| Rows Removed During Cleaning | 11 |

The dataset is **imbalanced**, with approximately **27% churn customers** and **73% non-churn customers**.

### Features Used

**Numerical Features**

- `MonthlyCharges`
- `TotalCharges`
- `tenure`

**Categorical Features**

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`

`customerID` is excluded because it is an identifier rather than a predictive customer characteristic.

---

## 🧹 Data Preprocessing

The following preprocessing steps are performed:

1. Load the Telco Customer Churn dataset.
2. Convert `TotalCharges` from object/string to numeric.
3. Remove 11 records where `TotalCharges` could not be converted.
4. Separate numerical and categorical features.
5. Convert the target variable:
   - `Yes` → `1`
   - `No` → `0`
6. Create a **70/30 stratified train-test split**.
7. Build reusable preprocessing pipelines using `ColumnTransformer`.

### Numerical Pipeline

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
```

### Categorical Pipeline

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
```

Using pipelines ensures that the same preprocessing transformations are consistently applied during both model training and prediction.

---

## 🤖 Machine Learning Models

Three classification algorithms are compared:

### 1. Logistic Regression

```python
LogisticRegression(
    class_weight='balanced',
    random_state=42
)
```

`class_weight='balanced'` helps compensate for the unequal distribution between churn and non-churn customers.

**5-Fold Cross-Validation ROC-AUC: ~0.848**

---

### 2. Random Forest

The initial Random Forest model uses:

- 200 estimators
- Balanced class weights

**Initial 5-Fold Cross-Validation ROC-AUC: ~0.822**

The model is subsequently optimized using `GridSearchCV`.

#### Best Random Forest Parameters

```text
max_depth         = 10
min_samples_leaf  = 5
min_samples_split = 2
n_estimators      = 400
```

**Best Random Forest CV ROC-AUC: 0.8381**

---

### 3. XGBoost

XGBoost is trained with class-imbalance handling using `scale_pos_weight`.

```python
XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
```

---

## 📈 Model Comparison

| Model | ROC-AUC | F1 Score | Precision | Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | **0.86** | 0.64 | 0.52 | **0.84** |
| Random Forest | 0.85 | **0.65** | **0.56** | 0.78 |
| XGBoost | **0.86** | 0.64 | 0.54 | 0.79 |

---

## 📊 XGBoost Test Performance

| Metric | Score |
|---|---:|
| Accuracy | ~77% |
| ROC-AUC | **0.8574** |
| Churn Precision | 0.54 |
| Churn Recall | 0.79 |
| Churn F1 Score | 0.64 |

### Classification Report

```text
              precision    recall    f1-score    support

           0       0.91      0.76       0.83       1549
           1       0.54      0.79       0.64        561

    accuracy                           0.77       2110
```

---

## 🏆 Model Interpretation

There is no single model that performs best across every metric.

### Logistic Regression

- Highest churn **Recall: 0.84**
- Useful when failing to identify a potential churner is costly
- Strong overall ROC-AUC performance

### Random Forest

- Highest **F1 Score: 0.65**
- Highest **Precision: 0.56**
- Good balance between precision and recall

### XGBoost

- **ROC-AUC: 0.8574**
- **Recall: 0.79**
- Strong ability to distinguish churners from non-churners

For a real retention campaign, model selection should depend on the business cost associated with **false negatives and false positives**, rather than accuracy alone.

---

## 🔍 Model Explainability

The project uses:

- **Random Forest Feature Importance**
- **SHAP (SHapley Additive exPlanations)**

SHAP helps explain:

- Which customer characteristics have the greatest influence on churn
- Which features increase predicted churn probability
- Which features decrease predicted churn probability
- Why individual customers receive particular churn predictions

The notebook includes both **SHAP summary plots** and **feature-importance visualizations**.

---

## 💼 Business Use Case

The trained machine learning model can assign each customer a **churn probability**.

Retention teams can use these probabilities to:

- Rank customers according to churn risk
- Identify high-risk customers
- Prioritize retention campaigns
- Create targeted offers
- Investigate common churn characteristics
- Reduce unnecessary retention spending on low-risk customers
- Monitor customer churn risk over time

### Retention Workflow

```text
Customer Data
      ↓
Data Preprocessing
      ↓
Machine Learning Model
      ↓
Churn Probability
      ↓
Customer Risk Prioritization
      ↓
Retention Campaign
      ↓
Reduced Customer Churn
```

---

## 🧠 Evaluation Metrics

Because approximately **27% of customers churn**, accuracy alone can provide an incomplete view of model performance.

### Recall

Measures how many actual churners the model successfully identifies.

### Precision

Measures how many customers predicted to churn actually belong to the churn class.

### F1 Score

Balances precision and recall into a single metric.

### ROC-AUC

Measures the model's ability to distinguish churn customers from non-churn customers across different classification thresholds.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Jupyter Notebook
- Joblib

---

## 📂 Repository Structure

```text
Customer-Churn-Prediction/
│
├── README.md
├── Customer_Churn_Prediction.ipynb
├── Telco_Customer_Churn.csv
├── requirements.txt
│
├── models/
│   └── xgb_model.pkl
│
└── images/
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── feature_importance.png
    └── shap_summary.png
```

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Customer-Churn-Prediction
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib jupyter
```

### 3. Configure Dataset Path

Keep `Telco_Customer_Churn.csv` in the same directory as the notebook.

```python
data = pd.read_csv("Telco_Customer_Churn.csv")
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Customer_Churn_Prediction.ipynb
```

Run all notebook cells in sequence.

---

## 💾 Model Saving

The trained XGBoost pipeline is saved using Joblib:

```python
joblib.dump(xgb_model, "xgb_model.pkl")
```

Because preprocessing and the classifier are included in the pipeline, the saved model can apply the required transformations before generating predictions.

---

## 🚀 Future Improvements

Potential extensions include:

- Threshold optimization based on retention costs
- Precision-recall curve analysis
- Advanced XGBoost hyperparameter tuning
- Additional feature engineering
- Probability calibration
- Streamlit deployment
- FastAPI model API
- Interactive churn-risk dashboard
- Automated model monitoring
- Automated model retraining
- CRM integration for retention campaigns

---

## 📌 Key Takeaway

This project demonstrates an **end-to-end machine learning classification workflow** for solving a real-world customer retention problem.

**Data Cleaning → Preprocessing → Class Imbalance Handling → Model Training → Model Comparison → Hyperparameter Tuning → Evaluation → Explainability → Model Saving**

The tested machine learning models achieve approximately **0.85–0.86 ROC-AUC**, demonstrating useful predictive performance for identifying customers at risk of churn.

---

## 👤 Author

**Abhishek Jogu**

Data Analyst | SQL | Python | Power BI | Machine Learning

⭐ If you find this project useful, consider giving the repository a star.
