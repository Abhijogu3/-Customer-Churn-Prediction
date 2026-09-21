###📊### Customer Churn Prediction
An end-to-end machine learning project for predicting telecom customer churn and identifying customers who are at higher risk of leaving the business.

The project builds a reusable preprocessing and modeling pipeline, compares multiple classification algorithms, addresses class imbalance, tunes a Random Forest model, and applies SHAP-based model explainability.

🎯 Business Problem
Customer churn directly affects recurring revenue and customer lifetime value. The goal of this project is to use customer demographics, subscribed services, account information, contract details, and billing behavior to predict whether a customer is likely to churn.

The resulting churn probability can help a retention team prioritize customers for proactive intervention.

📁 Dataset
The notebook uses Telco_Customer_Churn.csv.

Item Value

Original records 7,043 Original columns 21 Target Churn Non-churn customers 5,163 Churn customers 1,869 Churn rate ~27% Rows removed during TotalCharges cleaning 11

The dataset is imbalanced: approximately 27% of customers churn while approximately 73% do not.

Features used
Numerical features - MonthlyCharges - TotalCharges - tenure

Categorical features - gender - SeniorCitizen - Partner - Dependents - PhoneService - MultipleLines - InternetService - OnlineSecurity - OnlineBackup - DeviceProtection - TechSupport - StreamingTV - StreamingMovies - Contract - PaperlessBilling - PaymentMethod

customerID is not used as a predictive feature.

🧹 Data Preprocessing
The notebook performs the following preprocessing steps:

Loads the Telco customer churn dataset.
Converts TotalCharges from string/object to numeric.
Removes 11 records where TotalCharges could not be converted.
Separates numerical and categorical features.
Maps the target:
Yes → 1
No → 0
Creates a 70/30 stratified train-test split.
Uses a ColumnTransformer and Scikit-learn pipelines to prevent inconsistent preprocessing.
Numerical pipeline
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
Categorical pipeline
Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
This design makes the preprocessing reusable and ensures that transformations applied during training are also applied during prediction.

🤖 Machine Learning Models
Three classification algorithms are compared:

1. Logistic Regression
Logistic Regression is used with:

LogisticRegression(
    class_weight='balanced',
    random_state=42
)
class_weight='balanced' helps compensate for the unequal distribution of churn and non-churn customers.

The notebook reports a 5-fold cross-validation ROC-AUC of approximately 0.848 for Logistic Regression.

2. Random Forest
The initial Random Forest uses 200 estimators and balanced class weights.

Its reported 5-fold cross-validation ROC-AUC is approximately 0.822.

The model is then tuned using GridSearchCV.

Best Random Forest parameters found:

max_depth        = 10
min_samples_leaf = 5
min_samples_split= 2
n_estimators     = 400
Best reported Random Forest CV ROC-AUC:

0.8381
3. XGBoost
XGBoost is trained with class-imbalance handling through scale_pos_weight.

Key configuration:

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
📈 Model Comparison
The final comparison table recorded in the notebook is:

Model ROC-AUC F1 Score Precision Recall

Logistic Regression 0.86 0.64 0.52 0.84 Random Forest 0.85 0.65 0.56 0.78 XGBoost 0.86 0.64 0.54 0.79

XGBoost detailed test results
XGBoost achieved:

Accuracy: ~77%
ROC-AUC: 0.8574
Churn Precision: 0.54
Churn Recall: 0.79
Churn F1 Score: 0.64
              precision    recall  f1-score   support

           0       0.91      0.76      0.83      1549
           1       0.54      0.79      0.64       561

    accuracy                           0.77      2110
🏆 Model Interpretation
There is no single winner for every metric:

Logistic Regression provides the highest reported churn recall (0.84), which is useful when missing a churner is expensive.
Random Forest provides the strongest reported F1 score (0.65) and precision (0.56) in the final comparison.
XGBoost achieves a strong ROC-AUC of 0.8574 while maintaining churn recall of 0.79.
For a retention use case, model selection should depend on the business cost of false negatives versus false positives, rather than accuracy alone.

🔍 Model Explainability
The project extracts Random Forest feature importance and also uses SHAP (SHapley Additive exPlanations) to explain model predictions.

SHAP helps answer questions such as:

Which customer characteristics have the greatest influence on churn predictions?
Which features push a particular customer toward a higher churn probability?
Which features reduce predicted churn risk?
Both SHAP summary visualization and bar-based feature importance visualization are included in the notebook.

💼 Business Use Case
A trained churn model can assign each customer a probability of churn.

A retention team could use these probabilities to:

Rank customers by churn risk.
Prioritize high-risk customers for retention campaigns.
Design targeted offers.
Investigate the characteristics associated with churn.
Reduce unnecessary retention spending on low-risk customers.
Track churn-risk scores over time.
Example retention workflow
Customer Data
      ↓
Data Preprocessing
      ↓
Trained Churn Model
      ↓
Churn Probability
      ↓
Customer Risk Prioritization
      ↓
Retention Action
🧠 Why ROC-AUC, Recall and F1 Matter
Because only about 27% of customers in the dataset churn, accuracy alone can be misleading.

For this project:

Recall measures how many actual churners the model successfully identifies.
Precision measures how many customers predicted to churn actually churn.
F1 Score balances precision and recall.
ROC-AUC evaluates the model's ability to distinguish churners from non-churners across classification thresholds.
🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
SHAP
Jupyter Notebook
Joblib
📂 Suggested Repository Structure
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
▶️ How to Run the Project
1. Clone the repository
git clone <your-repository-url>
cd Customer-Churn-Prediction
2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib jupyter
3. Update the dataset path
The current notebook reads the dataset from a local Windows path. For a GitHub repository, change it to:

data = pd.read_csv("Telco_Customer_Churn.csv")
4. Launch Jupyter
jupyter notebook
Open:

Customer_Churn_Prediction.ipynb
and run the notebook cells.

💾 Model Saving
The notebook saves the trained XGBoost pipeline using Joblib:

joblib.dump(xgb_model, "xgb_model.pkl")
Because preprocessing and the classifier are contained in the same pipeline, the saved model can apply the required preprocessing before generating predictions.

🚀 Future Improvements
Potential extensions include:

Threshold optimization based on retention campaign cost.
Precision-recall curve analysis.
Hyperparameter tuning for XGBoost.
Additional feature engineering.
Probability calibration.
Deployment through Streamlit or FastAPI.
Customer-level churn-risk dashboard.
Automated model monitoring and retraining.
Integration with CRM/customer-retention workflows.
📌 Key Takeaway
This project demonstrates an end-to-end classification workflow for a real business problem: from data cleaning and preprocessing to class-imbalance handling, model comparison, hyperparameter tuning, evaluation, explainability, and model persistence.

The results show that churn can be predicted with useful discriminatory performance, with the tested models reaching approximately 0.85--0.86 ROC-AUC.

👤 Author
Abhishek Jogu

Data Analyst | SQL | Python | Power BI | Machine Learning

If you find this project useful, consider giving the repository a ⭐.
