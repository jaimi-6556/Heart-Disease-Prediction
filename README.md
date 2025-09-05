# ❤️ Heart Disease Prediction using Machine Learning

This project predicts the likelihood of **heart disease** based on patient health records using different machine learning algorithms.  
The dataset is derived from the **UCI Heart Disease Dataset**.

---

## 📂 Dataset
- Source: UCI Heart Disease Dataset
- Columns:
  - `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure),  
    `chol` (serum cholesterol), `fbs` (fasting blood sugar), `restecg` (resting ECG results),  
    `thalch` (maximum heart rate), `exang` (exercise induced angina),  
    `oldpeak` (ST depression), `slope` (slope of ST segment), `ca` (major vessels), `thal` (thalassemia)  
  - **Target**: `num` → converted to binary:
    - `0` → No Heart Disease  
    - `1–4` → Presence of Heart Disease

---

## ⚙️ Project Workflow
1. **Data Preprocessing**
   - Handle missing values with median/mode imputation
   - Encode categorical variables
   - Standardize numerical features

2. **Exploratory Data Analysis (EDA)**
   - Target distribution visualization
   - Feature vs Target plots (age, sex, cholesterol, chest pain, etc.)
   - Correlation heatmap

3. **Model Training**
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - XGBoost (optional, if installed)

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - ROC Curve & AUC
   - Model comparison (bar chart of accuracies)

---

## 📊 Results
- Random Forest, KNN, and SVM generally perform the best.  
- ROC curves and AUC values are included for detailed evaluation.  

Example accuracy comparison:

| Model              | Accuracy |
|--------------------|----------|
| Logistic Regression| ~84%     |
| Decision Tree      | ~76%     |
| Random Forest      | ~87%     |
| KNN                | ~91%     |
| SVM                | ~88%     |

## 🚀 How to Run
```bash
# Clone this repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the training script
python heart_disease_ml.py

📈 Visualizations

Target distribution plots

Age & cholesterol analysis

Correlation heatmap

Model accuracy comparison bar chart

ROC Curves for all classifiers

🛠 Requirements

Python 3.8+

pandas

numpy

matplotlib

seaborn

scikit-learn

📌 Future Work

Hyperparameter tuning with GridSearchCV / RandomizedSearchCV

Deploy model with Flask / Streamlit

Add SHAP explainability for model interpretability
