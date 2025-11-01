# Hospital Readmission Prediction Pipeline

## 📋 Project Overview

This project develops a machine learning pipeline to predict whether patients would be readmitted to hospitals within 30 days after initial hospitalization. The project uses anonymized patient information from three hospitals to build and evaluate predictive models.

## 🎯 Business Objective

The primary goal is to predict patient readmission within 30 days using a binary classification model. This helps healthcare providers:

- Identify high-risk patients early
- Improve patient care planning
- Reduce healthcare costs
- Allocate resources more effectively

**Target Variable:** `Readmission within 30 days` (1 = readmitted, 0 = not readmitted)

## 📊 Dataset

- **Training Dataset:** `healthcare_readmissions_train.csv`
- **Total Observations:** 8,038
- **Features:** 22 variables (21 predictors + 1 response variable)
- **Response Variable:** Binary readmission flag

### Feature Categories

#### 1. Demographic Variables
- `Age`: Patient age
- `Gender`: Male or Female
- `Ethnicity`: Caucasian, Hispanic, African American, Other

#### 2. Lifestyle Variables
- `Living Situation`: Lives Alone, Lives with Family, Assisted Living
- `Exercise Frequency`: None, Occasional, Regular
- `Diet Type`: Balanced, High-fat, Vegetarian, Other

#### 3. Clinical Variables
- `Height`: Patient height in meters
- `Weight`: Patient weight in kg
- `Adjusted Weight`: Health system-specific adjustments to patient weight (kg)
- `BMI`: Body Mass Index
- `Smoker`: Boolean indicating if patient is a current smoker
- `Has Diabetes`: Boolean (0 or 1)
- `Has Hypertension`: Boolean (0 or 1)
- `Has Chronic Kidney Disease`: Boolean (0 or 1)

#### 4. Utilization Variables
- `Number of Prior Visits`: Number of previous hospitalizations
- `Medications Prescribed`: Number of different prescription medications
- `Length of Stay`: Length of hospital stay in days
- `Type of Treatment`: None, Minor Surgery, Major Surgery, Other Treatment
- `Hospital ID`: Unique identifier of hospitals (Hosp1, Hosp2, Hosp3)
- `PatientID`: Unique patient identifier

## 🔍 Exploratory Data Analysis (EDA)

### Data Quality Issues

**Missing Values:**
- `Number of Prior Visits` / `Medications Prescribed`: Imputed by mean (<30% missing)
- `Type of Treatment` / `Exercise Frequency`: Imputed with new category 'missing value' (>30% missing to avoid information bias)

**Data Types:**
- String variables requiring encoding: Gender, Ethnicity, Hospital ID, Smoker, Exercise Frequency, Diet Type, Type of Treatment

### Key Findings

- **Class Distribution:** Moderately unbalanced → SMOTE oversampling required
- **Correlations:** 
  - Weight and Adjusted Weight are highly correlated (r=0.88) but keeping both improves model performance
  - Some clinical variables show right-skewed distributions → log transformation applied
- **Feature Importance:** Variables with low importance include Ethnicity, Type of Treatment, Adjusted Weight, Hospital ID, Height, Gender

## 🔧 Data Preparation

### Feature Engineering

1. **Missing Value Imputation:**
   - Numerical variables (<30% missing): Mean imputation
   - Categorical variables (>30% missing): New 'missing value' category

2. **Transformations:**
   - Log-transform applied to: `Length of Stay`, `Medications Prescribed`, `Number of Prior Visits` (due to positive skew)

3. **Encoding:**
   - Label encoding for: Gender, Ethnicity, Hospital ID, Smoker, Exercise Frequency, Diet Type, Type of Treatment

4. **Scaling:**
   - StandardScaler applied to all variables after encoding

5. **Handling Class Imbalance:**
   - SMOTE oversampling applied to training set
   - Final dataset after SMOTE: 12,181 rows × 18 columns

### Dataset Partitioning

| Split | Rows | Features |
|-------|------|----------|
| X_train | 10,574 | 17 |
| y_train | 10,574 | 1 |
| X_val | 1,608 | 17 |
| y_val | 1,608 | 1 |

## 🤖 Model Development

### Model Comparison

| Model | F1 Score |
|-------|----------|
| Logistic Regression | 0.765 |
| Random Forest | 0.750 |
| XGBoost (Hyperparameter Group 1) | **0.800** |
| XGBoost (Hyperparameter Group 2) | 0.790 |
| Neural Network | 0.755 |
| **AutoML (Ensemble)** | **0.8814** |

### Selected Model: XGBoost

**Best Hyperparameters (Grid Search):**
- `colsample_bytree`: 1.0
- `learning_rate`: 0.1
- `max_depth`: 9
- `n_estimators`: 200
- `subsample`: 0.8

**Threshold:** 0.5 (default, optimal based on experiments)

**Performance:**
- Training F1 Score: 0.800
- Test F1 Score: 0.7658
- ROC AUC: 0.72

### AutoML Model

AutoML created an ensemble model with 25 components:
- 14 Neural Network models
- 11 Boosted Tree models
- Test F1 Score: 0.8814 (superior to manually tuned models)

*Note: For demonstration purposes, the pipeline uses the manually tuned XGBoost model to showcase the complete pipeline building process.*

## 🏗️ Pipeline Architecture

### Training Pipeline

1. **Initial Data Preparation**
   - Load and clean data
   - Handle missing values
   - Encode categorical variables

2. **Age Imputation** (Training & Validation)
   - Mean imputation for missing age values

3. **Feature Engineering**
   - Apply log transformations
   - Scale features using StandardScaler

4. **Clustering** (Optional, for training & validation)

5. **SMOTE Oversampling**
   - Apply only to training set to balance classes

6. **Model Training**
   - Train XGBoost with optimized hyperparameters

7. **Model Assessment**
   - Evaluate on validation set
   - Generate performance metrics

### Inference Pipeline

1. **Initial Data Preparation**
   - Same preprocessing steps as training

2. **Age Imputation**
   - Use imputation parameters learned from training

3. **Feature Engineering**
   - Apply same transformations as training

4. **Clustering** (if applicable)

5. **Prediction**
   - Generate readmission predictions using trained model

## 📈 Model Performance

### Key Improvements

- **SMOTE Impact:** Increased F1 score by ~5%
- **Feature Selection:** Experimentation showed keeping Weight variable (despite correlation with Adjusted Weight) improved performance

### Validation Set Performance

- F1 Score: 0.7658
- ROC AUC: 0.72

### Test Set Performance

- F1 Score: 0.7658
- ROC AUC: 0.72

## 💡 Key Insights & Recommendations

1. **Feature Engineering:**
   - SMOTE oversampling is critical for handling class imbalance
   - Log transformations help normalize skewed distributions

2. **Feature Selection:**
   - High correlation doesn't always mean redundancy (Weight vs Adjusted Weight)
   - Feature importance can guide variable selection
   - Low importance variables: Ethnicity, Type of Treatment, Adjusted Weight, Hospital ID, Height, Gender

3. **Model Selection:**
   - XGBoost performs best among manually tuned models
   - AutoML achieves superior performance but at the cost of interpretability

4. **Future Improvements:**
   - Try more sophisticated ensemble models
   - Integrate AutoML model into inference pipeline
   - Further feature engineering and selection based on domain knowledge

## 🚀 Installation

```bash
# Clone the repository
git clone <repository-url>
cd hospital-readmission-pipeline

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn  # for SMOTE
```

## 📝 Usage

### Training

```python
# Run training pipeline
python train_pipeline.py
```

### Inference

```python
# Run inference pipeline
python inference_pipeline.py --input <input_file.csv>
```

## 📁 Project Structure

```
hospital-readmission-pipeline/
├── README.md
├── requirements.txt
├── data/
│   ├── healthcare_readmissions_train.csv
│   └── healthcare_readmissions_test.csv
├── src/
│   ├── train_pipeline.py
│   ├── inference_pipeline.py
│   ├── preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── models/
│   └── xgboost_model.pkl
└── notebooks/
    └── eda.ipynb
```

## 📊 Results Summary

- **Best Manual Model:** XGBoost with F1 Score of 0.7658
- **Best Overall Model:** AutoML Ensemble with F1 Score of 0.8814
- **SMOTE Impact:** +5% improvement in F1 score
- **Key Features:** Clinical variables (diabetes, hypertension) and utilization variables (medications, prior visits) are most predictive
