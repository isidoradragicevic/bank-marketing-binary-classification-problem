# Term Deposit Subscription Prediction (Bank Marketing Dataset)

This machine learning project aims to predict whether a client will subscribe to a term deposit based on the Bank Marketing dataset. The dataset contains information collected from a Portuguese bank's marketing campaign, and the target variable is binary: whether the client subscribed to a term deposit (`yes` or `no`).

## 📊 Dataset Overview

- **Source:** UCI Machine Learning Repository / Kaggle
- **Target Variable:** `y` – Has the client subscribed to a term deposit?
- **Type:** Binary classification
- **Size:** ~45,000 observations
- **Class Imbalance:** The positive class (`yes`) is significantly underrepresented

## 🔍 Exploratory Data Analysis (EDA)

- Explored dataset structure and summary statistics 
- Identified no missing values and removed duplicates
- Checked class distribution (imbalanced classes)
- Assessed feature types and class-wise patterns to retain meaningful variables. 
- Visualized numerical and categorical features
- Encoded categorical variables and scaled numerical ones

## ⚙️ Preprocessing

- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the target classes
- Used **StandardScaler** for numerical features
- Created **pipelines** for consistent preprocessing and modeling

## 🧠 Models Tested

1. K-Nearest Neighbors (KNN)
2. Logistic Regression (LR)
3. Linear Discriminant Analysis (LDA)
4. Gaussian Naive Bayes (GNB)

Each model was tuned using **GridSearchCV** with `roc_auc` as the scoring metric. The best models and their parameters were saved.

## 🏆 Best Model

- **Logistic Regression**
- Achieved the highest ROC AUC during cross-validation **0.785**

## 📈 Evaluation

- Evaluated using:
  - **ROC Curve**
  - **Precision-Recall Curve**
  - **Confusion Matrix**
  - **F1, Precision, Recall, Accuracy**
- Analyzed different **classification thresholds** (`0.3`, `0.5`, `0.7`) to understand trade-offs:
  - `Threshold = 0.3`: High recall for the positive class, useful when false negatives are costly
  - `Threshold = 0.7`: Balanced performance with higher precision, preferred for reducing false positives

## 📌 Conclusion

This project demonstrates the full machine learning workflow:
- Data cleaning and preprocessing
- Class imbalance handling
- Model selection with cross-validation and hyperparameter tuning
- In-depth evaluation with custom threshold analysis

I did not apply the final model using .predict() on the test set, the analysis using predict_proba and performance metrics (confusion matrices, ROC curve, and precision-recall curve) provides a solid evaluation of the model's behavior under different threshold strategies.

This threshold analysis not only demonstrates the model's flexibility but also shows the importance of aligning evaluation metrics with real-world decision-making.

---

## 💻 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

---

