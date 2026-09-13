# Credit Card Fraud Detection

A Random Forest-based binary classification model for detecting fraudulent credit card transactions, built on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) containing 284,807 transactions (492 fraudulent).

## Results

| Metric | Score |
|---|---|
| Accuracy | 99.96% |
| Precision | 98.72% |
| Recall | 78.57% |
| F1-Score | 0.875 |
| Matthews Correlation Coefficient | 0.881 |

## Project Overview

Credit card fraud detection is a classic imbalanced classification problem — only **0.17%** of transactions in the dataset are fraudulent. This project builds a Random Forest classifier to identify fraud while managing the extreme class imbalance inherent in real-world transaction data.

### Pipeline

1. **Exploratory Data Analysis** — Dataset inspection, descriptive statistics, class distribution analysis, and fraud vs. legitimate transaction comparison
2. **Correlation Analysis** — Heatmap visualization of feature correlations across all 30 PCA-transformed features
3. **Feature Engineering** — Separation of features (`V1`–`V28`, `Time`, `Amount`) and target variable (`Class`)
4. **Train-Test Split** — 80/20 split (`random_state=42`)
5. **Model Training** — Random Forest Classifier (scikit-learn)
6. **Evaluation** — Accuracy, Precision, Recall, F1-Score, MCC, and Confusion Matrix

### Key Observations

- Average fraudulent transaction amount ($122.21) is higher than average legitimate transaction amount ($88.29)
- The dataset is heavily imbalanced: 284,315 legitimate vs. 492 fraudulent transactions
- The model achieves high precision (98.72%) with moderate recall (78.57%), meaning it rarely flags legitimate transactions as fraud but misses ~21% of actual fraud cases
- MCC of 0.881 indicates strong performance even accounting for class imbalance

## Dataset

The dataset contains transactions made by European cardholders in September 2013. Features `V1` through `V28` are PCA-transformed (original features are confidential). `Time` and `Amount` are the only non-transformed features.

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Rows:** 284,807
- **Features:** 30 (+ 1 target)
- **Fraud rate:** 0.17%

> **Note:** The `creditcard.csv` file is not included in this repo due to size. Download it from Kaggle and place it in the root directory.

## Tech Stack

- Python 3.12
- pandas, NumPy
- scikit-learn (RandomForestClassifier, train_test_split, classification metrics)
- Matplotlib, Seaborn

## Usage

```bash
# Clone the repo
git clone https://github.com/Prathivann/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Download the dataset from Kaggle and place creditcard.csv in the root directory

# Run the notebook
jupyter notebook Credit_Card_Fraud_Detection_-_non_stratify.ipynb
```

## Repository Structure

```
credit-card-fraud-detection/
├── Credit_Card_Fraud_Detection_-_non_stratify.ipynb   # Main notebook
├── README.md
└── .gitignore
```

## Future Improvements

- Address class imbalance with SMOTE or stratified sampling
- Compare against other models (XGBoost, Logistic Regression, Isolation Forest)
- Add threshold tuning to optimize the precision-recall tradeoff
- Implement cross-validation for more robust evaluation

## Author

**Prithivan R**
[LinkedIn](https://linkedin.com/in/prithivan-r-817948219/)
