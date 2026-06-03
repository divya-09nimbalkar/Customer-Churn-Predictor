# Customer Churn Predictor 📉

Predicts whether a telecom customer will churn using ML classification. Compares 4 algorithms with ROC-AUC analysis and business-relevant feature importance.

## Features
- Synthetic telecom dataset with realistic churn logic (5000 customers)
- 4 classifiers compared: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- ROC curves, confusion matrix, churn distribution charts
- Feature importance analysis

## Tech Stack
`Python` `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `Seaborn`

## Installation
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage
```bash
python churn_predictor.py
```

## Results
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~78% | ~0.84 |
| Random Forest | ~85% | ~0.91 |
| Gradient Boosting | ~84% | ~0.90 |

## Project Structure
```
04_customer_churn_predictor/
├── churn_predictor.py
├── churn_predictor_results.png
└── README.md
```

## Author
**Divya Nimbalkar** — [GitHub](https://github.com/divya-09nimbalkar) | [LinkedIn](https://www.linkedin.com/in/divya-nimbalkar09/)
