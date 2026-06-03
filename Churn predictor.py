"""
Customer Churn Predictor
=========================
Predicts whether a telecom customer will churn (leave)
using classification models. Includes ROC-AUC analysis
and feature importance.

Author: Divya Nimbalkar
Tech Stack: Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# 1. DATASET GENERATION
# ──────────────────────────────────────────────

def generate_churn_dataset(n: int = 5000) -> pd.DataFrame:
    """Generate synthetic telecom customer churn dataset."""
    np.random.seed(42)

    tenure           = np.random.randint(1, 72, n)
    monthly_charges  = np.random.uniform(20, 120, n)
    total_charges    = tenure * monthly_charges + np.random.normal(0, 50, n)
    num_services     = np.random.randint(1, 8, n)
    contract         = np.random.choice(['Month-to-month', 'One year', 'Two year'], n,
                                         p=[0.55, 0.25, 0.20])
    paperless        = np.random.randint(0, 2, n)
    support_calls    = np.random.randint(0, 10, n)
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n)
    payment_method   = np.random.choice(['Electronic check', 'Mailed check',
                                          'Bank transfer', 'Credit card'], n)
    senior_citizen   = np.random.choice([0, 1], n, p=[0.84, 0.16])

    # Churn probability based on realistic business logic
    contract_enc = {'Month-to-month': 0.40, 'One year': 0.10, 'Two year': 0.03}
    churn_prob = (
        0.35 * (np.array([contract_enc[c] for c in contract]))
        + 0.20 * (1 - tenure / 72)
        + 0.15 * (support_calls / 10)
        + 0.10 * (monthly_charges / 120)
        + 0.10 * senior_citizen * 0.3
        + 0.10 * np.random.uniform(0, 1, n)
    )
    churn = (churn_prob > np.random.uniform(0.25, 0.55, n)).astype(int)

    df = pd.DataFrame({
        'tenure': tenure, 'monthly_charges': monthly_charges,
        'total_charges': total_charges, 'num_services': num_services,
        'contract': contract, 'paperless_billing': paperless,
        'support_calls': support_calls, 'internet_service': internet_service,
        'payment_method': payment_method, 'senior_citizen': senior_citizen,
        'churn': churn
    })
    return df


# ──────────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    df_enc = df.copy()
    le = LabelEncoder()
    for col in ['contract', 'internet_service', 'payment_method']:
        df_enc[col] = le.fit_transform(df_enc[col])

    X = df_enc.drop('churn', axis=1)
    y = df_enc['churn']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y


# ──────────────────────────────────────────────
# 3. MODEL TRAINING
# ──────────────────────────────────────────────

def train_models(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'Logistic Regression':   LogisticRegression(random_state=42, max_iter=500),
        'Decision Tree':         DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest':         RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':     GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    print("=" * 60)
    print("    CUSTOMER CHURN PREDICTOR — MODEL RESULTS")
    print("=" * 60)
    print(f"  Churn rate in dataset: {y.mean()*100:.1f}%\n")
    print(f"  {'Model':<25} {'Accuracy':>9}  {'ROC-AUC':>8}")
    print("  " + "-" * 48)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        results[name] = {'model': model, 'y_pred': y_pred,
                         'y_test': y_test, 'y_prob': y_prob,
                         'acc': acc, 'auc': auc}
        print(f"  {name:<25} {acc*100:>8.2f}%  {auc:>8.4f}")

    best = max(results, key=lambda k: results[k]['auc'])
    print(f"\n  Best Model: {best} (AUC = {results[best]['auc']:.4f})")
    print("\n" + classification_report(y_test, results[best]['y_pred'],
                                        target_names=['Retained', 'Churned']))
    return results, best, X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# 4. VISUALIZATIONS
# ──────────────────────────────────────────────

def plot_results(df, results, best_name, X):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Customer Churn Predictor — Analysis', fontsize=14, fontweight='bold')

    # Churn distribution
    counts = df['churn'].value_counts()
    axes[0, 0].pie(counts, labels=['Retained', 'Churned'], autopct='%1.1f%%',
                   colors=['steelblue', 'coral'], startangle=90)
    axes[0, 0].set_title('Churn Distribution')

    # ROC curves
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
        axes[0, 1].plot(fpr, tpr, lw=1.5, label=f"{name} ({res['auc']:.3f})")
    axes[0, 1].plot([0, 1], [0, 1], 'k--')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curves — All Models')
    axes[0, 1].legend(fontsize=8)

    # Confusion matrix
    cm = confusion_matrix(results[best_name]['y_test'], results[best_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Retained', 'Churned'],
                yticklabels=['Retained', 'Churned'])
    axes[1, 0].set_title(f'Confusion Matrix ({best_name})')

    # Feature importance
    rf = results['Random Forest']['model']
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    imp.plot(kind='barh', ax=axes[1, 1], color='mediumseagreen')
    axes[1, 1].set_title('Random Forest — Feature Importance')

    plt.tight_layout()
    plt.savefig('churn_predictor_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  Plot saved as 'churn_predictor_results.png'")


# ──────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────

def main():
    print("\n  Generating customer dataset...")
    df = generate_churn_dataset(n=5000)
    print(f"  Dataset: {len(df)} customers | Churn Rate: {df['churn'].mean()*100:.1f}%\n")

    X, y = preprocess(df)
    results, best_name, *_ = train_models(X, y)
    plot_results(df, results, best_name, X)


if __name__ == "__main__":
    main()