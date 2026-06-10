import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

import sklearn
print("=" * 55)
print("  AI/ML Task 2 -- Maincrafts Technology Internship")
print("=" * 55)
print(f"  pandas    : {pd.__version__}")
print(f"  numpy     : {np.__version__}")
print(f"  sklearn   : {sklearn.__version__}")
print(f"  matplotlib: {matplotlib.__version__}")
print()

print("-" * 40)
print("STEP 2: Loading California Housing Dataset")
print("-" * 40)
data = fetch_california_housing(as_frame=True)
df = pd.concat([data.data, data.target.rename("HousePrice")], axis=1)

print(f"[OK] Dataset shape: {df.shape}")
print(f"     Columns: {list(df.columns)}")
print(f"     Missing values: {df.isnull().sum().sum()}")
print()

print("-" * 40)
print("STEP 3: Separating Features (X) and Target (y)")
print("-" * 40)
X = df.drop("HousePrice", axis=1)
y = df["HousePrice"]
print(f"[OK] X shape: {X.shape}")
print(f"[OK] y shape: {y.shape}")
print()

print("-" * 40)
print("STEP 4: Feature Scaling with StandardScaler")
print("-" * 40)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("[OK] StandardScaler applied")
print("     Before scaling -- means:", X.mean().round(2).values)
print("     After  scaling -- means:", X_scaled_df.mean().round(6).values)
print()

n_features = len(X.columns)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(16, 4 * n_rows))
fig.suptitle('Feature Distributions: Before vs After StandardScaler',
             fontsize=14, fontweight='bold', y=1.02)
features = X.columns
for i, feat in enumerate(features):
    row_before = (i // n_cols) * 2
    row_after  = row_before + 1
    col = i % n_cols
    axes[row_before][col].hist(X[feat], bins=30, color='#E74C3C', alpha=0.8, edgecolor='white')
    axes[row_before][col].set_title(f'{feat}\n(Before)', fontsize=9, color='#E74C3C')
    axes[row_after][col].hist(X_scaled_df[feat], bins=30, color='#2ECC71', alpha=0.8, edgecolor='white')
    axes[row_after][col].set_title(f'{feat}\n(After)', fontsize=9, color='#2ECC71')
plt.tight_layout()
plt.savefig('feature_scaling_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("     Saved: feature_scaling_comparison.png")
print()

print("-" * 40)
print("STEP 5: Train-Test Split (80% / 20%)")
print("-" * 40)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"[OK] X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"     y_train: {y_train.shape}, y_test: {y_test.shape}")
print()

print("-" * 40)
print("STEP 6: Defining Models")
print("-" * 40)
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42)
}
for name in models:
    print(f"  * {name}")
print()

print("-" * 40)
print("STEP 7: Model Evaluation and Comparison")
print("-" * 40)
results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    rmse = root_mean_squared_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    results[name] = {"RMSE": rmse, "R2 Score": r2}
    print(f"  {name:25s}  RMSE={rmse:.4f}  R2={r2:.4f}")

results_df = pd.DataFrame(results).T.sort_values("RMSE")

print()
print("=" * 55)
print("  MODEL PERFORMANCE COMPARISON TABLE")
print("=" * 55)
print(results_df.to_string())
print("=" * 55)
print()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold')

model_names = list(results_df.index)
rmse_vals = results_df['RMSE'].values
r2_vals   = results_df['R2 Score'].values
palette = ['#2ECC71', '#3498DB', '#E74C3C']

bars1 = axes[0].bar(model_names, rmse_vals, color=palette, edgecolor='black', linewidth=0.8)
axes[0].set_title('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('RMSE')
for bar, val in zip(bars1, rmse_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[0].set_ylim(0, max(rmse_vals) * 1.2)

bars2 = axes[1].bar(model_names, r2_vals, color=palette, edgecolor='black', linewidth=0.8)
axes[1].set_title('R2 Score (Higher is Better)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('R2 Score')
for bar, val in zip(bars2, r2_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('model_comparison_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("     Saved: model_comparison_metrics.png")
print()

print("-" * 40)
print("STEP 8: Visual Performance Validation")
print("-" * 40)

colors = ['#3498DB', '#27AE60', '#E74C3C']
model_list = list(models.keys())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Actual vs Predicted House Prices -- All Models', fontsize=15, fontweight='bold')
for idx, name in enumerate(model_list):
    y_pred = predictions[name]
    r2_val   = results[name]['R2 Score']
    rmse_val = results[name]['RMSE']
    ax = axes[idx]
    ax.scatter(y_test, y_pred, alpha=0.3, color=colors[idx], s=15, label='Predictions')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    ax.set_title(f'{name}\nR2={r2_val:.4f} | RMSE={rmse_val:.4f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual House Prices', fontsize=10)
    ax.set_ylabel('Predicted House Prices', fontsize=10)
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("     Saved: actual_vs_predicted.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Residual Analysis -- Prediction Errors', fontsize=15, fontweight='bold')
for idx, name in enumerate(model_list):
    y_pred = predictions[name]
    residuals = y_test.values - y_pred
    ax = axes[idx]
    ax.scatter(y_pred, residuals, alpha=0.3, color=colors[idx], s=15)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'{name}\nResiduals vs Predicted', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Values', fontsize=10)
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=10)
plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("     Saved: residual_analysis.png")

best_model_name = results_df['RMSE'].idxmin()
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_best, alpha=0.4, color='#2E86AB', s=15, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', linewidth=2, label='Perfect Fit')
plt.xlabel('Actual House Prices', fontsize=12)
plt.ylabel('Predicted House Prices', fontsize=12)
plt.title(f'Best Model: {best_model_name}\nActual vs Predicted House Prices',
          fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('best_model_prediction.png', dpi=150, bbox_inches='tight')
plt.close()
print("     Saved: best_model_prediction.png")
print()

best_rmse = results_df.loc[best_model_name, 'RMSE']
best_r2   = results_df.loc[best_model_name, 'R2 Score']

print("=" * 55)
print("  FINAL MODEL SELECTION REPORT")
print("=" * 55)
print(f"  Best Model : {best_model_name}")
print(f"  RMSE       : {best_rmse:.4f}")
print(f"  R2 Score   : {best_r2:.4f}")
print()
print("  Rankings:")
print(results_df.to_string())
print()
print("  Justification:")
print(f"  '{best_model_name}' achieves the lowest RMSE and highest R2,")
print(f"  indicating it generalizes best to unseen test data.")
print("=" * 55)
print()
print("[OK] Task 2 completed successfully! All charts saved.")
