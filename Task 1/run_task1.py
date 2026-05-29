"""
run_task1.py — Runs the full ML pipeline for Task 1
Maincraft Internship | AI & Machine Learning

This script replicates the Jupyter Notebook as a standalone Python script.
Run this if you don't have Jupyter installed.
"""

import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
pd.set_option('display.float_format', lambda x: '%.4f' % x)

print("=" * 60)
print("  Task 1: Linear Regression — House Price Predictor")
print("  Maincraft Internship | AI & Machine Learning")
print("=" * 60)

# ─── 1. Load Dataset ─────────────────────────────────────────────────────────
print("\n[1/7] Loading California Housing Dataset...")
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target
print(f"      Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"      Features: {list(california.feature_names)}")
print(f"      Target: MedHouseVal (Median House Value in $100,000s)")

# ─── 2. EDA ──────────────────────────────────────────────────────────────────
print("\n[2/7] Exploratory Data Analysis...")
print(f"\n  Statistical Summary:")
print(df.describe().to_string())

missing = df.isnull().sum()
if missing.sum() == 0:
    print("\n  ✅ No missing values found. Dataset is clean.")
else:
    print(f"\n  ⚠️  Missing values:\n{missing[missing > 0]}")

# Plot 1: Target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['MedHouseVal'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_title('Distribution of Median House Value', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Median House Value ($100,000s)')
axes[0].set_ylabel('Count')
axes[0].axvline(df['MedHouseVal'].mean(), color='red', linestyle='--', label=f'Mean: {df["MedHouseVal"].mean():.2f}')
axes[0].axvline(df['MedHouseVal'].median(), color='orange', linestyle='--', label=f'Median: {df["MedHouseVal"].median():.2f}')
axes[0].legend()
axes[1].boxplot(df['MedHouseVal'], vert=True, patch_artist=True, boxprops=dict(facecolor='steelblue', alpha=0.7))
axes[1].set_title('Box Plot of Median House Value', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Median House Value ($100,000s)')
axes[1].set_xticks([])
plt.tight_layout()
plt.savefig('eda_target_distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print("      Saved: eda_target_distribution.png")

# Plot 2: Feature distributions
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
feature_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#937860', '#DA8BC3', '#8C8C8C', '#CCB974']
for i, feature in enumerate(california.feature_names):
    axes[i].hist(df[feature], bins=40, color=feature_colors[i], edgecolor='white', alpha=0.8)
    axes[i].set_title(f'{feature}', fontsize=11, fontweight='bold')
plt.suptitle('Feature Distributions — California Housing Dataset', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda_feature_distributions.png', dpi=120, bbox_inches='tight')
plt.close()
print("      Saved: eda_feature_distributions.png")

# Plot 3: Correlation heatmap
fig, ax = plt.subplots(figsize=(11, 8))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0, linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png', dpi=120, bbox_inches='tight')
plt.close()
print("      Saved: eda_correlation_heatmap.png")

# Correlation summary
target_corr = corr_matrix['MedHouseVal'].drop('MedHouseVal').sort_values(ascending=False)
print("\n  Top Features Correlated with MedHouseVal:")
for feat, val in target_corr.items():
    print(f"    {feat:12s}: {val:+.4f}")

# Plot 4: Scatter plots
top_features = target_corr.abs().nlargest(4).index.tolist()
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
for i, feat in enumerate(top_features):
    sample = df.sample(2000, random_state=42)
    axes[i].scatter(sample[feat], sample['MedHouseVal'], alpha=0.3, s=8, color=colors[i])
    axes[i].set_xlabel(feat, fontsize=11)
    axes[i].set_ylabel('MedHouseVal' if i == 0 else '')
    axes[i].set_title(f'{feat} vs MedHouseVal\n(r = {df[feat].corr(df["MedHouseVal"]):.3f})', fontsize=11, fontweight='bold')
plt.suptitle('Top Features vs Target Variable', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_scatter_plots.png', dpi=120, bbox_inches='tight')
plt.close()
print("      Saved: eda_scatter_plots.png")

# ─── 3. Preprocessing ────────────────────────────────────────────────────────
print("\n[3/7] Preprocessing...")
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"      Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"      StandardScaler applied (mean≈0, std≈1)")

# ─── 4. Model Training ───────────────────────────────────────────────────────
print("\n[4/7] Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print(f"      Intercept (β₀): {model.intercept_:.4f}")
print(f"      Coefficients:")
for feat, coef in zip(california.feature_names, model.coef_):
    bar = '█' * int(abs(coef) * 5)
    print(f"        {feat:12s}: {coef:+.4f}  {bar}")

# Plot 5: Coefficients
coef_df = pd.DataFrame({'Feature': california.feature_names, 'Coefficient': model.coef_}).sort_values('Coefficient', ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
colors_bar = ['#55A868' if c > 0 else '#C44E52' for c in coef_df['Coefficient']]
bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_bar, alpha=0.85, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Linear Regression Coefficients', fontsize=13, fontweight='bold')
ax.set_xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('model_coefficients.png', dpi=120, bbox_inches='tight')
plt.close()
print("      Saved: model_coefficients.png")

# ─── 5. Evaluation ───────────────────────────────────────────────────────────
print("\n[5/7] Evaluating model...")
y_train_pred = model.predict(X_train_scaled)
y_test_pred  = model.predict(X_test_scaled)

train_mae  = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2   = r2_score(y_train, y_train_pred)

test_mae   = mean_absolute_error(y_test, y_test_pred)
test_rmse  = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2    = r2_score(y_test, y_test_pred)

print(f"\n  {'Metric':<30} {'Train':>10} {'Test':>10}")
print(f"  {'─'*52}")
print(f"  {'MAE (Mean Absolute Error)':<30} {train_mae:>10.4f} {test_mae:>10.4f}")
print(f"  {'RMSE (Root Mean Sq Error)':<30} {train_rmse:>10.4f} {test_rmse:>10.4f}")
print(f"  {'R² Score':<30} {train_r2:>10.4f} {test_r2:>10.4f}")
print(f"\n  Test MAE in dollars : ${test_mae * 100000:,.0f}")
print(f"  Test R² percent     : {test_r2 * 100:.1f}% variance explained")

# Plot 6: Actual vs Predicted + Residuals
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].scatter(y_test, y_test_pred, alpha=0.3, s=10, color='#4C72B0')
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Values ($100,000s)', fontsize=11)
axes[0].set_ylabel('Predicted Values ($100,000s)', fontsize=11)
axes[0].set_title(f'Actual vs Predicted\n(R² = {test_r2:.4f})', fontsize=13, fontweight='bold')
axes[0].legend()
residuals = y_test - y_test_pred
axes[1].hist(residuals, bins=50, color='#55A868', edgecolor='white', alpha=0.8)
axes[1].axvline(0, color='red', linestyle='--', lw=2, label='Zero Error')
axes[1].axvline(residuals.mean(), color='orange', linestyle='--', lw=2, label=f'Mean = {residuals.mean():.3f}')
axes[1].set_xlabel('Residual (Actual − Predicted)', fontsize=11)
axes[1].set_ylabel('Count', fontsize=11)
axes[1].set_title('Residuals Distribution', fontsize=13, fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig('eval_actual_vs_predicted.png', dpi=120, bbox_inches='tight')
plt.close()
print("      Saved: eval_actual_vs_predicted.png")

# Plot 7: Residuals vs Predicted + Metrics bar
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].scatter(y_test_pred, residuals, alpha=0.3, s=10, color='#8172B2')
axes[0].axhline(0, color='red', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted Values ($100,000s)', fontsize=11)
axes[0].set_ylabel('Residuals', fontsize=11)
axes[0].set_title('Residuals vs Predicted Values', fontsize=13, fontweight='bold')
metrics_train = [train_mae, train_rmse, train_r2]
metrics_test  = [test_mae, test_rmse, test_r2]
labels = ['MAE', 'RMSE', 'R²']
x = np.arange(len(labels))
width = 0.35
bars1 = axes[1].bar(x - width/2, metrics_train, width, label='Training', color='#4C72B0', alpha=0.8)
bars2 = axes[1].bar(x + width/2, metrics_test, width, label='Test', color='#C44E52', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_title('Model Metrics: Train vs Test', fontsize=13, fontweight='bold')
axes[1].legend()
for bar in list(bars1) + list(bars2):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('eval_residuals_and_metrics.png', dpi=120, bbox_inches='tight')
plt.close()
print("      Saved: eval_residuals_and_metrics.png")

# ─── 6. Save Model ───────────────────────────────────────────────────────────
print("\n[6/7] Saving model...")
model_data = {
    'model': model,
    'scaler': scaler,
    'feature_names': list(california.feature_names),
    'metrics': {
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }
}
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
size_kb = os.path.getsize('house_price_model.pkl') / 1024
print(f"      Saved: house_price_model.pkl ({size_kb:.1f} KB)")

# ─── 7. Summary ──────────────────────────────────────────────────────────────
print("\n[7/7] Final Summary")
print("=" * 60)
print(f"  Model          : Linear Regression (OLS)")
print(f"  Dataset        : California Housing (20,640 samples)")
print(f"  Train/Test     : 80% / 20% split")
print(f"  Preprocessing  : StandardScaler")
print()
print(f"  TEST METRICS:")
print(f"  ├─ MAE    : {test_mae:.4f}  (~${test_mae*100000:,.0f} avg error)")
print(f"  ├─ RMSE   : {test_rmse:.4f}  (~${test_rmse*100000:,.0f})")
print(f"  └─ R²     : {test_r2:.4f}  ({test_r2*100:.1f}% variance explained)")
print()
print("  SAVED FILES:")
saved = ['house_price_model.pkl',
         'eda_target_distribution.png', 'eda_feature_distributions.png',
         'eda_correlation_heatmap.png', 'eda_scatter_plots.png',
         'model_coefficients.png', 'eval_actual_vs_predicted.png',
         'eval_residuals_and_metrics.png']
for f in saved:
    exists = "✅" if os.path.exists(f) else "❌"
    print(f"    {exists} {f}")
print()
print("  IMPROVEMENT IDEAS:")
print("  1. Feature engineering (rooms_per_person, income_per_room)")
print("  2. Outlier treatment with IQR clipping")
print("  3. Try Ridge/Lasso, RandomForest, XGBoost")
print("  4. Polynomial features (degree-2)")
print("  5. Log-transform the target variable")
print("  6. 5-fold cross-validation for robust evaluation")
print("=" * 60)
print("\n✅ Task 1 completed successfully!")
