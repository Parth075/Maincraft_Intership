"""
run_task3.py — Runs the full ML pipeline for Task 3
Maincraft Internship | AI & Machine Learning

Task 3: Model Validation, Overfitting Control & Hyperparameter Tuning
This script replicates the Jupyter Notebook as a standalone Python script.
Run this if you don't have Jupyter installed.
"""

import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#F8F9FA',
    'axes.facecolor':   '#FFFFFF',
    'axes.grid':        True,
    'grid.alpha':       0.4,
    'grid.linestyle':   '--',
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

COLORS = {
    'lr':   '#4C72B0',
    'ridge':'#55A868',
    'tree': '#C44E52',
    'tuned':'#8172B2',
    'cv':   '#DD8452',
    'bg':   '#F8F9FA',
}

pd.set_option('display.float_format', lambda x: '%.4f' % x)

print("=" * 65)
print("  Task 3: Model Validation, Overfitting Control & HP Tuning")
print("  Maincraft Internship | AI & Machine Learning")
print("=" * 65)

# ─── Step 1: Load Dataset ─────────────────────────────────────────────────────
print("\n[1/10] Loading California Housing Dataset...")
data   = fetch_california_housing(as_frame=True)
df     = pd.concat([data.data, data.target.rename("HousePrice")], axis=1)
X      = df.drop("HousePrice", axis=1)
y      = df["HousePrice"]
print(f"       Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"       Features: {list(X.columns)}")

# ─── Step 2: Feature Scaling ──────────────────────────────────────────────────
print("\n[2/10] Feature Scaling with StandardScaler...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"       StandardScaler applied — mean≈0, std≈1")

# ─── Step 3: Train-Test Split ─────────────────────────────────────────────────
print("\n[3/10] Train-Test Split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"       Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─── Step 4: Detect Overfitting (Untuned Decision Tree) ───────────────────────
print("\n[4/10] Detecting Overfitting — Untuned Decision Tree...")
tree_raw = DecisionTreeRegressor(random_state=42)
tree_raw.fit(X_train, y_train)

train_pred_raw = tree_raw.predict(X_train)
test_pred_raw  = tree_raw.predict(X_test)

train_rmse_raw = np.sqrt(mean_squared_error(y_train, train_pred_raw))
test_rmse_raw  = np.sqrt(mean_squared_error(y_test,  test_pred_raw))
train_r2_raw   = r2_score(y_train, train_pred_raw)
test_r2_raw    = r2_score(y_test,  test_pred_raw)

print(f"       Train RMSE: {train_rmse_raw:.4f}   Train R²: {train_r2_raw:.4f}")
print(f"       Test  RMSE: {test_rmse_raw:.4f}   Test  R²: {test_r2_raw:.4f}")
overfit_gap = test_rmse_raw - train_rmse_raw
print(f"       Overfitting gap (Test-Train RMSE): {overfit_gap:.4f}  ← LARGE = Overfitting!")

# ─── Step 5: Cross-Validation on all Baseline Models ─────────────────────────
print("\n[5/10] Cross-Validation (5-Fold) on Baseline Models...")

baseline_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Decision Tree (raw)": DecisionTreeRegressor(random_state=42),
}

cv_results = {}
for name, m in baseline_models.items():
    scores = cross_val_score(
        m, X_scaled, y,
        cv=5, scoring="neg_root_mean_squared_error"
    )
    cv_rmse = -scores.mean()
    cv_std  = scores.std()
    cv_results[name] = {"CV RMSE": cv_rmse, "CV Std": cv_std}
    print(f"       {name:30s}: CV RMSE = {cv_rmse:.4f} ± {cv_std:.4f}")

# ─── Step 6: GridSearchCV — Decision Tree ─────────────────────────────────────
print("\n[6/10] GridSearchCV — Tuning Decision Tree...")
param_grid_tree = {
    "max_depth":         [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
}
grid_tree = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid_tree,
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1,
    verbose=0,
)
grid_tree.fit(X_train, y_train)
best_tree_params = grid_tree.best_params_
best_tree        = grid_tree.best_estimator_
print(f"       Best params: {best_tree_params}")
print(f"       Best CV RMSE: {-grid_tree.best_score_:.4f}")

# ─── Step 7: GridSearchCV — Ridge Regression ──────────────────────────────────
print("\n[7/10] GridSearchCV — Tuning Ridge Regression...")
param_grid_ridge = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
grid_ridge = GridSearchCV(
    Ridge(),
    param_grid_ridge,
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1,
)
grid_ridge.fit(X_train, y_train)
best_ridge_params = grid_ridge.best_params_
best_ridge        = grid_ridge.best_estimator_
print(f"       Best params: {best_ridge_params}")
print(f"       Best CV RMSE: {-grid_ridge.best_score_:.4f}")

# ─── Step 8: Evaluate All Models on Test Set ──────────────────────────────────
print("\n[8/10] Evaluating All Models on Test Set...")

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr    = lr.predict(X_test)
y_pred_ridge = best_ridge.predict(X_test)
y_pred_tree  = best_tree.predict(X_test)

def evaluate(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"       {label:35s}: RMSE={rmse:.4f}  R²={r2:.4f}")
    return rmse, r2

lr_rmse,    lr_r2    = evaluate(y_test, y_pred_lr,    "Linear Regression")
ridge_rmse, ridge_r2 = evaluate(y_test, y_pred_ridge, f"Ridge (alpha={best_ridge_params['alpha']})")
tree_rmse,  tree_r2  = evaluate(y_test, y_pred_tree,  f"Tuned Decision Tree ({best_tree_params})")

# Also get Train RMSE for tuned tree (overfitting check)
train_pred_tuned = best_tree.predict(X_train)
train_rmse_tuned = np.sqrt(mean_squared_error(y_train, train_pred_tuned))
train_r2_tuned   = r2_score(y_train, train_pred_tuned)

# CV for tuned models
cv_tree_tuned = -cross_val_score(
    best_tree, X_scaled, y, cv=5, scoring="neg_root_mean_squared_error"
).mean()
cv_ridge_tuned = -cross_val_score(
    best_ridge, X_scaled, y, cv=5, scoring="neg_root_mean_squared_error"
).mean()
cv_lr = -cross_val_score(
    lr, X_scaled, y, cv=5, scoring="neg_root_mean_squared_error"
).mean()

print(f"\n       Overfitting Check (Tuned Tree):")
print(f"         Train RMSE: {train_rmse_tuned:.4f}  |  Test RMSE: {tree_rmse:.4f}")
print(f"         Gap reduced from {overfit_gap:.4f} → {tree_rmse - train_rmse_tuned:.4f}")

# ─── Step 9: Build Final Results Table ────────────────────────────────────────
print("\n[9/10] Building Model Comparison Table...")

results_df = pd.DataFrame({
    "Model":     ["Linear Regression", f"Ridge (α={best_ridge_params['alpha']})", "Decision Tree (Raw)", f"Tuned Decision Tree"],
    "Test RMSE": [lr_rmse, ridge_rmse, test_rmse_raw, tree_rmse],
    "Test R²":   [lr_r2,   ridge_r2,   test_r2_raw,   tree_r2],
    "CV RMSE":   [cv_lr,   cv_ridge_tuned, cv_results["Decision Tree (raw)"]["CV RMSE"], cv_tree_tuned],
    "Train RMSE":[
        np.sqrt(mean_squared_error(y_train, lr.predict(X_train))),
        np.sqrt(mean_squared_error(y_train, best_ridge.predict(X_train))),
        train_rmse_raw,
        train_rmse_tuned,
    ]
})
print(results_df.to_string(index=False))

# Determine best model
best_idx   = results_df["Test RMSE"].idxmin()
best_model_name = results_df.loc[best_idx, "Model"]
best_rmse  = results_df.loc[best_idx, "Test RMSE"]
best_r2    = results_df.loc[best_idx, "Test R²"]
print(f"\n       🏆 Best Model: {best_model_name} (Test RMSE={best_rmse:.4f}, R²={best_r2:.4f})")

# ─── Step 10: Plots ────────────────────────────────────────────────────────────
print("\n[10/10] Generating Plots...")

# ── Plot 1: Overfitting Detection Bar Chart ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Step 1 — Overfitting Detection: Untuned Decision Tree", fontsize=14, fontweight='bold', y=1.02)

categories = ["Train", "Test"]
rmse_vals  = [train_rmse_raw, test_rmse_raw]
r2_vals    = [train_r2_raw,   test_r2_raw]
bar_colors = [COLORS['tree'], '#E88080']

b1 = axes[0].bar(categories, rmse_vals, color=bar_colors, edgecolor='black', linewidth=0.8, width=0.5)
axes[0].set_title("RMSE: Train vs Test (Raw Tree)", fontsize=12, fontweight='bold')
axes[0].set_ylabel("RMSE")
for bar, val in zip(b1, rmse_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].annotate(f'Gap: {overfit_gap:.4f}\n(Overfitting!)',
                 xy=(0.5, max(rmse_vals)*0.6), xycoords='data',
                 ha='center', fontsize=11, color='red',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#FFE0E0', ec='red'))
axes[0].set_ylim(0, max(rmse_vals) * 1.4)

b2 = axes[1].bar(categories, r2_vals, color=bar_colors, edgecolor='black', linewidth=0.8, width=0.5)
axes[1].set_title("R² Score: Train vs Test (Raw Tree)", fontsize=12, fontweight='bold')
axes[1].set_ylabel("R² Score")
for bar, val in zip(b2, r2_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[1].set_ylim(0, 1.25)
axes[1].axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect R²=1.0')
axes[1].legend()

plt.tight_layout()
plt.savefig('plot1_overfitting_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print("       Saved: plot1_overfitting_detection.png")

# ── Plot 2: Cross-Validation Scores ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
cv_models = list(cv_results.keys())
cv_means  = [cv_results[m]["CV RMSE"] for m in cv_models]
cv_stds   = [cv_results[m]["CV Std"]  for m in cv_models]
bar_c     = [COLORS['lr'], COLORS['ridge'], COLORS['tree']]

bars = ax.bar(cv_models, cv_means, color=bar_c, edgecolor='black', linewidth=0.8,
              yerr=cv_stds, capsize=6, error_kw={'linewidth': 2})
ax.set_title("5-Fold Cross-Validation RMSE — Baseline Models", fontsize=13, fontweight='bold')
ax.set_ylabel("CV RMSE (lower is better)")
for bar, val, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
            f'{val:.4f}\n±{std:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, max(cv_means) * 1.5)
plt.tight_layout()
plt.savefig('plot2_cross_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("       Saved: plot2_cross_validation.png")

# ── Plot 3: GridSearchCV Heatmap (Decision Tree) ──────────────────────────────
cv_results_tree = grid_tree.cv_results_
params_df = pd.DataFrame(cv_results_tree['params'])
params_df['mean_test_score'] = -cv_results_tree['mean_test_score']

# Use only max_depth vs min_samples_split with best min_samples_leaf
best_leaf = best_tree_params['min_samples_leaf']
sub = params_df[params_df['min_samples_leaf'] == best_leaf]
pivot = sub.pivot_table(index='max_depth', columns='min_samples_split', values='mean_test_score')

fig, ax = plt.subplots(figsize=(9, 5))
import matplotlib.colors as mcolors
im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(pivot.columns)))
ax.set_yticks(range(len(pivot.index)))
ax.set_xticklabels([f'min_split={c}' for c in pivot.columns])
ax.set_yticklabels([f'max_depth={r}' for r in pivot.index])
ax.set_title(f"GridSearchCV Heatmap — Decision Tree\n(min_samples_leaf={best_leaf})", fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='CV RMSE')
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9,
                color='white' if val > pivot.values.mean() else 'black', fontweight='bold')
plt.tight_layout()
plt.savefig('plot3_gridsearch_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("       Saved: plot3_gridsearch_heatmap.png")

# ── Plot 4: Before vs After Tuning Comparison ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Overfitting Reduction — Before vs After Tuning", fontsize=14, fontweight='bold')

labels      = ['Train RMSE', 'Test RMSE']
before_vals = [train_rmse_raw, test_rmse_raw]
after_vals  = [train_rmse_tuned, tree_rmse]
x = np.arange(len(labels))
w = 0.35

bars1 = axes[0].bar(x - w/2, before_vals, w, label='Raw Tree', color=COLORS['tree'], alpha=0.85, edgecolor='black')
bars2 = axes[0].bar(x + w/2, after_vals,  w, label='Tuned Tree', color=COLORS['tuned'], alpha=0.85, edgecolor='black')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].set_title("Decision Tree: RMSE Before vs After Tuning", fontsize=11, fontweight='bold')
axes[0].set_ylabel("RMSE")
axes[0].legend()
for bar in list(bars1) + list(bars2):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].set_ylim(0, max(before_vals + after_vals) * 1.35)

# Gap comparison
gaps = [train_rmse_raw - 0, test_rmse_raw - train_rmse_raw,
        train_rmse_tuned - 0, tree_rmse - train_rmse_tuned]
gap_labels = ['Raw\nTrain', 'Raw\nGap', 'Tuned\nTrain', 'Tuned\nGap']
gap_colors = [COLORS['tree'], '#E74C3C', COLORS['tuned'], '#27AE60']
bars3 = axes[1].bar(gap_labels, [train_rmse_raw, overfit_gap, train_rmse_tuned, tree_rmse - train_rmse_tuned],
                    color=gap_colors, edgecolor='black', linewidth=0.8)
axes[1].set_title("Overfitting Gap: Raw vs Tuned", fontsize=11, fontweight='bold')
axes[1].set_ylabel("RMSE Value")
for bar in bars3:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[1].set_ylim(0, max(train_rmse_raw, overfit_gap, train_rmse_tuned, tree_rmse - train_rmse_tuned) * 1.5)
plt.tight_layout()
plt.savefig('plot4_before_after_tuning.png', dpi=150, bbox_inches='tight')
plt.close()
print("       Saved: plot4_before_after_tuning.png")

# ── Plot 5: Final Model Comparison ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Final Model Comparison — All Models", fontsize=14, fontweight='bold')

model_names_plot = results_df["Model"].values
rmse_plot = results_df["Test RMSE"].values
r2_plot   = results_df["Test R²"].values
cv_plot   = results_df["CV RMSE"].values
palette   = [COLORS['lr'], COLORS['ridge'], COLORS['tree'], COLORS['tuned']]

for ax_i, (vals, title, ylabel, good) in enumerate([
    (rmse_plot, "Test RMSE (↓ Lower is Better)", "RMSE", False),
    (r2_plot,   "Test R² Score (↑ Higher is Better)", "R²", True),
    (cv_plot,   "CV RMSE (↓ Lower is Better)", "CV RMSE", False),
]):
    bars = axes[ax_i].bar(model_names_plot, vals, color=palette, edgecolor='black', linewidth=0.8)
    axes[ax_i].set_title(title, fontsize=11, fontweight='bold')
    axes[ax_i].set_ylabel(ylabel)
    axes[ax_i].set_xticklabels(model_names_plot, rotation=15, ha='right', fontsize=8)
    best_val = min(vals) if not good else max(vals)
    for bar, val in zip(bars, vals):
        axes[ax_i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        if abs(val - best_val) < 1e-9:
            bar.set_edgecolor('gold')
            bar.set_linewidth(3)

plt.tight_layout()
plt.savefig('plot5_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("       Saved: plot5_model_comparison.png")

# ── Plot 6: Actual vs Predicted — Best 3 models ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Actual vs Predicted — All Models", fontsize=14, fontweight='bold')
preds   = [y_pred_lr, y_pred_ridge, y_pred_tree]
labels3 = ["Linear Regression", f"Ridge (α={best_ridge_params['alpha']})", "Tuned Decision Tree"]
colors3 = [COLORS['lr'], COLORS['ridge'], COLORS['tuned']]
for i, (pred, label, color) in enumerate(zip(preds, labels3, colors3)):
    rmse_i = np.sqrt(mean_squared_error(y_test, pred))
    r2_i   = r2_score(y_test, pred)
    axes[i].scatter(y_test, pred, alpha=0.3, s=8, color=color)
    lo = min(y_test.min(), pred.min())
    hi = max(y_test.max(), pred.max())
    axes[i].plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect Fit')
    axes[i].set_title(f"{label}\nRMSE={rmse_i:.4f}  R²={r2_i:.4f}", fontsize=10, fontweight='bold')
    axes[i].set_xlabel("Actual House Price ($100K)")
    axes[i].set_ylabel("Predicted House Price ($100K)" if i == 0 else "")
    axes[i].legend(fontsize=8)
plt.tight_layout()
plt.savefig('plot6_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("       Saved: plot6_actual_vs_predicted.png")

# ── Plot 7: Residuals Analysis ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Residual Analysis — Prediction Errors", fontsize=14, fontweight='bold')
for i, (pred, label, color) in enumerate(zip(preds, labels3, colors3)):
    residuals = y_test.values - pred
    axes[i].scatter(pred, residuals, alpha=0.3, s=8, color=color)
    axes[i].axhline(0, color='red', linestyle='--', lw=2)
    axes[i].set_title(f"{label}\nResiduals vs Predicted", fontsize=10, fontweight='bold')
    axes[i].set_xlabel("Predicted Values")
    axes[i].set_ylabel("Residuals" if i == 0 else "")
plt.tight_layout()
plt.savefig('plot7_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("       Saved: plot7_residuals.png")

# ── Save Model ─────────────────────────────────────────────────────────────────
joblib.dump({
    'best_tree':   best_tree,
    'best_ridge':  best_ridge,
    'scaler':      scaler,
    'best_tree_params':  best_tree_params,
    'best_ridge_params': best_ridge_params,
    'results_df':  results_df.to_dict(),
}, 'task3_best_model.pkl')
print("\n       Saved: task3_best_model.pkl")

# ── Save results JSON for notebook/PDF use ────────────────────────────────────
metrics = {
    "train_rmse_raw":   train_rmse_raw,
    "test_rmse_raw":    test_rmse_raw,
    "train_r2_raw":     train_r2_raw,
    "test_r2_raw":      test_r2_raw,
    "overfit_gap":      overfit_gap,
    "lr_rmse":          lr_rmse,
    "lr_r2":            lr_r2,
    "cv_lr":            cv_lr,
    "ridge_rmse":       ridge_rmse,
    "ridge_r2":         ridge_r2,
    "cv_ridge_tuned":   cv_ridge_tuned,
    "best_ridge_alpha": best_ridge_params['alpha'],
    "tree_rmse":        tree_rmse,
    "tree_r2":          tree_r2,
    "cv_tree_tuned":    cv_tree_tuned,
    "train_rmse_tuned": train_rmse_tuned,
    "train_r2_tuned":   train_r2_tuned,
    "best_tree_params": best_tree_params,
    "best_model_name":  best_model_name,
    "best_rmse":        best_rmse,
    "best_r2":          best_r2,
    "overfit_gap_tuned": tree_rmse - train_rmse_tuned,
}
with open('task3_results.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("       Saved: task3_results.json")

# ─── Final Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL SUMMARY — Task 3")
print("=" * 65)
print(f"\n  {'Model':<35} {'Test RMSE':>10} {'Test R²':>9} {'CV RMSE':>10}")
print(f"  {'─'*68}")
for _, row in results_df.iterrows():
    marker = " ← BEST" if row['Model'] == best_model_name else ""
    print(f"  {row['Model']:<35} {row['Test RMSE']:>10.4f} {row['Test R²']:>9.4f} {row['CV RMSE']:>10.4f}{marker}")

print(f"\n  Overfitting (Raw Tree): gap = {overfit_gap:.4f}")
print(f"  After Tuning (Tuned Tree): gap = {tree_rmse - train_rmse_tuned:.4f}")
gap_reduction = (1 - (tree_rmse - train_rmse_tuned) / overfit_gap) * 100
print(f"  Gap reduced by {gap_reduction:.1f}%  ✅")
print(f"\n  🏆 Best Model : {best_model_name}")
print(f"     Test RMSE  : {best_rmse:.4f}  (~${best_rmse*100000:,.0f})")
print(f"     Test R²    : {best_r2:.4f}  ({best_r2*100:.1f}% variance explained)")
print(f"     CV RMSE    : {results_df[results_df['Model']==best_model_name]['CV RMSE'].values[0]:.4f}")

print("\n  SAVED FILES:")
files = [
    'plot1_overfitting_detection.png', 'plot2_cross_validation.png',
    'plot3_gridsearch_heatmap.png',    'plot4_before_after_tuning.png',
    'plot5_model_comparison.png',      'plot6_actual_vs_predicted.png',
    'plot7_residuals.png',             'task3_best_model.pkl',
    'task3_results.json',
]
for f in files:
    status = "✅" if os.path.exists(f) else "❌"
    print(f"    {status} {f}")

print("\n" + "=" * 65)
print("✅ Task 3 pipeline completed successfully!")
print("=" * 65)
