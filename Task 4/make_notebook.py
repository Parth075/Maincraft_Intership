"""
make_notebook.py — Generates the Task 4 Jupyter Notebook
"""
import json, uuid, os

nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.9.0'}
    },
    'cells': []
}

def md(src):
    return {'cell_type': 'markdown', 'id': str(uuid.uuid4())[:8],
            'metadata': {}, 'source': src}

def code(src):
    return {'cell_type': 'code', 'id': str(uuid.uuid4())[:8],
            'metadata': {}, 'execution_count': None, 'outputs': [], 'source': src}

# ── Cell contents ─────────────────────────────────────────────────────────────

cell_md_title = [
    "# AI & Machine Learning — Task 4\n",
    "## Classification Models, Evaluation Metrics & Handling Imbalanced Data\n",
    "**Maincrafts Technology** | www.maincrafts.com | hr@maincrafts.com\n\n",
    "---\n",
    "### Objective\n",
    "In Task-3, we mastered regression models. In Task-4 we **move to classification** — one of the most critical ML problem types in industry.\n\n",
    "We will:\n",
    "- Train and evaluate classification models\n",
    "- Use proper evaluation metrics beyond accuracy\n",
    "- Handle class imbalance\n",
    "- Compare multiple classifiers scientifically\n",
]

cell_code_imports = (
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import joblib\n\n"
    "from sklearn.datasets import load_breast_cancer\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.tree import DecisionTreeClassifier\n"
    "from sklearn.metrics import (\n"
    "    confusion_matrix, classification_report,\n"
    "    roc_curve, roc_auc_score,\n"
    "    precision_score, recall_score, f1_score, accuracy_score,\n"
    "    precision_recall_curve\n"
    ")\n\n"
    "plt.rcParams.update({\n"
    "    'figure.facecolor': '#F8F9FA',\n"
    "    'axes.facecolor':   '#FFFFFF',\n"
    "    'axes.grid':        True,\n"
    "    'grid.alpha':       0.35,\n"
    "    'grid.linestyle':   '--',\n"
    "    'font.family':      'DejaVu Sans',\n"
    "    'axes.spines.top':  False,\n"
    "    'axes.spines.right':False,\n"
    "})\n\n"
    "BLUE   = '#2563EB'\n"
    "GREEN  = '#16A34A'\n"
    "RED    = '#DC2626'\n"
    "PURPLE = '#7C3AED'\n"
    "ORANGE = '#D97706'\n\n"
    "print('All libraries imported successfully!')"
)

cell_md_step2 = [
    "## Step 2 — Load Dataset\n",
    "Using the **Breast Cancer dataset** from scikit-learn (built-in, no download needed).\n\n",
    "| Class | Label | Meaning |\n",
    "|-------|-------|---------|\n",
    "| 0 | Malignant | Cancerous |\n",
    "| 1 | Benign    | Non-cancerous |\n",
]

cell_code_load = (
    "data = load_breast_cancer()\n"
    "X = pd.DataFrame(data.data, columns=data.feature_names)\n"
    "y = pd.Series(data.target, name='target')\n\n"
    "n_mal = int((y==0).sum())\n"
    "n_ben = int((y==1).sum())\n"
    "ratio = n_ben / n_mal\n\n"
    "print(f'Dataset shape: {X.shape}')\n"
    "print(f'Classes: 0=Malignant ({n_mal}), 1=Benign ({n_ben})')\n"
    "print(f'Class ratio: {ratio:.2f}:1  (mild imbalance)')\n"
    "X.head(3)"
)

cell_code_split = (
    "X_train, X_test, y_train, y_test = train_test_split(\n"
    "    X, y, test_size=0.2, random_state=42, stratify=y\n"
    ")\n\n"
    "n_test_mal = int((y_test==0).sum())\n"
    "n_test_ben = int((y_test==1).sum())\n"
    "print(f'Train set: {X_train.shape[0]} samples')\n"
    "print(f'Test  set: {X_test.shape[0]} samples')\n"
    "print(f'Test class distribution: Malignant={n_test_mal}, Benign={n_test_ben}')"
)

cell_code_scale = (
    "scaler = StandardScaler()\n"
    "X_train_sc = scaler.fit_transform(X_train)\n"
    "X_test_sc  = scaler.transform(X_test)\n\n"
    "mean_val = X_train_sc[:,0].mean()\n"
    "std_val  = X_train_sc[:,0].std()\n"
    "print('StandardScaler applied.')\n"
    "print(f'Mean of first feature (train): {mean_val:.6f}  (should be ~0)')\n"
    "print(f'Std  of first feature (train): {std_val:.6f}  (should be ~1)')"
)

cell_code_lr = (
    "lr = LogisticRegression(max_iter=1000, random_state=42)\n"
    "lr.fit(X_train_sc, y_train)\n"
    "y_pred_lr  = lr.predict(X_test_sc)\n"
    "y_prob_lr  = lr.predict_proba(X_test_sc)[:, 1]\n\n"
    "acc_lr   = accuracy_score(y_test, y_pred_lr)\n"
    "prec_lr  = precision_score(y_test, y_pred_lr)\n"
    "rec_lr   = recall_score(y_test, y_pred_lr)\n"
    "f1_lr    = f1_score(y_test, y_pred_lr)\n"
    "auc_lr   = roc_auc_score(y_test, y_prob_lr)\n\n"
    "print(f'Accuracy : {acc_lr:.4f}')\n"
    "print(f'Precision: {prec_lr:.4f}')\n"
    "print(f'Recall   : {rec_lr:.4f}')\n"
    "print(f'F1-Score : {f1_lr:.4f}')\n"
    "print(f'AUC      : {auc_lr:.4f}')"
)

cell_md_cm = [
    "## Step 6 — Confusion Matrix & Classification Report\n\n",
    "### Interpretation Guide\n",
    "| Term | Description |\n",
    "|------|-------------|\n",
    "| **True Positive (TP)** | Correctly predicted Benign |\n",
    "| **True Negative (TN)** | Correctly predicted Malignant |\n",
    "| **False Positive (FP)** | Malignant predicted as Benign (missed cancer!) |\n",
    "| **False Negative (FN)** | Benign predicted as Malignant (false alarm) |\n",
]

cell_code_cm = (
    "cm_lr = confusion_matrix(y_test, y_pred_lr)\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n"
    "# Heatmap\n"
    "sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',\n"
    "            xticklabels=['Malignant','Benign'],\n"
    "            yticklabels=['Malignant','Benign'],\n"
    "            linewidths=2, linecolor='white', ax=axes[0],\n"
    "            annot_kws={'size':16, 'weight':'bold'})\n"
    "axes[0].set_xlabel('Predicted Label', fontsize=11)\n"
    "axes[0].set_ylabel('True Label', fontsize=11)\n"
    "axes[0].set_title('Confusion Matrix — LR Baseline', fontsize=12, fontweight='bold')\n\n"
    "# Metrics bar\n"
    "metrics = [acc_lr, prec_lr, rec_lr, f1_lr, auc_lr]\n"
    "labels  = ['Accuracy','Precision','Recall','F1','AUC']\n"
    "colors  = [BLUE, '#0891B2', GREEN, PURPLE, ORANGE]\n"
    "bars = axes[1].bar(labels, metrics, color=colors, alpha=0.85, edgecolor='white')\n"
    "for bar, val in zip(bars, metrics):\n"
    "    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,\n"
    "                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')\n"
    "axes[1].set_ylim(0, 1.12)\n"
    "axes[1].set_title('LR Baseline — All Metrics', fontsize=12, fontweight='bold')\n"
    "axes[1].set_ylabel('Score')\n"
    "plt.tight_layout()\n"
    "plt.savefig(r'd:\\Maincraft Internship\\Task 4\\nb_plot_cm_lr.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n\n"
    "print('\\nClassification Report:')\n"
    "print(classification_report(y_test, y_pred_lr, target_names=['Malignant','Benign']))"
)

cell_md_metrics = [
    "## Step 7 — Precision, Recall & F1-Score Interpretation\n\n",
    "**Key questions:**\n",
    "1. **Which metric is more important for medical diagnosis?** → **Recall**\n",
    "   - Missing a cancer (FN) is far worse than a false alarm (FP)\n",
    "   - High Recall = we catch as many real cancer cases as possible\n\n",
    "2. **What happens if Recall is low?** → Patients with cancer go undetected\n\n",
    "3. **Why F1-Score for imbalanced data?** → It balances precision and recall; accuracy alone inflates when one class dominates\n\n",
    "**Formulas:**\n",
    "$$\\text{Precision} = \\frac{TP}{TP+FP} \\qquad "
    "\\text{Recall} = \\frac{TP}{TP+FN} \\qquad "
    "\\text{F1} = \\frac{2 \\times P \\times R}{P+R}$$\n",
]

cell_code_roc = (
    "fig, ax = plt.subplots(figsize=(8, 6))\n\n"
    "fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)\n"
    "ax.plot(fpr_lr, tpr_lr, color=BLUE, lw=2.5,\n"
    "        label=f'LR Baseline  (AUC = {auc_lr:.3f})')\n"
    "ax.plot([0,1],[0,1], 'k--', lw=1.2, alpha=0.5, label='Random Chance')\n"
    "ax.fill_between(fpr_lr, tpr_lr, alpha=0.08, color=BLUE)\n"
    "ax.set_xlabel('False Positive Rate', fontsize=12)\n"
    "ax.set_ylabel('True Positive Rate', fontsize=12)\n"
    "ax.set_title('ROC Curve — LR Baseline', fontsize=14, fontweight='bold')\n"
    "ax.legend(fontsize=11)\n"
    "ax.set_xlim([0,1]); ax.set_ylim([0,1.02])\n"
    "plt.tight_layout()\n"
    "plt.savefig(r'd:\\Maincraft Internship\\Task 4\\nb_plot_roc_lr.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print(f'AUC Score: {auc_lr:.4f}')\n"
    "print('AUC = 1.0 -> perfect classifier | AUC = 0.5 -> random classifier')"
)

cell_md_imbalance = [
    "## Step 9 — Handle Class Imbalance\n",
    "### Technique: `class_weight='balanced'`\n\n",
    "The weight for each class is computed as:\n",
    "$$w_{class} = \\frac{n_{samples}}{n_{classes} \\times n_{class\\_samples}}$$\n\n",
    "This penalises errors on the minority class (Malignant) more heavily during training.\n",
]

cell_code_balanced = (
    "lr_bal = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')\n"
    "lr_bal.fit(X_train_sc, y_train)\n"
    "y_pred_bal = lr_bal.predict(X_test_sc)\n"
    "y_prob_bal = lr_bal.predict_proba(X_test_sc)[:, 1]\n\n"
    "acc_bal  = accuracy_score(y_test, y_pred_bal)\n"
    "prec_bal = precision_score(y_test, y_pred_bal)\n"
    "rec_bal  = recall_score(y_test, y_pred_bal)\n"
    "f1_bal   = f1_score(y_test, y_pred_bal)\n"
    "auc_bal  = roc_auc_score(y_test, y_prob_bal)\n\n"
    "print('Results — LR Balanced vs LR Baseline:')\n"
    "header = f\"{'Metric':<12} {'Baseline':>10} {'Balanced':>10} {'Change':>10}\"\n"
    "print(header)\n"
    "print('-' * 45)\n"
    "for name, v1, v2 in zip(['Accuracy','Precision','Recall','F1','AUC'],\n"
    "                         [acc_lr,prec_lr,rec_lr,f1_lr,auc_lr],\n"
    "                         [acc_bal,prec_bal,rec_bal,f1_bal,auc_bal]):\n"
    "    delta = v2 - v1\n"
    "    arrow = 'UP' if delta > 0 else 'DN'\n"
    "    print(f'{name:<12} {v1:>10.4f} {v2:>10.4f} {arrow} {abs(delta):.4f}')\n\n"
    "cm_bal = confusion_matrix(y_test, y_pred_bal)\n"
    "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
    "sns.heatmap(cm_bal, annot=True, fmt='d', cmap='Purples',\n"
    "            xticklabels=['Malignant','Benign'],\n"
    "            yticklabels=['Malignant','Benign'],\n"
    "            linewidths=2, linecolor='white', ax=axes[0],\n"
    "            annot_kws={'size':16, 'weight':'bold'})\n"
    "axes[0].set_title('Confusion Matrix — LR Balanced', fontsize=12, fontweight='bold')\n"
    "axes[0].set_xlabel('Predicted Label'); axes[0].set_ylabel('True Label')\n\n"
    "fpr_bal, tpr_bal, _ = roc_curve(y_test, y_prob_bal)\n"
    "axes[1].plot(fpr_lr,  tpr_lr,  color=BLUE,   lw=2, label=f'LR Baseline (AUC={auc_lr:.3f})')\n"
    "axes[1].plot(fpr_bal, tpr_bal, color=PURPLE, lw=2, linestyle='--', label=f'LR Balanced (AUC={auc_bal:.3f})')\n"
    "axes[1].plot([0,1],[0,1],'k--',lw=1.2,alpha=0.5)\n"
    "axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')\n"
    "axes[1].set_title('ROC: Baseline vs Balanced', fontsize=12, fontweight='bold')\n"
    "axes[1].legend(fontsize=9)\n"
    "plt.tight_layout()\n"
    "plt.savefig(r'd:\\Maincraft Internship\\Task 4\\nb_plot_balanced.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print(f'\\nClassification Report (LR Balanced):')\n"
    "print(classification_report(y_test, y_pred_bal, target_names=['Malignant','Benign']))"
)

cell_md_tree = [
    "## Step 10 — Compare with Decision Tree Classifier\n",
    "**Decision Tree** does NOT require feature scaling.\n\n",
    "Intern must compare:\n",
    "- Logistic Regression vs Decision Tree\n",
    "- Stability vs interpretability\n",
    "- Overfitting behavior\n",
]

cell_code_tree = (
    "tree = DecisionTreeClassifier(random_state=42)\n"
    "tree.fit(X_train, y_train)\n"
    "y_pred_tree = tree.predict(X_test)\n"
    "y_prob_tree = tree.predict_proba(X_test)[:, 1]\n\n"
    "acc_tree  = accuracy_score(y_test, y_pred_tree)\n"
    "prec_tree = precision_score(y_test, y_pred_tree)\n"
    "rec_tree  = recall_score(y_test, y_pred_tree)\n"
    "f1_tree   = f1_score(y_test, y_pred_tree)\n"
    "auc_tree  = roc_auc_score(y_test, y_prob_tree)\n\n"
    "# Overfitting check\n"
    "train_acc_tree = accuracy_score(y_train, tree.predict(X_train))\n"
    "train_acc_bal  = accuracy_score(y_train, lr_bal.predict(X_train_sc))\n"
    "print(f'Decision Tree  — Train: {train_acc_tree:.4f}  Test: {acc_tree:.4f}  Gap: {train_acc_tree-acc_tree:.4f}')\n"
    "print(f'LR Balanced    — Train: {train_acc_bal:.4f}   Test: {acc_bal:.4f}  Gap: {train_acc_bal-acc_bal:.4f}')\n\n"
    "cm_tree = confusion_matrix(y_test, y_pred_tree)\n"
    "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
    "sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Oranges',\n"
    "            xticklabels=['Malignant','Benign'],\n"
    "            yticklabels=['Malignant','Benign'],\n"
    "            linewidths=2, linecolor='white', ax=axes[0],\n"
    "            annot_kws={'size':16, 'weight':'bold'})\n"
    "axes[0].set_title('Confusion Matrix — Decision Tree', fontsize=12, fontweight='bold')\n"
    "axes[0].set_xlabel('Predicted Label'); axes[0].set_ylabel('True Label')\n\n"
    "fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)\n"
    "axes[1].plot(fpr_lr,   tpr_lr,   color=BLUE,   lw=2, label=f'LR Baseline (AUC={auc_lr:.3f})')\n"
    "axes[1].plot(fpr_bal,  tpr_bal,  color=PURPLE, lw=2, linestyle='--', label=f'LR Balanced (AUC={auc_bal:.3f})')\n"
    "axes[1].plot(fpr_tree, tpr_tree, color=ORANGE, lw=2, linestyle='-.', label=f'Decision Tree (AUC={auc_tree:.3f})')\n"
    "axes[1].plot([0,1],[0,1],'k--',lw=1.2,alpha=0.5)\n"
    "axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')\n"
    "axes[1].set_title('ROC — All Three Models', fontsize=12, fontweight='bold')\n"
    "axes[1].legend(fontsize=9)\n"
    "plt.tight_layout()\n"
    "plt.savefig(r'd:\\Maincraft Internship\\Task 4\\nb_plot_tree_compare.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Classification Report (Decision Tree):')\n"
    "print(classification_report(y_test, y_pred_tree, target_names=['Malignant','Benign']))"
)

cell_code_compare = (
    "metrics_df = pd.DataFrame({\n"
    "    'Model':     ['LR Baseline', 'LR Balanced (Best)', 'Decision Tree'],\n"
    "    'Accuracy':  [acc_lr,  acc_bal,  acc_tree],\n"
    "    'Precision': [prec_lr, prec_bal, prec_tree],\n"
    "    'Recall':    [rec_lr,  rec_bal,  rec_tree],\n"
    "    'F1-Score':  [f1_lr,   f1_bal,   f1_tree],\n"
    "    'AUC':       [auc_lr,  auc_bal,  auc_tree],\n"
    "})\n"
    "metrics_df.set_index('Model', inplace=True)\n"
    "metrics_df = metrics_df.round(4)\n"
    "print(metrics_df.to_string())\n\n"
    "fig, ax = plt.subplots(figsize=(11, 5))\n"
    "x = range(len(metrics_df.columns))\n"
    "w = 0.26\n"
    "bar_colors = [BLUE, PURPLE, ORANGE]\n"
    "for i, (model, row) in enumerate(metrics_df.iterrows()):\n"
    "    offset = (i-1)*w\n"
    "    bars = ax.bar([xi + offset for xi in x], row.values, w,\n"
    "                  label=model, color=bar_colors[i], alpha=0.88, edgecolor='white')\n"
    "    for bar, v in zip(bars, row.values):\n"
    "        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,\n"
    "                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')\n"
    "ax.set_xticks(list(x))\n"
    "ax.set_xticklabels(metrics_df.columns, fontsize=11)\n"
    "ax.set_ylim(0, 1.13)\n"
    "ax.set_ylabel('Score', fontsize=12)\n"
    "ax.set_title('Complete Model Comparison — All Metrics', fontsize=14, fontweight='bold')\n"
    "ax.legend(fontsize=10)\n"
    "plt.tight_layout()\n"
    "plt.savefig(r'd:\\Maincraft Internship\\Task 4\\nb_plot_full_comparison.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)

cell_md_final = [
    "## Summary & Final Model Decision\n\n",
    "### Why NOT just use Accuracy?\n",
    "- In medical diagnosis, **missing a cancer (False Negative) is far more costly** than a false alarm.\n",
    "- Accuracy is misleading when classes are imbalanced.\n",
    "- A model predicting 'Benign' for everyone would get ~63% accuracy but catch **zero cancers**.\n\n",
    "### Selected Model: Logistic Regression with `class_weight='balanced'`\n\n",
    "| Reason | Detail |\n",
    "|--------|--------|\n",
    "| Highest AUC | Best discrimination across all thresholds |\n",
    "| High Recall | Minimises missed cancer cases |\n",
    "| Imbalance handled | Auto-adjusts class weights |\n",
    "| No overfitting | Stable generalisation |\n",
    "| Interpretable | Coefficient = feature importance |\n",
]

cell_code_final = (
    "joblib.dump(lr_bal, r'd:\\Maincraft Internship\\Task 4\\task4_best_model.pkl')\n"
    "print('Best model saved: task4_best_model.pkl')\n"
    "print()\n"
    "print('=' * 55)\n"
    "print('  TASK 4 COMPLETE — Final Results')\n"
    "print('=' * 55)\n"
    "print(f'  Best Model : Logistic Regression (balanced)')\n"
    "print(f'  Accuracy   : {acc_bal:.4f}')\n"
    "print(f'  Precision  : {prec_bal:.4f}')\n"
    "print(f'  Recall     : {rec_bal:.4f}')\n"
    "print(f'  F1-Score   : {f1_bal:.4f}')\n"
    "print(f'  AUC        : {auc_bal:.4f}')\n"
    "print('=' * 55)"
)

# ── Assemble cells ─────────────────────────────────────────────────────────────
nb['cells'] = [
    md(cell_md_title),
    md(["## Step 1 — Import Required Libraries\n"]),
    code(cell_code_imports),
    md(cell_md_step2),
    code(cell_code_load),
    md(["## Step 3 — Train-Test Split\n",
        "We use `stratify=y` to preserve the class distribution in both sets.\n"]),
    code(cell_code_split),
    md(["## Step 4 — Feature Scaling\n",
        "`StandardScaler` normalises features to mean=0, std=1. Essential for Logistic Regression.\n"]),
    code(cell_code_scale),
    md(["## Step 5 — Train Baseline Logistic Regression\n",
        "Logistic Regression is the go-to baseline for binary classification.\n"]),
    code(cell_code_lr),
    md(cell_md_cm),
    code(cell_code_cm),
    md(cell_md_metrics),
    md(["## Step 8 — ROC Curve & AUC Score\n"]),
    code(cell_code_roc),
    md(cell_md_imbalance),
    code(cell_code_balanced),
    md(cell_md_tree),
    code(cell_code_tree),
    md(["## Full Model Comparison Table\n"]),
    code(cell_code_compare),
    md(cell_md_final),
    code(cell_code_final),
]

nb_path = r'd:\Maincraft Internship\Task 4\AI_ML_Task4_Classification_Evaluation.ipynb'
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Notebook saved: {nb_path}')
print(f'Total cells: {len(nb["cells"])}')
