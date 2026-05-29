"""
predict_ui.py — House Price Predictor UI Script
Maincraft Internship | AI & Machine Learning Task 1

Usage:
    python predict_ui.py

Loads the saved LinearRegression model and lets you input feature values
to predict median house prices.
"""

import pickle
import os
import sys

# ─── Load Model ─────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'house_price_model.pkl')

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file 'house_price_model.pkl' not found.")
        print("   Please run the Jupyter Notebook first to train and save the model.")
        sys.exit(1)
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data['feature_names'], data['metrics']

# ─── Feature Descriptions ────────────────────────────────────────────────────

FEATURE_DESCRIPTIONS = {
    'MedInc'     : 'Median income in block group (in $10,000s)',
    'HouseAge'   : 'Median house age in block group (in years)',
    'AveRooms'   : 'Average number of rooms per household',
    'AveBedrms'  : 'Average number of bedrooms per household',
    'Population' : 'Block group population',
    'AveOccup'   : 'Average number of household members',
    'Latitude'   : 'Block group latitude (e.g., 37.88 for San Francisco area)',
    'Longitude'  : 'Block group longitude (e.g., -122.23 for San Francisco area)',
}

FEATURE_EXAMPLES = {
    'MedInc'     : 3.87,
    'HouseAge'   : 29.0,
    'AveRooms'   : 5.43,
    'AveBedrms'  : 1.10,
    'Population' : 1425.0,
    'AveOccup'   : 3.07,
    'Latitude'   : 34.20,
    'Longitude'  : -118.39,
}

# ─── Print Header ────────────────────────────────────────────────────────────

def print_header():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     🏠  California House Price Predictor                     ║")
    print("║     Linear Regression Model  |  Maincraft Internship         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

def print_metrics(metrics):
    print("  📊 Model Performance Metrics:")
    print(f"     MAE  : {metrics['test_mae']:.4f}  (avg error ~${metrics['test_mae']*100000:,.0f})")
    print(f"     RMSE : {metrics['test_rmse']:.4f}  (~${metrics['test_rmse']*100000:,.0f})")
    print(f"     R²   : {metrics['test_r2']:.4f}  ({metrics['test_r2']*100:.1f}% variance explained)")
    print()

# ─── Get User Input ──────────────────────────────────────────────────────────

def get_feature_inputs(feature_names):
    print("  📝 Enter feature values for the block group you want to predict.")
    print("     (Press Enter to use the example value shown in brackets)\n")

    values = []
    for feat in feature_names:
        desc    = FEATURE_DESCRIPTIONS.get(feat, feat)
        example = FEATURE_EXAMPLES.get(feat, 0.0)

        while True:
            try:
                user_input = input(f"  {feat:12s} [{example:8.2f}] — {desc}\n  > ").strip()
                if user_input == '':
                    val = example
                else:
                    val = float(user_input)
                values.append(val)
                break
            except ValueError:
                print(f"    ⚠️  Please enter a valid number.\n")

    return values

# ─── Predict ─────────────────────────────────────────────────────────────────

def predict(model, scaler, feature_names, values):
    import numpy as np
    X = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return pred

# ─── Display Results ─────────────────────────────────────────────────────────

def display_result(values, feature_names, prediction):
    print()
    print("  ─" * 32)
    print("  📋 Input Summary:")
    for feat, val in zip(feature_names, values):
        print(f"     {feat:12s}: {val:.4f}")
    print()
    print("  ─" * 32)
    price_100k = prediction
    price_dollars = prediction * 100_000
    print(f"  🏠 Predicted Median House Value: ${price_100k:.4f} (×$100,000)")
    print(f"                                  ≈ ${price_dollars:,.0f}")
    print("  ─" * 32)
    print()

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print_header()

    # Load model
    print("  Loading model...")
    model, scaler, feature_names, metrics = load_model()
    print("  ✅ Model loaded successfully!\n")
    print_metrics(metrics)

    while True:
        try:
            values   = get_feature_inputs(feature_names)
            prediction = predict(model, scaler, feature_names, values)
            display_result(values, feature_names, prediction)

            again = input("  Predict another house? (y/n): ").strip().lower()
            if again != 'y':
                print("\n  Thanks for using the House Price Predictor! 👋\n")
                break
            print()

        except KeyboardInterrupt:
            print("\n\n  Exiting... Goodbye! 👋\n")
            break

if __name__ == '__main__':
    main()
