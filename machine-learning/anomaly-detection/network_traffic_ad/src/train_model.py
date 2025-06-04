# src/train_model.py
import argparse
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

def train(input_csv, model_out):
    df = pd.read_csv(input_csv)
    print(f"Loaded processed data: {df.shape}")

    # 1) Separate features and labels
    X = df.drop(columns=["is_attack"])
    y = df["is_attack"]

    # 2) Train IsolationForest on all data but only use “normal” points as ‘inliers’
    X_norm = X[y == 0]

    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,  # initial guess: 10% anomalies
        random_state=42,
        n_jobs=-1
    )
    print("Fitting IsolationForest on normal samples...")
    model.fit(X_norm)
    print("Training complete.")

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    print(f"Model saved to {model_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True, help="Processed CSV")
    parser.add_argument("--model_out", required=True, help="Path to save the model (.joblib)")
    args = parser.parse_args()
    train(args.input, args.model_out)
