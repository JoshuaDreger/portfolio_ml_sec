# src/evaluate.py
import argparse
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score

def evaluate(input_csv, model_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(input_csv)
    X = df.drop(columns=["is_attack"])
    y_true = df["is_attack"]

    model = joblib.load(model_path)
    print("Loaded model:", model)

    # 1) Compute anomaly scores (negative = more anomalous)
    scores = model.decision_function(X)  # higher → more “normal”
    y_pred = model.predict(X)            # {1: inlier, -1: outlier}
    y_pred_bin = pd.Series(y_pred).map({1: 0, -1: 1})  # convert to 0/1

    # 2) Metrics
    print("Classification Report (IsolationForest vs. true labels):")
    print(classification_report(y_true, y_pred_bin, digits=4))

    # 3) ROC AUC
    auc = roc_auc_score(y_true, -scores)  # invert score so higher → more attack
    print(f"ROC AUC score: {auc:.4f}")

    # 4) Plot histogram of anomaly scores
    plt.figure(figsize=(6,4))
    plt.hist(scores[y_true==0], bins=50, alpha=0.7, label="Normal")
    plt.hist(scores[y_true==1], bins=50, alpha=0.7, label="Attack")
    plt.legend()
    plt.title("Histogram of IsolationForest Scores")
    plt.xlabel("Score (higher = more normal)")
    plt.ylabel("Count")
    plt.tight_layout()
    hist_path = os.path.join(outdir, "anomaly_score_histogram.png")
    plt.savefig(hist_path)
    print(f"Saved histogram to {hist_path}")
    plt.close()

    # 5) Save a small CSV with results (optional)
    results_df = pd.DataFrame({
        "score": scores,
        "y_true": y_true,
        "y_pred": y_pred_bin
    })
    results_df.sample(5).to_csv(os.path.join(outdir, "sample_results.csv"), index=False)
    print("Saved sample results.")
    # End
    print("Evaluation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True, help="Processed CSV")
    parser.add_argument("--model",    required=True, help="Path to trained model (.joblib)")
    parser.add_argument("--outdir",   required=True, help="Directory to save plots/CSVs")
    args = parser.parse_args()
    evaluate(args.input, args.model, args.outdir)
