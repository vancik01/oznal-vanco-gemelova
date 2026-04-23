"""Compare three feature sets on the same task: predict result at minute 15.

  A. Oracle baseline (oracle_features.csv)
  B. Frame-derived (frame_features.csv)
  C. Combined (oracle + frame)

For each: 5-fold stratified CV of LogisticRegression + RandomForest.
Metrics: Accuracy, AUC-ROC, log-loss, Brier.
"""
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

HERE = Path(__file__).parent
ORACLE = HERE / "oracle_features.csv"
FRAME = HERE / "frame_features.csv"
COMBINED = HERE / "combined_features.csv"

SEED = 42
N_SPLITS = 5

KEY_COLS = {"ogid", "side"}
LABEL = "result"


def load(p):
    df = pd.read_csv(p)
    return df


def evaluate(name, df):
    y = df[LABEL].astype(int).values
    X = df.drop(columns=list(KEY_COLS) + [LABEL]).select_dtypes(include=[np.number]).fillna(0)
    print(f"\n=== {name} ===")
    print(f"  shape: {X.shape}   positive rate: {y.mean():.3f}")
    print(f"  cols: {list(X.columns)[:8]}{'...' if X.shape[1]>8 else ''}")

    for model_name, pipe in [
        ("LogReg", Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, C=1.0))])),
        ("RandomForest", RandomForestClassifier(n_estimators=400, max_depth=8, random_state=SEED, n_jobs=-1)),
    ]:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        res = cross_validate(
            pipe, X, y, cv=skf,
            scoring=["accuracy", "roc_auc", "neg_log_loss", "neg_brier_score"],
            n_jobs=-1, return_train_score=False,
        )
        acc = res["test_accuracy"]; auc = res["test_roc_auc"]
        ll  = -res["test_neg_log_loss"]; br = -res["test_neg_brier_score"]
        print(f"  {model_name:14s}  acc={acc.mean():.3f}±{acc.std():.3f}  auc={auc.mean():.3f}±{auc.std():.3f}  logloss={ll.mean():.3f}  brier={br.mean():.3f}")


def feature_importance_report(df, label, top_k=10):
    y = df[LABEL].astype(int).values
    X = df.drop(columns=list(KEY_COLS) + [LABEL]).select_dtypes(include=[np.number]).fillna(0)
    rf = RandomForestClassifier(n_estimators=400, max_depth=8, random_state=SEED, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\n[{label}] Top-{top_k} RF importances:")
    for name, val in imp.head(top_k).items():
        print(f"  {name:35s} {val:.4f}")


def main():
    oracle = load(ORACLE)
    frame = load(FRAME)
    combined = load(COMBINED)
    evaluate("A. Oracle baseline (@15 min)", oracle)
    evaluate("B. Frame-derived (trajectories + events)", frame)
    evaluate("C. Combined", combined)
    feature_importance_report(combined, "combined")


if __name__ == "__main__":
    main()
