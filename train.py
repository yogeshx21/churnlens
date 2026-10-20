"""ChurnLens training: python train.py   (add --fast for a quick run)"""
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import MULTI, RAW_FEATURES, clean, encode, load_data

warnings.filterwarnings("ignore")
FAST = "--fast" in sys.argv
SEED = 42
CV = 3 if FAST else 5
N_ITER = 2 if FAST else 6


def get_boosting(pos_weight):
    try:
        from xgboost import XGBClassifier
        m = XGBClassifier(eval_metric="logloss", scale_pos_weight=pos_weight,
                          random_state=SEED, n_jobs=-1)
        grid = {"n_estimators": [100, 200, 300], "max_depth": [2, 3, 4],
                "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.8, 1.0]}
        return "XGBoost", m, grid
    except Exception:
        m = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
        grid = {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.05, 0.1],
                "max_iter": [100, 200, 300]}
        return "GradientBoosting", m, grid


def metrics(y, proba, thr):
    pred = (proba >= thr).astype(int)
    return {
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, proba),
        "PR-AUC": average_precision_score(y, proba),
    }


def main():
    df = clean(load_data())
    y = df["Churn"]
    raw = df[RAW_FEATURES]
    X = encode(raw)
    columns = list(X.columns)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=SEED)

    bname, bmodel, bgrid = get_boosting(pos_weight)
    candidates = {
        "Logistic Regression": (
            make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
            {"logisticregression__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
            {"n_estimators": [150, 300], "max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 20]}),
        bname: (bmodel, bgrid),
    }

    fitted, rows = {}, []
    for name, (model, grid) in candidates.items():
        print("Tuning", name, "...")
        s = RandomizedSearchCV(model, grid, n_iter=min(N_ITER, int(np.prod([len(v) for v in grid.values()]))),
                               scoring="f1", cv=skf, random_state=SEED, n_jobs=-1)
        s.fit(Xtr, ytr)
        fitted[name] = s.best_estimator_
        m = metrics(yte, s.best_estimator_.predict_proba(Xte)[:, 1], 0.5)
        m.update({"Model": name, "CV F1": s.best_score_})
        rows.append(m)
    mt = pd.DataFrame(rows).set_index("Model")[["CV F1", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]].round(3)
    best_name = mt["CV F1"].idxmax()
    best = fitted[best_name]
    print(mt)
    print("Selected model:", best_name)

    # threshold tuning on training data only (cross-validated probabilities)
    oof = cross_val_predict(best, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
    grid_thr = np.arange(0.2, 0.75, 0.025)
    thr = float(grid_thr[int(np.argmax([f1_score(ytr, (oof >= t).astype(int)) for t in grid_thr]))])
    final = metrics(yte, best.predict_proba(Xte)[:, 1], thr)
    print("Threshold %.3f ->" % thr, {k: round(v, 3) for k, v in final.items()})

    # fairness audit on the test set
    te = df.loc[Xte.index]
    proba = best.predict_proba(Xte)[:, 1]
    pred = (proba >= thr).astype(int)
    groups = {
        "gender": te["gender"].values if "gender" in te else np.array(["n/a"] * len(te)),
        "senior": np.where(te["SeniorCitizen"].values == 1, "Senior", "Non-senior"),
    }
    frows = []
    for gname, gvals in groups.items():
        for g in sorted(set(gvals)):
            mk = gvals == g
            frows.append({
                "Group type": gname, "Group": g, "Customers": int(mk.sum()),
                "Precision": precision_score(yte[mk], pred[mk], zero_division=0),
                "Recall": recall_score(yte[mk], pred[mk], zero_division=0),
                "Predicted churn rate": pred[mk].mean(),
                "Actual churn rate": yte[mk].mean(),
            })
    fair = pd.DataFrame(frows).round(3)

    bundle = {
        "model": best, "model_name": best_name, "columns": columns, "threshold": thr,
        "X_train": Xtr.reset_index(drop=True), "y_train": ytr.reset_index(drop=True), "X_test_sample": Xte.sample(min(60, len(Xte)), random_state=0),
        "metrics": mt, "final_metrics": {k: round(v, 3) for k, v in final.items()},
        "fairness": fair,
        "options": {c: sorted(df[c].unique().tolist()) for c in MULTI},
        "tenure_max": int(df["tenure"].max()),
        "charges": (float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())),
    }
    os.makedirs("models", exist_ok=True)
    joblib.dump(bundle, os.path.join("models", "bundle.joblib"))
    print("Saved models/bundle.joblib")
    print(fair)


if __name__ == "__main__":
    main()
