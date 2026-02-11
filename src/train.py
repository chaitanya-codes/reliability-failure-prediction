import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from preprocessing import load_data, preprocess

DATA_PATH = "../data/sensor_data.csv"

def train():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess(df)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced"),
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=5
        )
    }

    best_model = None
    best_auc = 0

    for name, model in models.items():
        print(f"\nTraining {name}...")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)

        print("CV ROC-AUC:", np.mean(cv_scores))
        print("Test ROC-AUC:", auc)
        print("F1 Score:", f1)

        if auc > best_auc:
            best_auc = auc
            best_model = model

    joblib.dump(best_model, "../best_model.pkl")
    print("\nBest model saved!")

if __name__ == "__main__":
    train()