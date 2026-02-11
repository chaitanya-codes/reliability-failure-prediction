import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from preprocessing import load_data, preprocess

DATA_PATH = "../data/sensor_data.csv"

def evaluate():
    model = joblib.load("../best_model.pkl")
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess(df)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    print("ROC-AUC:", roc_auc_score(y_test, probs))

if __name__ == "__main__":
    evaluate()