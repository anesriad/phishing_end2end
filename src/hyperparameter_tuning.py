# hyperparameter_tuning.py
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from data_loader import load_email_data
from preprocess import preprocess_dataframe
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib


def objective(trial):
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = "liblinear" if penalty == "l1" else "lbfgs"

    model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)
    score = cross_val_score(model, X_train, y_train, scoring="f1", cv=3).mean()
    return score


if __name__ == "__main__":
    df = load_email_data("Phishing_Email.csv")
    df = preprocess_dataframe(df)

    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best parameters:", study.best_params)

    # Save best model and vectorizer
    best_model = LogisticRegression(**study.best_params, max_iter=1000)
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, "src/models/model.pkl")
    joblib.dump(vectorizer, "src/models/vectorizer.pkl")