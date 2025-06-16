# evaluate.py
import joblib
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))


if __name__ == '__main__':
    from data_loader import load_email_data
    from preprocess import preprocess_dataframe
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split

    df = load_email_data("Phishing_Email.csv")
    df = preprocess_dataframe(df)

    vectorizer = joblib.load("src/models/vectorizer.pkl")
    X = vectorizer.transform(df['clean_text'])
    y = df['label']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load("src/models/model.pkl")
    evaluate_model(model, X_test, y_test)
