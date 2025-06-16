# src/train.py
from data_loader import load_email_data
from preprocess import preprocess_dataframe
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def main():
    df = load_email_data("Phishing_Email.csv")
    df = preprocess_dataframe(df)

    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))


if __name__ == "__main__":
    main()
