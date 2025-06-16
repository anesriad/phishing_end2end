# src/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from nltk.corpus import stopwords

app = FastAPI()
model = joblib.load("src/models/model.pkl")
vectorizer = joblib.load("src/models/vectorizer.pkl")
stop_words = set(stopwords.words("english"))

class EmailInput(BaseModel):
    email: str

def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    return " ".join([word for word in text.split() if word not in stop_words])

@app.post("/predict")
def predict(input: EmailInput):
    cleaned = clean_text(input.email)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector).max()
    label = "Phishing" if pred == 1 else "Safe"
    return {"label": label, "confidence": round(prob, 2)}