from fastapi import FastAPI, Request, Form, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib, re
from nltk.corpus import stopwords

app = FastAPI()
templates = Jinja2Templates(directory="src/templates")

model = joblib.load("src/models/model.pkl")
vectorizer = joblib.load("src/models/vectorizer.pkl")
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    return " ".join([word for word in text.split() if word not in stop_words])

class EmailInput(BaseModel):
    email: str

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_form(request: Request, email: str = Form(...)):
    cleaned = clean_text(email)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector).max()
    label = "Phishing" if pred == 1 else "Safe"
    result = {"label": label, "confidence": round(prob, 2)}
    return templates.TemplateResponse("form.html", {"request": request, "result": result})

# 👇 JSON endpoint for testing and API
@app.post("/predict-json")
async def predict_json(input: EmailInput):
    cleaned = clean_text(input.email)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector).max()
    label = "Phishing" if pred == 1 else "Safe"
    return {"label": label, "confidence": round(prob, 2)}
