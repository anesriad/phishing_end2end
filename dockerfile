# ─── 1. Base image ─────────────────────────────────────────────
FROM python:3.11-slim

# ─── 2. Create & switch to the working dir ────────────────────
WORKDIR /app

# ─── 3. Install dependencies ──────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── 4. Copy in the trained model artifact ────────────────────
# (we ran `mlflow artifacts download … -o src/models/model.pkl`
# in CI, so this folder now exists in the repo)
COPY src/models ./src/models

# ─── 5. Copy the rest of the source code ───────────────────────
COPY . .

# ─── 6. Expose port & start ────────────────────────────────────
EXPOSE 8080
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
