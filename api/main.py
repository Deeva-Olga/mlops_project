import os
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# === Настройки ===
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
RUN_ID = os.getenv("MLFLOW_RUN_ID", "YOUR_RUN_ID_HERE")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# === Загрузка модели при старте ===
print(f"📥 Загрузка модели из MLflow (run_id={RUN_ID})...")
model_uri = f"runs:/{RUN_ID}/model"
model = mlflow.sklearn.load_model(model_uri)
print("✅ Модель загружена!")

# === FastAPI приложение ===
app = FastAPI(title="Product Category Prediction API", version="1.0.0")

class ProductRequest(BaseModel):
    text: str  # Текст товара (или JSON с полями name, description, etc.)

class PredictionResponse(BaseModel):
    category_ind: int
    confidence: float  # Опционально: уверенность модели

@app.get("/")
def root():
    return {"message": "Product Category Prediction API", "status": "running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ProductRequest):
    try:
        # Предсказание
        prediction = model.predict([request.text])[0]
        
        # Опционально: получить уверенность (вероятности)
        # probabilities = model.predict_proba([request.text])[0]
        # confidence = float(max(probabilities))
        
        return PredictionResponse(
            category_ind=int(prediction),
            confidence=1.0  # Замените на реальную уверенность
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(requests: List[ProductRequest]):
    try:
        texts = [r.text for r in requests]
        predictions = model.predict(texts)
        return {"predictions": [int(p) for p in predictions]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
