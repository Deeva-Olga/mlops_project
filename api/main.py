import os
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any

# === Настройки ===
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:////workspaces/mlops_project/mlflow.db")
RUN_ID = os.getenv("MLFLOW_RUN_ID", "1121eea5d65248948720fce2a9eb3326")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# === Загрузка модели при старте ===
print(f"📥 Загрузка модели из MLflow (run_id={RUN_ID})...")
model_uri = f"runs:/{RUN_ID}/model"
model = mlflow.sklearn.load_model(model_uri)
print("✅ Модель загружена!")

# === FastAPI приложение ===
app = FastAPI(title="Product Category Prediction API", version="1.0.0")

class ProductRequest(BaseModel):
    url: str
    texts: Dict[str, Any]  # Вложенный JSON с текстовыми полями
    image_url: Optional[str] = None

class PredictionResponse(BaseModel):
    category_ind: int

@app.get("/")
def root():
    return {"message": "Product Category Prediction API", "status": "running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ProductRequest):
    try:
        # Объединяем все текстовые поля в одну строку (как в train.py)
        text_parts = []
        for key, value in request.texts.items():
            if value and str(value).strip():
                text_parts.append(str(value))
        
        # Если есть URL, добавляем его тоже
        if request.url and request.url.strip():
            text_parts.append(request.url)
        
        # Если есть image_url, добавляем его тоже
        if request.image_url and request.image_url.strip():
            text_parts.append(request.image_url)
        
        # Объединяем всё в одну строку через пробел
        combined_text = " ".join(text_parts)
        
        # Если текст пустой, используем пустую строку
        if not combined_text.strip():
            combined_text = ""
        
        # Предсказание
        prediction = model.predict([combined_text])[0]
        
        return PredictionResponse(
            category_ind=int(prediction)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}
