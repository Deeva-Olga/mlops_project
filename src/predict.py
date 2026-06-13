import os
import pandas as pd
import mlflow
import mlflow.sklearn

# Настройки MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ID эксперимента и ран (можно получить из MLflow UI)
EXPERIMENT_NAME = "production-ml-spring-2026"
RUN_ID = "YOUR_RUN_ID_HERE"  # Замените на реальный ID из MLflow

# === Загрузка модели из MLflow ===
print(f"Загрузка модели из MLflow (run_id={RUN_ID})...")
model_uri = f"runs:/{RUN_ID}/model"
model = mlflow.sklearn.load_model(model_uri)
print("Модель загружена!")

# === Загрузка тестовых данных ===
test = pd.read_parquet("data/test.parquet.snappy")

# Подготовка текста (как в train.py)
TEXT_COLS = [c for c in test.columns if c not in ["category_ind", "ID"] and test[c].dtype == "object"]
for col in TEXT_COLS:
    test[col] = test[col].fillna("").astype(str)

test["text"] = test[TEXT_COLS].apply(lambda r: " ".join(r.astype(str)), axis=1)

# === Предсказания ===
print("🔮 Предсказания...")
predictions = model.predict(test["text"])

# === Создание submission.csv ===
submission = pd.DataFrame({
    "ID": test["ID"],
    "category_ind": predictions
})
submission.to_csv("submission.csv", index=False)
print(f"✅ submission.csv создан! {len(submission)} строк")
print(submission.head())
