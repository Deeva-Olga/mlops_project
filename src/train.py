import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# === 1. Настройки MLflow (читает из переменных окружения!) ===
# MLflow сам подключится к MinIO через эти переменные
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("production-ml-spring-2026")

# === 2. Загрузка данных ===
print("Загрузка данных...")
train = pd.read_parquet("data/train.parquet.snappy")
test = pd.read_parquet("data/test.parquet.snappy")

# === 3. Подготовка текста ===
TEXT_COLS = [c for c in train.columns if c not in ["category_ind", "ID"] and train[c].dtype == "object"]

for col in TEXT_COLS:
    train[col] = train[col].fillna("").astype(str)
    test[col] = test[col].fillna("").astype(str)

# Объединяем все текстовые поля в одно
train["text"] = train[TEXT_COLS].apply(lambda r: " ".join(r.astype(str)), axis=1)
test["text"] = test[TEXT_COLS].apply(lambda r: " ".join(r.astype(str)), axis=1)

y = train["category_ind"]
X_train_text = train["text"]

# === 4. Разделение на train/val ===
X_tr, X_val, y_tr, y_val = train_test_split(X_train_text, y, test_size=0.2, random_state=42)

# === 5. Создание Pipeline ===
# ВАЖНО: Pipeline объединяет векторизатор и модель в один объект!
# Теперь model.predict(["текст товара"]) будет работать напрямую
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=500, stop_words="russian", ngram_range=(1,2), min_df=2)),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1))
])

# === 6. Обучение с логированием в MLflow ===
print("Обучение модели...")
with mlflow.start_run(run_name="rf_pipeline_200"):
    # Обучение
    pipeline.fit(X_tr, y_tr)
    
    # Предсказания
    pred_tr = pipeline.predict(X_tr)
    pred_val = pipeline.predict(X_val)
    
    # Метрики
    train_acc = accuracy_score(y_tr, pred_tr)
    test_acc = accuracy_score(y_val, pred_val)
    train_f1 = f1_score(y_tr, pred_tr, average="macro", zero_division=0)
    test_f1 = f1_score(y_val, pred_val, average="macro", zero_division=0)
    
    # Логируем метрики
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("train_f1_macro", train_f1)
    mlflow.log_metric("test_f1_macro", test_f1)
    
    # Логируем параметры
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 20)
    mlflow.log_param("tfidf_max_features", 500)
    
    # Сохраняем classification_metrics.json
    metrics = {
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "train_f1_macro": round(train_f1, 4),
        "test_f1_macro": round(test_f1, 4),
    }
    with open("classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    mlflow.log_artifact("classification_metrics.json")
    
    # Сохраняем dataset_label_counts.json
    counts = {str(k): int(v) for k, v in pd.Series(y_tr).value_counts().to_dict().items()}
    with open("dataset_label_counts.json", "w") as f:
        json.dump(counts, f, indent=2)
    mlflow.log_artifact("dataset_label_counts.json")
    
    # === ЛОГИРОВАНИЕ МОДЕЛИ ===
    # MLflow сам загрузит модель в MinIO (через переменные окружения)
    mlflow.sklearn.log_model(pipeline, "model")
    
    print(f"Обучение завершено! test_f1={test_f1:.4f}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
