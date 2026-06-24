# MLOps: Классификация товаров по категориям

Проект для автоматической классификации товаров по их текстовому описанию, URL и изображениям с использованием MLflow для отслеживания экспериментов и FastAPI для предоставления REST API.

## Описание

Система предсказывает категорию товара на основе:
- Текстовых полей (название, описание, модель, производитель и др.)
- URL товара
- URL изображения товара

**Метрика качества:** Macro F1-Score

## Архитектура
┌─────────────┐

│   Данные    │ (Parquet)

└──────┬──────┘
       
┌─────────────────┐

│  train.py       │ → MLflow Tracking

│  • Обучение     │   (SQLite + File Store)

│  • Валидация    │

└──────┬──────────┘


┌─────────────────┐

│  FastAPI        │

│  • REST API     │

│  • /predict     │

└─────────────────┘

## Технологии
- ML: scikit-learn (TfidfVectorizer + RandomForest)
- API: FastAPI, Pydantic
- ML Tracking: MLflow
- Деплой: Bash-скрипт

## Запустить сервис
1. Запуск:
   $ ./deploy.sh

2. Открыть браузер:
   http://localhost:8000/docs

3. Тестирование:
   POST /predict
{
    "url": "https://example.com/product/123",
    "texts": {
        "name": "Смартфон Xiaomi Redmi Note 12",
        "description": "Смартфон с камерой 48Мп",
        "model": "Redmi Note 12",
        "type_prefix": "Смартфон",
        "vendor": "Xiaomi"
    },
    "image_url": "http://example.com/image.jpg"
}
   
   → {"category_ind": 42}


## Структура проекта

mlops_project/

├── src/

│   ├── train.py              # Скрипт обучения модели

│   └── predict.py            # Скрипт для batch-предсказаний

├── api/

│   ├── main.py               # FastAPI приложение

│   ├── requirements.txt      # Зависимости API

│   └── Dockerfile            # Docker образ

├── data/

│   ├── train.parquet.snappy  # Обучающие данные

│   ├── test.parquet.snappy   # Тестовые данные

│   └── tree.csv              # Иерархия категорий

├── run_service.sh            # Скрипт развёртывания

├── mlflow.db                 # MLflow база данных

└── README.md                 # Документация


## Презентация
https://docs.google.com/presentation/d/1Agi9-630prNYDQcPAzPnoeDjNGSDSBFS/edit?usp=sharing&ouid=108872823753120218896&rtpof=true&sd=true

