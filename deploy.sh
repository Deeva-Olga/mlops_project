#!/bin/bash

echo "Запуск MLOps стека..."

# Создаём бакет в MinIO для MLflow
echo "Создание бакета mlflow в MinIO..."
docker-compose up -d minio
sleep 5
docker-compose exec minio mc alias set local http://localhost:9000 minioadmin minioadmin
docker-compose exec minio mc mb local/mlflow --ignore-existing

# Запускаем весь стек
echo "Запуск docker-compose..."
docker-compose up -d --build

echo "Готово!"
echo "MLflow UI: http://localhost:5000"
echo "API: http://localhost:8000"
echo "MinIO Console: http://localhost:9001"
