# Research RAG Demo

RAG-ассистент для исследований по российскому банковскому и финтех-рынку.

## Быстрый старт

```bash
# 1. Создать окружение
conda env create -f environment.yml
conda activate research-rag

# 2. Настроить ключи
cp .env.example .env
# Отредактировать .env — вставить свой GIGACHAT_CREDENTIALS

# 3. Проверить подключение к GigaChat
python src/generation/test_gigachat.py
```

## Структура

```
src/
├── ingestion/    # Сбор и чанкинг документов
├── indexing/     # Эмбеддинги и Qdrant
├── retrieval/    # Поиск релевантных чанков
├── generation/   # LLM-генерация ответов
└── evaluation/   # Тест-сет и метрики RAGAS
```
