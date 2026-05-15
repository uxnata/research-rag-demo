"""
Индексирование чанков: эмбеддинги + загрузка в Qdrant.

Что делает:
1. Загружает чанки из data/processed/chunks.json
2. Скачивает модель эмбеддингов deepvk/USER-bge-m3 (один раз, ~2 GB)
3. Превращает текст каждого чанка в вектор (массив из 1024 чисел)
4. Сохраняет вектора + метаданные в локальный Qdrant

Что такое эмбеддинг:
    Модель читает текст и сжимает его смысл в массив чисел.
    Похожие по смыслу тексты → похожие массивы → близкие точки
    в 1024-мерном пространстве. Потом, когда приходит вопрос,
    мы превращаем его в такой же вектор и ищем ближайших «соседей».

Почему deepvk/USER-bge-m3:
    - Заточена под русский язык (в отличие от англоязычных моделей)
    - Открытая, бесплатная
    - Размерность 1024 — хороший баланс качества и скорости

Использование:
    python src/indexing/indexer.py
"""

import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

# Пути
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
QDRANT_PATH = Path(__file__).parents[2] / "data" / "qdrant_db"

# Имя коллекции в Qdrant (как таблица в обычной БД).
COLLECTION_NAME = "cbr_reports"

# Модель эмбеддингов (LaBSE-ru-turbo).
# При первом запуске скачается с Hugging Face (~500 MB),
# потом будет грузиться из кеша (~/.cache/huggingface/).
EMBEDDING_MODEL = "sergeyzh/LaBSE-ru-turbo"

# Размерность вектора — зависит от модели. У LaBSE-ru-turbo это 768.
VECTOR_SIZE = 768

# Размер батча для эмбеддингов.
# Почему не все 242 чанка сразу? Потому что 8 GB RAM.
# Батчами по 16 — модель обрабатывает 16 текстов, отдаёт результат,
# освобождает память, берёт следующие 16.
BATCH_SIZE = 16


def load_chunks() -> list[dict]:
    """Загружает чанки из JSON."""
    chunks_path = PROCESSED_DIR / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Не нашёл {chunks_path}. Сначала запусти chunker.py"
        )
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_qdrant_collection(client: QdrantClient):
    """
    Создаёт коллекцию в Qdrant (если ещё нет).

    Коллекция = хранилище векторов с одинаковой размерностью.
    Distance.COSINE — мера похожести: чем ближе к 1.0,
    тем более похожи тексты по смыслу.
    """
    # Проверяем, существует ли коллекция
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"Коллекция '{COLLECTION_NAME}' уже существует — пересоздаю")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    print(f"Коллекция '{COLLECTION_NAME}' создана")


def main():
    print("=" * 50)
    print("Индексирование чанков ЦБ РФ")
    print("=" * 50)

    # Шаг 1: загружаем чанки
    chunks = load_chunks()
    print(f"\n1. Загружено {len(chunks)} чанков")

    # Шаг 2: загружаем модель эмбеддингов
    print(f"\n2. Загружаю модель {EMBEDDING_MODEL}...")
    print("   (первый раз скачивает ~2 GB, потом из кеша)")
    start = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"   Модель загружена за {time.time() - start:.1f} сек")

    # Шаг 3: создаём Qdrant
    print(f"\n3. Создаю Qdrant в {QDRANT_PATH}")
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))
    create_qdrant_collection(client)

    # Шаг 4: эмбеддим и загружаем батчами
    print(f"\n4. Эмбеддинг и загрузка ({len(chunks)} чанков, батч={BATCH_SIZE})")
    start = time.time()

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]

        # Собираем тексты батча
        texts = [chunk["text"] for chunk in batch]

        # Эмбеддим батч. model.encode возвращает numpy-массив
        # размером (len(texts), 1024) — по 1024 числа на каждый текст.
        embeddings = model.encode(texts, show_progress_bar=False)

        # Формируем точки для Qdrant.
        # Каждая точка = id + вектор + payload (метаданные).
        # payload — это любые данные, которые мы хотим хранить
        # рядом с вектором. Потом при поиске они вернутся вместе
        # с результатом — не нужно делать отдельный запрос.
        points = []
        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            point = PointStruct(
                id=i + j,  # уникальный ID точки
                vector=embedding.tolist(),  # numpy → list (Qdrant хочет list)
                payload={
                    "text": chunk["text"],
                    "source": chunk["metadata"]["source"],
                    "page": chunk["metadata"]["page"],
                    "year": chunk["metadata"]["year"],
                    "month": chunk["metadata"]["month"],
                    "month_name": chunk["metadata"]["month_name"],
                    "report_type": chunk["metadata"]["report_type"],
                },
            )
            points.append(point)

        # Загружаем батч в Qdrant
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"   Батч {i // BATCH_SIZE + 1}/"
              f"{(len(chunks) - 1) // BATCH_SIZE + 1}: "
              f"загружено {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    elapsed = time.time() - start
    print(f"\n   Готово за {elapsed:.1f} сек")

    # Шаг 5: проверка — делаем тестовый поиск
    print("\n5. Тестовый поиск: 'ключевая ставка'")
    query_vector = model.encode("ключевая ставка").tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3,
    ).points

    for rank, result in enumerate(results, 1):
        print(f"\n   --- Результат {rank} (score: {result.score:.4f}) ---")
        print(f"   Источник: {result.payload['source']}, "
              f"стр. {result.payload['page']}")
        print(f"   Период: {result.payload['month_name']} "
              f"{result.payload['year']}")
        # Первые 200 символов текста
        print(f"   Текст: {result.payload['text'][:200]}...")

    print(f"\nВсё готово! Qdrant база: {QDRANT_PATH}")
    print(f"Коллекция: {COLLECTION_NAME}, точек: {len(chunks)}")


if __name__ == "__main__":
    main()
