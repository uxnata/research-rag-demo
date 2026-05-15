"""
Скрипт обновления: проверяет новые отчёты ЦБ и добавляет в базу.

Что делает:
1. Заходит на сайт ЦБ, скачивает PDF, которых ещё нет
2. Чанкит только новые файлы
3. Добавляет новые чанки в Qdrant (без пересоздания всей базы)

Можно запускать вручную или по расписанию (cron).

Использование:
    python src/ingestion/update.py
"""

import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Импортируем наши модули — переиспользуем уже написанный код.
# Это главный плюс модульной структуры проекта:
# вместо копипасты функций — один импорт.
from src.ingestion.scraper_cbr import get_pdf_links, download_pdf, RAW_DATA_DIR
from src.ingestion.chunker import chunk_pdf, PROCESSED_DIR

# Пути и настройки (те же, что в indexer.py)
QDRANT_PATH = Path(__file__).parents[2] / "data" / "qdrant_db"
COLLECTION_NAME = "cbr_reports"
EMBEDDING_MODEL = "sergeyzh/LaBSE-ru-turbo"
BATCH_SIZE = 16

# Файл, в котором храним список уже проиндексированных PDF.
# Зачем отдельный файл, а не проверка в Qdrant?
# Потому что проверять «есть ли чанки из этого PDF в базе» —
# дорогой запрос. А файл со списком — одна строка кода.
INDEXED_FILES_PATH = PROCESSED_DIR / "indexed_files.json"


def get_indexed_files() -> set:
    """Какие файлы уже проиндексированы."""
    if INDEXED_FILES_PATH.exists():
        with open(INDEXED_FILES_PATH, "r") as f:
            return set(json.load(f))
    return set()


def save_indexed_files(files: set):
    """Сохраняем обновлённый список проиндексированных файлов."""
    INDEXED_FILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEXED_FILES_PATH, "w") as f:
        json.dump(sorted(files), f, indent=2)


def get_next_id(client: QdrantClient) -> int:
    """
    Определяем следующий свободный ID для новых точек в Qdrant.

    Почему не можно просто использовать len(points)?
    Потому что если ты удалишь точки из середины,
    len не покажет максимальный ID. count_points — надёжнее.
    """
    info = client.get_collection(COLLECTION_NAME)
    return info.points_count


def main():
    print("=" * 50)
    print("Обновление базы отчётов ЦБ РФ")
    print("=" * 50)

    # === Шаг 1: скачиваем новые PDF ===
    print("\n1. Проверяю новые отчёты на сайте ЦБ...")
    pdf_links = get_pdf_links()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    new_files = []
    for item in pdf_links:
        filepath = RAW_DATA_DIR / item["filename"]
        if not filepath.exists():
            time.sleep(2)
            if download_pdf(item["url"], filepath):
                new_files.append(filepath)

    if not new_files:
        print("\nНовых отчётов нет. База актуальна.")
        return

    print(f"\nСкачано {len(new_files)} новых отчётов")

    # === Шаг 2: чанкуем только новые файлы ===
    print("\n2. Чанкую новые файлы...")
    new_chunks = []
    for pdf_path in new_files:
        chunks = chunk_pdf(pdf_path)
        new_chunks.extend(chunks)
        print(f"  {pdf_path.name}: {len(chunks)} чанков")

    print(f"  Итого новых чанков: {len(new_chunks)}")

    # === Шаг 3: эмбеддим и добавляем в Qdrant ===
    print("\n3. Индексирую новые чанки...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(path=str(QDRANT_PATH))

    # Определяем стартовый ID для новых точек
    start_id = get_next_id(client)

    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i : i + BATCH_SIZE]
        texts = [chunk["text"] for chunk in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

        points = []
        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            point = PointStruct(
                id=start_id + i + j,
                vector=embedding.tolist(),
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

        client.upsert(collection_name=COLLECTION_NAME, points=points)

    # === Шаг 4: обновляем список проиндексированных файлов ===
    indexed = get_indexed_files()
    for f in new_files:
        indexed.add(f.name)
    save_indexed_files(indexed)

    # Итого
    total = client.get_collection(COLLECTION_NAME).points_count
    print(f"\nГотово!")
    print(f"  Добавлено: {len(new_chunks)} чанков из {len(new_files)} отчётов")
    print(f"  Всего в базе: {total} чанков")
    print(f"  Новые отчёты: {', '.join(f.name for f in new_files)}")


if __name__ == "__main__":
    main()
