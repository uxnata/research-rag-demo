"""
Чанкер для PDF-отчётов ЦБ РФ.

Стратегия:
- Один чанк = одна страница (каждая страница ЦБ = одна тема)
- Фильтруем мусор от графиков (строки без русского текста)
- Приклеиваем метаданные: дата отчёта, номер страницы
- Пропускаем титульную страницу и оглавление

Использование:
    python src/ingestion/chunker.py
"""

import re
import json
from pathlib import Path
from pypdf import PdfReader

# Пути
RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

# Минимальное количество русских букв в строке, чтобы считать её полезной.
# Почему 3, а не 1? Бывает одна случайная кириллическая буква в мусоре.
MIN_RUSSIAN_CHARS = 3

# Минимальная длина чанка в символах после очистки.
# Если на странице осталось меньше — значит там только график, пропускаем.
MIN_CHUNK_LENGTH = 100

# Регулярка: ищем русские буквы (а-яА-ЯёЁ).
RUSSIAN_LETTERS = re.compile(r"[а-яА-ЯёЁ]")

# Паттерн для извлечения даты из имени файла.
FILENAME_DATE = re.compile(r"razv_bs_(\d{2})_(\d{2})\.pdf")

# Строки-мусор: только те, что ТОЧНО не содержат полезной информации.
# Консервативный подход — лучше оставить немного мусора,
# чем выкинуть полезные данные.
JUNK_PATTERNS = [
    "Сноски для текста",
    "Сноски для таблиц",
    "Источник: ф.о.",
    "Источники: ф.о.",
]

# Названия месяцев для человекочитаемых метаданных.
MONTHS = {
    "01": "январь", "02": "февраль", "03": "март",
    "04": "апрель", "05": "май", "06": "июнь",
    "07": "июль", "08": "август", "09": "сентябрь",
    "10": "октябрь", "11": "ноябрь", "12": "декабрь",
}


def is_junk_line(line: str) -> bool:
    """
    Проверяет, является ли строка известным мусорным паттерном.
    """
    stripped = line.strip()
    for pattern in JUNK_PATTERNS:
        if stripped == pattern or stripped.startswith(pattern):
            return True
    return False


def is_useful_line(line: str) -> bool:
    """
    Определяет, полезная ли строка или мусор от графика.

    Правила:
    1. Известный мусорный паттерн → мусор
    2. Начинается с • (буллет) → полезная
    3. Есть русские буквы (минимум 3) → полезная
    4. Всё остальное → мусор
    """
    stripped = line.strip()

    if not stripped:
        return False

    if is_junk_line(stripped):
        return False

    if stripped.startswith("•"):
        return True

    russian_count = len(RUSSIAN_LETTERS.findall(stripped))
    return russian_count >= MIN_RUSSIAN_CHARS


def is_junk_page(raw_text: str) -> bool:
    """
    Определяет, является ли страница мусорной:
    - Примечания (сноски к формам отчетности)
    - Глоссарий (список аббревиатур: ББЛ, БУЛ, ДКП...)
    - Оглавление
    """
    first_500 = raw_text[:500]
    markers = [
        "Примечания (",
        "В подобного рода формах",
        "Банк с базовой лицензией",
        "Банк с универсальной лицензией",
        "Денежно-кредитная политика",
    ]
    return any(m in first_500 for m in markers)


def clean_page_text(raw_text: str) -> str:
    """
    Берёт сырой текст страницы из pypdf,
    убирает мусорные строки, возвращает чистый текст.
    """
    lines = raw_text.split("\n")
    useful_lines = [line.strip() for line in lines if is_useful_line(line)]
    return "\n".join(useful_lines)


def extract_date_from_filename(filename: str) -> dict:
    """
    Извлекает дату отчёта из имени файла.

    razv_bs_26_03.pdf → {"year": 2026, "month": 3, "month_name": "март"}
    """
    match = FILENAME_DATE.search(filename)
    if not match:
        return {"year": 0, "month": 0, "month_name": "неизвестно"}

    year_short, month = match.groups()
    year = 2000 + int(year_short)
    return {
        "year": year,
        "month": int(month),
        "month_name": MONTHS.get(month, "неизвестно"),
    }


def chunk_pdf(pdf_path: Path) -> list[dict]:
    """
    Основная функция: берёт PDF, возвращает список чанков.

    Каждый чанк — словарь:
    {
        "text": "...",           # очищенный текст
        "metadata": {
            "source": "razv_bs_26_03.pdf",
            "page": 3,
            "year": 2026,
            "month": 3,
            "month_name": "март",
            "report_type": "О развитии банковского сектора РФ"
        }
    }
    """
    reader = PdfReader(pdf_path)
    date_info = extract_date_from_filename(pdf_path.name)
    chunks = []

    for page_num, page in enumerate(reader.pages):
        # Пропускаем первые 2 страницы (титулка + оглавление)
        if page_num < 2:
            continue

        raw_text = page.extract_text() or ""

        # Пропускаем мусорные страницы (примечания, глоссарий)
        if is_junk_page(raw_text):
            continue

        clean_text = clean_page_text(raw_text)

        # Если после очистки осталось слишком мало — пропускаем.
        # Значит на странице был только график без текста.
        if len(clean_text) < MIN_CHUNK_LENGTH:
            continue

        chunk = {
            "text": clean_text,
            "metadata": {
                "source": pdf_path.name,
                "page": page_num + 1,  # люди считают с 1, не с 0
                "year": date_info["year"],
                "month": date_info["month"],
                "month_name": date_info["month_name"],
                "report_type": "О развитии банковского сектора РФ",
            },
        }
        chunks.append(chunk)

    return chunks


def main():
    """Обрабатываем все PDF из data/raw/, сохраняем чанки в JSON."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Собираем все PDF-файлы, сортируем по имени (= по дате)
    pdf_files = sorted(RAW_DIR.glob("razv_bs_*.pdf"))

    if not pdf_files:
        print("Нет PDF-файлов в data/raw/. Сначала запусти scraper_cbr.py")
        return

    print(f"Найдено {len(pdf_files)} PDF-файлов")

    all_chunks = []
    for pdf_path in pdf_files:
        chunks = chunk_pdf(pdf_path)
        all_chunks.extend(chunks)
        print(f"  {pdf_path.name}: {len(chunks)} чанков")

    # Сохраняем все чанки в один JSON.
    # ensure_ascii=False — чтобы русские буквы были читаемыми,
    # а не \u0410\u0411\u0412...
    output_path = PROCESSED_DIR / "chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nИтого: {len(all_chunks)} чанков")
    print(f"Сохранено в: {output_path}")

    # Покажем пример чанка — чтобы глазами проверить качество
    if all_chunks:
        print("\n=== ПРИМЕР ЧАНКА ===")
        example = all_chunks[0]
        print(f"Источник: {example['metadata']['source']}, "
              f"стр. {example['metadata']['page']}")
        print(f"Период: {example['metadata']['month_name']} "
              f"{example['metadata']['year']}")
        print(f"Текст ({len(example['text'])} символов):")
        print(example["text"][:500])


if __name__ == "__main__":
    main()
