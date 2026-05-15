"""
Скрейпер для скачивания ежемесячных отчётов ЦБ РФ
«О развитии банковского сектора Российской Федерации».

Как работает:
1. Заходим на страницу аналитики ЦБ
2. Парсим HTML — ищем все ссылки на PDF с паттерном "razv_bs"
3. Скачиваем каждый PDF в data/raw/

Использование:
    python src/ingestion/scraper_cbr.py
"""

import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Базовый URL сайта ЦБ.
# Ссылки на PDF в HTML — относительные (/Collection/Collection/File/...),
# поэтому нужен базовый URL, чтобы склеить полный адрес.
BASE_URL = "https://cbr.ru"

# Страница, где живут ссылки на отчёты о развитии банковского сектора.
REPORTS_PAGE = "https://cbr.ru/analytics/bank_sector/develop/"

# Куда сохраняем скачанные PDF.
# Path — удобная обёртка над строками-путями (часть стандартной библиотеки).
# __file__ — путь к текущему скрипту, .parents[2] — два уровня вверх
# (из src/ingestion/ в корень проекта).
RAW_DATA_DIR = Path(__file__).parents[2] / "data" / "raw"

# Паттерн имени файла: razv_bs_25_01.pdf, razv_bs_26_02.pdf и т.д.
# Регулярное выражение:
#   razv_bs_    — фиксированный префикс
#   \d{2}       — две цифры (год)
#   _           — подчёркивание
#   \d{2}       — две цифры (месяц)
#   \.pdf       — расширение (точка экранирована, т.к. в regex . = любой символ)
FILENAME_PATTERN = re.compile(r"razv_bs_\d{2}_\d{2}\.pdf")

# Session — объект, который хранит cookies и заголовки между запросами.
# Как браузер: зашла на страницу, получила cookie, дальше все запросы
# идут с этим cookie. Без этого ЦБ возвращает 403 на скачивание PDF.
session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    # Referer — говорит серверу «я пришла с вашей страницы».
    # Без него сервер думает, что запрос пришёл из ниоткуда и блокирует.
    "Referer": "https://cbr.ru/analytics/bank_sector/develop/",
})


def get_pdf_links() -> list[dict]:
    """
    Заходим на страницу ЦБ и собираем все ссылки на PDF отчётов.

    Возвращает список словарей: [{"url": "...", "filename": "..."}, ...]
    """
    print(f"Загружаю страницу: {REPORTS_PAGE}")

    # Используем session вместо requests.get — чтобы сохранить cookies.
    response = session.get(REPORTS_PAGE, timeout=30)

    # raise_for_status() выбросит исключение, если сервер вернул ошибку
    # (404, 500 и т.д.). Без этой строки код продолжит работать с пустым
    # или ошибочным ответом — и ты будешь долго искать, почему нет PDF.
    response.raise_for_status()

    # BeautifulSoup — библиотека для парсинга HTML.
    # "html.parser" — встроенный парсер Python (не нужно ставить lxml).
    soup = BeautifulSoup(response.text, "html.parser")

    # Ищем все теги <a> (ссылки), у которых атрибут href содержит
    # паттерн нашего файла.
    pdf_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        # Ищем в href наш паттерн имени файла
        match = FILENAME_PATTERN.search(href)
        if match:
            full_url = BASE_URL + href if href.startswith("/") else href
            filename = match.group()  # например "razv_bs_25_01.pdf"
            pdf_links.append({"url": full_url, "filename": filename})

    print(f"Найдено {len(pdf_links)} отчётов")
    return pdf_links


def download_pdf(url: str, filepath: Path) -> bool:
    """
    Скачивает один PDF. Возвращает True если успешно.

    Почему отдельная функция, а не код прямо в цикле?
    1. Легче тестировать — можно вызвать с одним URL
    2. Легче добавить retry-логику потом
    3. Читаемость — основной цикл остаётся чистым
    """
    # Если файл уже скачан — пропускаем.
    # Это важно: если скрипт упал на 8-м файле из 24,
    # при повторном запуске он не будет качать первые 7 заново.
    if filepath.exists():
        print(f"  Уже есть: {filepath.name}, пропускаю")
        return True

    try:
        print(f"  Скачиваю: {filepath.name}...", end=" ")
        response = session.get(url, timeout=60)
        response.raise_for_status()

        # "wb" = write binary. PDF — бинарный файл, не текстовый.
        # Если открыть в режиме "w" (текстовый) — файл будет битым.
        filepath.write_bytes(response.content)

        # Размер в мегабайтах для наглядности
        size_mb = len(response.content) / (1024 * 1024)
        print(f"OK ({size_mb:.1f} MB)")
        return True

    except requests.RequestException as e:
        print(f"ОШИБКА: {e}")
        return False


def main():
    """Основная функция: собираем ссылки → скачиваем PDF."""

    # Создаём папку для данных, если её нет.
    # parents=True — создаёт все промежуточные папки.
    # exist_ok=True — не падает, если папка уже существует.
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Шаг 1: собираем ссылки
    pdf_links = get_pdf_links()

    if not pdf_links:
        print("Не нашёл ни одной ссылки на PDF. Возможно, ЦБ изменил структуру страницы.")
        return

    # Шаг 2: скачиваем
    success = 0
    failed = 0

    for item in pdf_links:
        filepath = RAW_DATA_DIR / item["filename"]
        # Пауза между запросами — уважаем сервер ЦБ.
        # Без паузы можно получить бан по IP.
        time.sleep(2)

        if download_pdf(item["url"], filepath):
            success += 1
        else:
            failed += 1

    # Итого
    print(f"\nГотово! Скачано: {success}, ошибок: {failed}")
    print(f"Файлы лежат в: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
