"""
Эвалуация RAG-системы с YandexGPT в роли судьи.

Три метрики:
1. Faithfulness — ответ подтверждён контекстом? (ловит галлюцинации)
2. Answer Relevancy — ответ отвечает на вопрос? (ловит уход от темы)
3. Context Precision — retriever нашёл правильные чанки? (ловит плохой поиск)

Использование:
    python src/evaluation/evaluate.py
"""

import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Добавляем корень проекта в путь
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.rag_pipeline import rag_query, EMBEDDING_MODEL, COLLECTION_NAME
from gigachat import GigaChat

load_dotenv()

# Пути
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_db"
TEST_SET_PATH = Path(__file__).parent / "test_set.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"

# YandexGPT настройки
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_MODEL_URI = f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest"

# Пауза между запросами к YandexGPT (чтобы не упереться в лимиты)
API_DELAY = 1


def call_yandex_judge(prompt: str) -> str:
    """
    Отправляет запрос YandexGPT-судье.
    Возвращает текстовый ответ.
    """
    response = requests.post(
        "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        headers={
            "Authorization": f"Api-Key {YANDEX_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "modelUri": YANDEX_MODEL_URI,
            "completionOptions": {"temperature": 0.1, "maxTokens": 500},
            "messages": [{"role": "user", "text": prompt}],
        },
        timeout=30,
    )

    if response.ok:
        return response.json()["result"]["alternatives"][0]["message"]["text"]
    else:
        print(f"  Ошибка YandexGPT: {response.status_code} {response.text[:200]}")
        return "ОШИБКА"


def judge_faithfulness(question: str, answer: str, context: str) -> float:
    """
    Faithfulness: подтверждён ли ответ контекстом?

    Судья получает ответ и контекст, оценивает от 0 до 1.
    1.0 = все утверждения подтверждены
    0.0 = всё выдумано
    """
    prompt = f"""Ты — эксперт-аудитор. Твоя задача — оценить, подтверждается ли ответ предоставленным контекстом.

Контекст (фрагменты отчётов ЦБ РФ):
{context[:6000]}

Ответ системы:
{answer}

ПРАВИЛА ОЦЕНКИ:
- Сравнивай СМЫСЛ, а не буквальное совпадение слов. «0,4 трлн руб.» и «0.4 трлн рублей» — это одно и то же.
- Если ответ честно говорит «данных нет» или «невозможно определить» — и в контексте действительно нет данных, это faithfulness = 1.0 (система не врёт).
- Общие выводы и обобщения допустимы, если они следуют из цифр в контексте.
- Считай только ФАКТИЧЕСКИЕ утверждения (цифры, даты, тренды). Вводные фразы типа «наблюдается тенденция» не считай.

Ответь ТОЛЬКО одним числом от 0.0 до 1.0:
- 1.0 = все фактические утверждения подтверждены контекстом
- 0.8 = почти все подтверждены, 1-2 мелких неточности
- 0.5 = примерно половина подтверждена
- 0.2 = большинство утверждений не подтверждены
- 0.0 = ответ полностью выдуман

Число:"""

    result = call_yandex_judge(prompt)
    try:
        # Извлекаем число из ответа — иногда модель добавляет пояснения
        import re
        numbers = re.findall(r"[01]\.\d+|[01]", result.strip())
        if numbers:
            score = float(numbers[0].replace(",", "."))
        else:
            score = float(result.strip().replace(",", "."))
        return max(0.0, min(1.0, score))
    except ValueError:
        print(f"  Не удалось распарсить faithfulness: '{result}'")
        return -1.0


def judge_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy: отвечает ли ответ на заданный вопрос?

    Судья оценивает, насколько ответ релевантен вопросу.
    """
    prompt = f"""Ты — эксперт-аудитор. Оцени, насколько ответ релевантен вопросу.

Вопрос: {question}

Ответ:
{answer}

ПРАВИЛА ОЦЕНКИ:
- Если ответ честно говорит «данных нет» на вопрос, по которому действительно нет данных — это relevancy = 0.8 (система правильно отказала).
- Если ответ уходит в смежную тему — это снижение.
- Оценивай, получил ли пользователь полезную информацию по своему вопросу.

Ответь ТОЛЬКО одним числом от 0.0 до 1.0:
- 1.0 = ответ полностью и точно отвечает на вопрос
- 0.8 = ответ по теме, но неполный или честный отказ при отсутствии данных
- 0.5 = ответ частично отвечает, есть отклонения
- 0.2 = ответ слабо связан с вопросом
- 0.0 = ответ совершенно не относится к вопросу

Число:"""

    result = call_yandex_judge(prompt)
    try:
        import re
        numbers = re.findall(r"[01]\.\d+|[01]", result.strip())
        if numbers:
            score = float(numbers[0].replace(",", "."))
        else:
            score = float(result.strip().replace(",", "."))
        return max(0.0, min(1.0, score))
    except ValueError:
        print(f"  Не удалось распарсить relevancy: '{result}'")
        return -1.0


def judge_context_precision(question: str, context: str, ground_truth: str) -> float:
    """
    Context Precision: содержит ли найденный контекст информацию
    для ответа на вопрос?

    Вместо сравнения имён файлов — просим YandexGPT оценить,
    насколько контекст полезен для ответа. Это надёжнее,
    потому что правильный ответ может быть в разных файлах.
    """
    if not ground_truth or ground_truth.startswith("Вопрос не относится"):
        return 1.0

    prompt = f"""Ты — эксперт-аудитор. Оцени, содержит ли предоставленный контекст достаточно информации для ответа на вопрос.

Вопрос: {question}

Эталонный ответ (что должна содержать информация):
{ground_truth}

Найденный контекст:
{context[:4000]}

Оцени, какая доля фактов из эталонного ответа ПРИСУТСТВУЕТ в контексте.
Ответь ТОЛЬКО одним числом от 0.0 до 1.0:
- 1.0 = в контексте есть ВСЯ информация для полного ответа
- 0.8 = почти вся информация есть
- 0.5 = примерно половина нужной информации
- 0.2 = очень мало полезной информации
- 0.0 = контекст совершенно не содержит нужной информации

Число:"""

    result = call_yandex_judge(prompt)
    try:
        import re
        numbers = re.findall(r"[01]\.\d+|[01]", result.strip())
        if numbers:
            score = float(numbers[0].replace(",", "."))
        else:
            score = float(result.strip().replace(",", "."))
        return max(0.0, min(1.0, score))
    except ValueError:
        print(f"  Не удалось распарсить precision: '{result}'")
        return -1.0


def build_context_from_results(results) -> str:
    """Собирает текстовый контекст из результатов Qdrant."""
    parts = []
    for r in results:
        parts.append(
            f"[{r.payload['source']}, стр. {r.payload['page']}]\n"
            f"{r.payload['text'][:1000]}"
        )
    return "\n\n".join(parts)


def main():
    print("=" * 60)
    print("Эвалуация RAG-системы")
    print("Генератор: GigaChat | Судья: YandexGPT")
    print("=" * 60)

    # Проверяем ключи
    if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
        raise ValueError("Нужны YANDEX_API_KEY и YANDEX_FOLDER_ID в .env")

    # Загружаем тест-сет
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    print(f"\nТест-сет: {len(test_set)} вопросов")

    # Инициализация
    print("Загружаю модели...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    qdrant_client = QdrantClient(path=str(QDRANT_PATH))

    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    giga = GigaChat(credentials=credentials, verify_ssl_certs=False)

    # Прогоняем каждый вопрос
    results_log = []
    total_faithfulness = []
    total_relevancy = []
    total_precision = []

    for item in test_set:
        print(f"\n--- Вопрос {item['id']}: {item['question'][:50]}...")

        # Получаем ответ от RAG
        rag_result = rag_query(item["question"], embed_model, qdrant_client, giga)

        # Получаем контекст для судьи
        query_vector = embed_model.encode(item["question"]).tolist()
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5,
        ).points
        context = build_context_from_results(search_results)

        # Источники, которые нашёл retriever
        retrieved_sources = [r.payload["source"] for r in search_results]

        # === Метрика 1: Faithfulness (YandexGPT судит) ===
        time.sleep(API_DELAY)
        faith = judge_faithfulness(
            item["question"], rag_result["answer"], context
        )
        print(f"  Faithfulness: {faith}")

        # === Метрика 2: Relevancy (YandexGPT судит) ===
        time.sleep(API_DELAY)
        relev = judge_relevancy(item["question"], rag_result["answer"])
        print(f"  Relevancy:    {relev}")

        # === Метрика 3: Context Precision (YandexGPT судит) ===
        if item.get("out_of_scope"):
            prec = -1.0  # не считаем для вопросов вне скоупа
            print(f"  Precision:    N/A (вне скоупа)")
        else:
            time.sleep(API_DELAY)
            prec = judge_context_precision(
                item["question"], context, item["ground_truth"]
            )
            print(f"  Precision:    {prec}")

        # Сохраняем
        entry = {
            "id": item["id"],
            "question": item["question"],
            "answer": rag_result["answer"],
            "ground_truth": item["ground_truth"],
            "category": item["category"],
            "faithfulness": faith,
            "relevancy": relev,
            "context_precision": prec,
            "retrieved_sources": retrieved_sources,
            "expected_sources": item["expected_sources"],
            "tokens_used": rag_result["tokens_used"],
            "fallback": rag_result.get("fallback"),
        }
        results_log.append(entry)

        if faith >= 0:
            total_faithfulness.append(faith)
        if relev >= 0:
            total_relevancy.append(relev)
        if prec >= 0:
            total_precision.append(prec)

    # === Итоговые метрики ===
    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ МЕТРИКИ")
    print("=" * 60)

    avg_faith = sum(total_faithfulness) / len(total_faithfulness) if total_faithfulness else 0
    avg_relev = sum(total_relevancy) / len(total_relevancy) if total_relevancy else 0
    avg_prec = sum(total_precision) / len(total_precision) if total_precision else 0

    print(f"  Faithfulness (верность контексту): {avg_faith:.3f}")
    print(f"  Relevancy (релевантность ответа):  {avg_relev:.3f}")
    print(f"  Context Precision (точность поиска): {avg_prec:.3f}")
    print(f"  Всего токенов GigaChat: "
          f"{sum(r['tokens_used'] for r in results_log)}")

    # Проблемные вопросы
    print(f"\n  Проблемные вопросы (faithfulness < 0.7):")
    for r in results_log:
        if 0 <= r["faithfulness"] < 0.7:
            print(f"    #{r['id']}: {r['question'][:50]}... "
                  f"(faith={r['faithfulness']})")

    # Сохраняем подробные результаты
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "avg_faithfulness": round(avg_faith, 3),
                    "avg_relevancy": round(avg_relev, 3),
                    "avg_context_precision": round(avg_prec, 3),
                    "total_questions": len(test_set),
                    "generator": "GigaChat Pro",
                    "judge": "YandexGPT",
                },
                "details": results_log,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nПодробные результаты: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
