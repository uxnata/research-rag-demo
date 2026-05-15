"""
RAG-пайплайн: вопрос → поиск чанков → GigaChat → ответ с цитатами.

Как работает:
1. Пользователь задаёт вопрос
2. Превращаем вопрос в вектор (тот же эмбеддер, что при индексации)
3. Ищем в Qdrant 5 самых похожих чанков
4. Собираем промпт: системная инструкция + чанки + вопрос
5. Отправляем в GigaChat
6. Получаем ответ с указанием источников

Использование:
    python src/retrieval/rag_pipeline.py "Как менялась ключевая ставка?"
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

# Загружаем .env
load_dotenv()

# Пути
QDRANT_PATH = Path(__file__).parents[2] / "data" / "qdrant_db"
COLLECTION_NAME = "cbr_reports"

# Та же модель, что при индексации — ОБЯЗАТЕЛЬНО.
# Если эмбеддить вопрос другой моделью, чем документы —
# вектора будут в разных пространствах и поиск сломается.
EMBEDDING_MODEL = "sergeyzh/LaBSE-ru-turbo"

# Сколько чанков отдавать в LLM после re-ranking.
TOP_K = 5

# Сколько чанков доставать из Qdrant ДО re-ranking.
# Берём с запасом — re-ranker потом отберёт лучшие TOP_K.
# 20 — хороший баланс: достаточно кандидатов, но re-ranker
# не тормозит (20 пар обрабатываются за ~1 сек).
RETRIEVAL_TOP_K = 20

# Re-ranker модель. Cross-encoder: берёт пару (вопрос, чанк)
# и выдаёт score релевантности. Точнее, чем cosine similarity,
# но медленнее — поэтому используем двухэтапную схему.
# cross-encoder/ms-marco-MiniLM-L-6-v2 — ~80 MB, лёгкая альтернатива при ограниченном диске.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Порог релевантности. Если лучший результат ниже этого score —
# значит в базе нет ничего по теме, и нет смысла тратить токены
# на GigaChat. Лучше честно сказать «нет данных».
# 0.5 — эмпирически подобранный порог для LaBSE-ru-turbo.
MIN_SCORE = 0.5

# Сколько раз пытаться вызвать GigaChat при ошибке.
MAX_RETRIES = 2
RETRY_DELAY = 3  # секунды между попытками

# Системный промпт — инструкция для GigaChat.
# Это САМАЯ важная часть RAG после retrieval.
# Плохой промпт = модель игнорирует контексты и галлюцинирует.
SYSTEM_PROMPT = """Ты — аналитик-исследователь Банка России. Твоя задача — отвечать на вопросы СТРОГО на основе предоставленных фрагментов отчётов ЦБ РФ.

Правила:
1. Отвечай ТОЛЬКО на основе предоставленных фрагментов. Не используй свои знания.
2. Для каждого факта указывай источник в формате [Источник: файл, стр. N].
3. Если в фрагментах нет информации для ответа — честно скажи об этом.
4. Используй точные цифры из фрагментов, не округляй.
5. Отвечай на русском языке, профессионально, но понятно.
6. Если данные из разных месяцев противоречат друг другу — это нормально, укажи динамику."""


def build_context(results) -> str:
    """
    Собирает найденные чанки в текстовый блок для промпта.

    Каждый чанк оборачиваем в понятную разметку, чтобы модель
    видела границы между фрагментами и знала, откуда каждый взят.
    """
    context_parts = []
    for i, result in enumerate(results, 1):
        payload = result.payload
        header = (
            f"--- Фрагмент {i} ---\n"
            f"Источник: {payload['source']}, стр. {payload['page']}\n"
            f"Период: {payload['month_name']} {payload['year']}\n"
        )
        context_parts.append(header + payload["text"])

    return "\n\n".join(context_parts)


def rerank(question: str, results, reranker) -> list:
    """
    Переранжирует результаты Qdrant с помощью cross-encoder.

    Как работает:
    1. Формируем пары (вопрос, текст_чанка) для каждого результата
    2. Cross-encoder оценивает каждую пару — насколько текст
       отвечает на вопрос (не просто «похож», а именно «отвечает»)
    3. Сортируем по новому score, берём лучшие TOP_K

    Почему это лучше cosine similarity:
    - Cosine сравнивает вектора по отдельности (bi-encoder)
    - Cross-encoder читает оба текста ВМЕСТЕ, видит связи между словами
    - «ключевая ставка» в вопросе + «решение по ставке» в тексте →
      cross-encoder понимает связь, cosine может пропустить
    """
    if not results:
        return results

    # Формируем пары для cross-encoder
    pairs = [(question, r.payload["text"]) for r in results]

    # Получаем scores — чем выше, тем релевантнее
    scores = reranker.predict(pairs)

    # Присваиваем новые scores результатам и сортируем
    scored_results = list(zip(results, scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)

    # Обновляем score в результатах и берём лучшие TOP_K
    reranked = []
    for result, new_score in scored_results[:TOP_K]:
        result.score = float(new_score)
        reranked.append(result)

    return reranked


def format_fallback_chunks(results) -> str:
    """
    Фолбек: если GigaChat недоступен, показываем сырые чанки.
    Лучше, чем ничего — пользователь хотя бы видит найденные фрагменты.
    """
    parts = []
    for i, r in enumerate(results, 1):
        p = r.payload
        parts.append(
            f"[{i}] {p['source']}, стр. {p['page']} "
            f"({p['month_name']} {p['year']}):\n"
            f"{p['text'][:300]}..."
        )
    return "\n\n".join(parts)


def expand_query(question: str, giga) -> list[str]:
    """
    Расширение запроса: просим GigaChat переформулировать вопрос
    в терминах, которые используются в отчётах ЦБ.

    Пример:
    «Каков ROE?» → [«доходность на капитал», «рентабельность капитала»]

    Возвращает список из 2-3 альтернативных формулировок.
    """
    try:
        payload = Chat(
            messages=[
                Messages(
                    role=MessagesRole.USER,
                    content=(
                        f"Переформулируй вопрос 2-3 разными способами, "
                        f"используя синонимы и термины из банковской отчётности ЦБ РФ. "
                        f"Ответь ТОЛЬКО списком формулировок, по одной на строку, без нумерации.\n\n"
                        f"Вопрос: {question}"
                    ),
                )
            ],
            temperature=0.3,
            max_tokens=200,
        )
        response = giga.chat(payload)
        expansions = response.choices[0].message.content.strip().split("\n")
        # Чистим и берём непустые строки
        return [e.strip().lstrip("—-•123456789. ") for e in expansions if e.strip()][:3]
    except Exception:
        return []


def rag_query(question: str, model, client, giga, reranker=None) -> dict:
    """
    Основная функция RAG с тремя фолбек-сценариями:

    1. Score < MIN_SCORE → «Нет данных по теме»
    2. GigaChat упал → retry, потом показываем сырые чанки
    3. Нормальный сценарий → ответ с цитатами

    Если передан reranker — двухэтапная схема:
    Qdrant (20 кандидатов) → re-ranker (лучшие 5) → GigaChat
    """
    # Шаг 0: расширяем запрос (query expansion)
    # Ищем не только по оригинальному вопросу, но и по синонимам.
    # Это помогает найти чанки, где используется другая терминология.
    queries = [question] + expand_query(question, giga)

    # Шаг 1: ищем по всем формулировкам и объединяем результаты
    fetch_k = RETRIEVAL_TOP_K if reranker else TOP_K
    all_results = {}  # id → result, чтобы убрать дубли

    for q in queries:
        query_vector = model.encode(q).tolist()
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=fetch_k,
        ).points
        for r in results:
            if r.id not in all_results or r.score > all_results[r.id].score:
                all_results[r.id] = r

    # Сортируем по score и берём top
    results = sorted(all_results.values(), key=lambda r: r.score, reverse=True)
    results = results[:fetch_k]

    # === ФОЛБЕК 1: проверяем релевантность ===
    # Проверяем ДО re-ranking, потому что cosine similarity (0-1)
    # и cross-encoder score (любые числа) — разные шкалы.
    best_score = results[0].score if results else 0

    if best_score < MIN_SCORE:
        return {
            "answer": (
                f"К сожалению, в отчётах ЦБ РФ не нашлось информации "
                f"по вашему запросу. Лучший результат поиска имеет "
                f"релевантность {best_score:.2f} (порог: {MIN_SCORE}).\n\n"
                f"Попробуйте переформулировать вопрос или задать вопрос "
                f"о банковском секторе, кредитовании, ставках или прибыли банков."
            ),
            "sources": [],
            "tokens_used": 0,
            "fallback": "low_score",
        }

    # Шаг 2.5: re-ranking (если есть re-ranker)
    # Делаем ПОСЛЕ проверки score — нет смысла ранжировать мусор.
    if reranker and results:
        results = rerank(question, results, reranker)

    # Шаг 3: собираем контекст
    context = build_context(results)

    # Шаг 4: формируем промпт
    user_message = (
        f"Вот фрагменты отчётов ЦБ РФ:\n\n{context}\n\n"
        f"---\n\n"
        f"Вопрос: {question}\n\n"
        f"Ответь на основе приведённых фрагментов, указывая источники."
    )

    payload = Chat(
        messages=[
            Messages(role=MessagesRole.SYSTEM, content=SYSTEM_PROMPT),
            Messages(role=MessagesRole.USER, content=user_message),
        ],
        temperature=0.1,
        max_tokens=1500,
    )

    # === ФОЛБЕК 2: retry при ошибке GigaChat ===
    sources = [
        {
            "file": r.payload["source"],
            "page": r.payload["page"],
            "period": f"{r.payload['month_name']} {r.payload['year']}",
            "score": round(r.score, 4),
        }
        for r in results
    ]

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = giga.chat(payload)
            answer = response.choices[0].message.content

            return {
                "answer": answer,
                "sources": sources,
                "tokens_used": response.usage.total_tokens,
                "fallback": None,
            }

        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"  GigaChat ошибка: {e}. "
                      f"Повтор через {RETRY_DELAY} сек...")
                time.sleep(RETRY_DELAY)
            else:
                # Все попытки исчерпаны — показываем сырые чанки
                fallback_text = (
                    f"GigaChat временно недоступен ({e}).\n"
                    f"Вот найденные фрагменты отчётов:\n\n"
                    f"{format_fallback_chunks(results)}"
                )
                return {
                    "answer": fallback_text,
                    "sources": sources,
                    "tokens_used": 0,
                    "fallback": "gigachat_down",
                }


def main():
    # Если вопрос передан через командную строку — используем его.
    # Иначе — интерактивный режим (задаём вопросы в цикле).
    question_from_args = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None

    # Инициализация (один раз)
    print("Загружаю модель эмбеддингов...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Загружаю re-ranker...")
    reranker = CrossEncoder(RERANKER_MODEL)

    print("Подключаюсь к Qdrant...")
    client = QdrantClient(path=str(QDRANT_PATH))

    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        raise ValueError("Нет GIGACHAT_CREDENTIALS в .env")
    giga = GigaChat(credentials=credentials, verify_ssl_certs=False)

    print("Готов!\n")

    if question_from_args:
        # Режим одного вопроса
        result = rag_query(question_from_args, model, client, giga, reranker)
        print(f"Вопрос: {question_from_args}\n")
        print(f"Ответ:\n{result['answer']}\n")
        print(f"Использовано токенов: {result['tokens_used']}")
        print(f"\nИсточники:")
        for s in result["sources"]:
            print(f"  {s['file']}, стр. {s['page']} "
                  f"({s['period']}, score: {s['score']})")
    else:
        # Интерактивный режим
        print("=" * 50)
        print("RAG-ассистент по отчётам ЦБ РФ")
        print("Задавай вопросы. 'выход' — для завершения.")
        print("=" * 50)

        while True:
            print()
            question = input("Вопрос: ").strip()
            if question.lower() in ("выход", "exit", "quit", "q"):
                print("Пока!")
                break
            if not question:
                continue

            result = rag_query(question, model, client, giga, reranker)
            print(f"\nОтвет:\n{result['answer']}")
            if result.get("fallback"):
                print(f"\n[Фолбек: {result['fallback']}]")
            else:
                print(f"\n[Токенов: {result['tokens_used']}]")
                print(f"[Источники: {', '.join(s['file'] + ' стр.' + str(s['page']) for s in result['sources'])}]")


if __name__ == "__main__":
    main()
