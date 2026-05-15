"""
Скрипт для проверки подключения к GigaChat API.
Запусти его первым — если видишь ответ модели, значит всё настроено.

Что происходит под капотом:
1. Читаем ключ из .env файла (не хардкодим в коде!)
2. Создаём клиент GigaChat
3. Отправляем простой вопрос
4. Печатаем ответ

Использование:
    python src/generation/test_gigachat.py
"""

import os
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

# Загружаем переменные из .env файла в переменные окружения.
# Без этой строки os.getenv("GIGACHAT_CREDENTIALS") вернёт None,
# потому что Python сам не знает про .env — это просто файл на диске.
load_dotenv()

# Достаём ключ. Если его нет — падаем с понятной ошибкой,
# а не с загадочным "NoneType has no attribute..." через 20 строк.
credentials = os.getenv("GIGACHAT_CREDENTIALS")
if not credentials:
    raise ValueError(
        "Не нашёл GIGACHAT_CREDENTIALS в .env файле. "
        "Скопируй .env.example в .env и вставь свой ключ."
    )


def test_connection():
    """Отправляем простой запрос и проверяем, что API отвечает."""

    # verify_ssl_certs=False — нужно для GigaChat,
    # у них самоподписанный сертификат на API-эндпоинте.
    # В проде так не делают, но для Сберовского API это штатная ситуация.
    giga = GigaChat(credentials=credentials, verify_ssl_certs=False)

    # Формируем запрос: системный промпт + вопрос пользователя.
    # Это та же структура, что потом будет в RAG —
    # system задаёт роль, user задаёт вопрос.
    payload = Chat(
        messages=[
            Messages(
                role=MessagesRole.SYSTEM,
                content="Ты — помощник-исследователь по финтех-рынку России."
            ),
            Messages(
                role=MessagesRole.USER,
                content="Назови три главных тренда российского финтеха в 2024 году. Кратко, по одному предложению на тренд."
            )
        ],
        temperature=0.3,  # Низкая температура = более детерминированные ответы.
                          # Для RAG обычно 0.1–0.3 — нам нужны факты, не креатив.
        max_tokens=500,
    )

    response = giga.chat(payload)

    # response.choices — список вариантов ответа (обычно один).
    # .message.content — текст ответа.
    answer = response.choices[0].message.content
    print("=== GigaChat отвечает ===")
    print(answer)
    print("=========================")
    print(f"Токенов использовано: {response.usage.total_tokens}")

    return answer


if __name__ == "__main__":
    test_connection()
