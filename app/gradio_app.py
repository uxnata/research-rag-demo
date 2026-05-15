"""
Gradio-интерфейс для RAG-ассистента по отчётам ЦБ РФ.

Использование:
    python app/gradio_app.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from gigachat import GigaChat

from src.retrieval.rag_pipeline import rag_query, EMBEDDING_MODEL, COLLECTION_NAME, RERANKER_MODEL

load_dotenv(PROJECT_ROOT / ".env")

QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_db"

print("Загружаю модель эмбеддингов...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

print("Загружаю re-ranker...")
reranker = CrossEncoder(RERANKER_MODEL)

print("Подключаюсь к Qdrant...")
qdrant_client = QdrantClient(path=str(QDRANT_PATH))

credentials = os.getenv("GIGACHAT_CREDENTIALS")
if not credentials:
    raise ValueError("Нет GIGACHAT_CREDENTIALS в .env")
giga_client = GigaChat(credentials=credentials, verify_ssl_certs=False)

print("Всё готово!\n")


def answer_question(message, history):
    if not message.strip():
        return "Задайте вопрос о банковском секторе РФ."

    result = rag_query(message, embed_model, qdrant_client, giga_client, reranker)
    answer = result["answer"]

    if result["sources"] and not result.get("fallback"):
        answer += "\n\n---\n**Источники:**\n"
        for s in result["sources"]:
            answer += (
                f"- {s['file']}, стр. {s['page']} "
                f"({s['period']}) — релевантность {s['score']}\n"
            )
        answer += f"\nИспользовано токенов: {result['tokens_used']}"

    return answer


demo = gr.ChatInterface(
    fn=answer_question,
    title="RAG-ассистент по отчётам ЦБ РФ",
    description=(
        "Задавайте вопросы о банковском секторе России. "
        "Ответы основаны на ежемесячных отчётах ЦБ РФ."
    ),
    examples=[
        "Какова прибыль банковского сектора?",
        "Как менялось корпоративное кредитование?",
        "Что происходит с ипотечным кредитованием?",
    ],
)

if __name__ == "__main__":
    demo.launch(server_port=7860)
