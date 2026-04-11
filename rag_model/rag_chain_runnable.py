import os
import re
import sys
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

try:
    from rag_model.vector_db import load_vector_store
except ModuleNotFoundError:
    from vector_db import load_vector_store

if os.name == "nt" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables
load_dotenv()

# -----------------------------
# 🔹 Initialize LLM
# -----------------------------
llm = init_chat_model(
    "openai/gpt-oss-120b",
     model_provider="groq"
)

# -----------------------------
# 🔹 Prompt Template
# -----------------------------
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent AI Finance Assistant. Your job is to analyze
expense data from uploaded PDF documents and answer user questions.

Rules:
1. Only use the information from the provided context.
2. Do NOT use your own knowledge or make assumptions.
3. If the answer is not in the context, respond:
   "This information is not available in the uploaded documents."
4. For financial questions, be precise with numbers and dates.
5. When relevant, also give a short financial suggestion or insight.
"""),
    ("human", """
Context:
{context}

Question:
{question}
""")
])

TOTAL_EXPENSE_TRIGGERS = (
    "total expense",
    "total expenses",
    "total spend",
    "overall expense",
    "overall spend",
)

TOTAL_PATTERNS = (
    r"invoice total\s*[:\-]?\s*(\d[\d,]*\.\d{2})",
    r"grand total\s*[:\-]?\s*(\d[\d,]*\.\d{2})",
    r"total amount\s*[:\-]?\s*(\d[\d,]*\.\d{2})",
)


def _looks_unreadable_text(text: str) -> bool:
    """Heuristic to skip extracted text that is mostly corrupted."""
    if not text:
        return True

    replacement_ratio = text.count("�") / max(len(text), 1)
    alnum_count = sum(char.isalnum() for char in text)
    return replacement_ratio > 0.10 or alnum_count < 20


def _dedupe_documents(docs: list) -> list:
    """Drop unreadable and duplicate chunks before prompt construction."""
    deduped_docs = []
    seen = set()

    for doc in docs:
        content = doc.page_content.strip()
        if _looks_unreadable_text(content):
            continue

        normalized = " ".join(content.split())[:500]
        if normalized in seen:
            continue

        seen.add(normalized)
        deduped_docs.append(doc)

    return deduped_docs


def _extract_total_amount(text: str) -> float | None:
    """Extract a likely invoice total from text when present."""
    normalized_text = " ".join(text.split()).lower()

    for pattern in TOTAL_PATTERNS:
        match = re.search(pattern, normalized_text)
        if match:
            return float(match.group(1).replace(",", ""))

    return None


def _try_answer_total_expenses(question: str, vector_store) -> str | None:
    """Answer simple total-expense questions deterministically from saved docs."""
    normalized_question = question.lower()
    if not any(trigger in normalized_question for trigger in TOTAL_EXPENSE_TRIGGERS):
        return None

    raw_docs = getattr(vector_store.docstore, "_dict", {}).values()
    docs = _dedupe_documents(list(raw_docs))

    totals = []
    sources = []

    for doc in docs:
        amount = _extract_total_amount(doc.page_content)
        if amount is None:
            continue

        totals.append(amount)
        sources.append(doc.metadata.get("source", "document"))

    if not totals:
        return None

    total_expense = sum(totals)
    unique_sources = sorted(set(sources))

    if len(unique_sources) == 1:
        return (
            f"Total expenses: Rs. {total_expense:.2f}\n\n"
            f"Source: {unique_sources[0]}"
        )

    return (
        f"Total expenses across {len(unique_sources)} readable document(s): "
        f"Rs. {total_expense:.2f}\n\n"
        f"Sources: {', '.join(unique_sources)}"
    )

# -----------------------------
# 🔹 RAG Function
# -----------------------------
def get_rag_response(question: str) -> str:
    """
    Takes a user question, retrieves relevant chunks from FAISS,
    and returns an LLM-generated answer.
    """

    # Load vector store
    vector_store = load_vector_store()

    direct_total_answer = _try_answer_total_expenses(question, vector_store)
    if direct_total_answer:
        return direct_total_answer

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )

    # Retrieve relevant documents
    docs = _dedupe_documents(retriever.invoke(question))

    if not docs:
        return "This information is not available in the uploaded documents."

    # Convert docs to context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create prompt
    prompt = PROMPT_TEMPLATE.invoke({
        "context": context,
        "question": question
    })

    # Call LLM
    response = llm.invoke(prompt)

    return response.content


# -----------------------------
# 🔹 MAIN EXECUTION (IMPORTANT)
# -----------------------------
if __name__ == "__main__":
    print("📊 AI Finance Assistant Ready!")

    while True:
        question = input("\nAsk your question (type 'exit' to quit): ")

        if question.lower() == "exit":
            print("Goodbye 👋")
            break

        answer = get_rag_response(question)
        print("\n💡 Answer:\n", answer)
