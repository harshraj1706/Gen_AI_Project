import os
import tempfile
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

try:
    from rag_model.chunking import load_and_chunk_pdf
    from rag_model.rag_chain_runnable import get_rag_response
    from rag_model.vector_db import (
        build_vector_store,
        clear_vector_store,
        get_vector_store_count,
        vector_store_exists,
    )
except ModuleNotFoundError:
    from chunking import load_and_chunk_pdf
    from rag_chain_runnable import get_rag_response
    from vector_db import (
        build_vector_store,
        clear_vector_store,
        get_vector_store_count,
        vector_store_exists,
    )


app = FastAPI(
    title="AI Finance Assistant API",
    description="Upload expense PDFs, build a FAISS index, and ask questions.",
    version="1.0.0",
)


class HealthResponse(BaseModel):
    status: str


class StoreStatusResponse(BaseModel):
    index_exists: bool
    vector_count: int


class UploadResponse(BaseModel):
    processed_files: list[str]
    skipped_files: list[str]
    errors: list[str]
    chunks_added: int
    index_exists: bool
    vector_count: int


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    question: str
    answer: str


class ClearResponse(BaseModel):
    cleared: bool
    index_exists: bool
    vector_count: int


def _format_error_message(prefix: str, error: Exception) -> str:
    return f"{prefix}: {error}"


def _store_status() -> StoreStatusResponse:
    index_exists = vector_store_exists()
    return StoreStatusResponse(
        index_exists=index_exists,
        vector_count=get_vector_store_count() if index_exists else 0,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/documents/status", response_model=StoreStatusResponse)
def documents_status() -> StoreStatusResponse:
    return _store_status()


@app.post("/documents", response_model=UploadResponse)
async def upload_documents(
    files: Annotated[list[UploadFile], File(description="One or more PDF files")],
) -> UploadResponse:
    all_chunks = []
    processed_files = []
    skipped_files = []
    errors = []

    for uploaded_file in files:
        if not uploaded_file.filename:
            skipped_files.append("unnamed file")
            continue

        if not uploaded_file.filename.lower().endswith(".pdf"):
            skipped_files.append(uploaded_file.filename)
            continue

        uploaded_file_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(await uploaded_file.read())
                uploaded_file_path = tmp_file.name

            chunks = load_and_chunk_pdf(
                uploaded_file_path,
                source_name=uploaded_file.filename,
            )
            all_chunks.extend(chunks)

            if chunks:
                processed_files.append(uploaded_file.filename)
            else:
                skipped_files.append(uploaded_file.filename)
        except Exception as exc:
            errors.append(_format_error_message(uploaded_file.filename, exc))
        finally:
            await uploaded_file.close()
            if uploaded_file_path and os.path.exists(uploaded_file_path):
                os.remove(uploaded_file_path)

    if all_chunks:
        try:
            build_vector_store(all_chunks)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=_format_error_message("Could not build the FAISS index", exc),
            ) from exc

    status = _store_status()
    return UploadResponse(
        processed_files=processed_files,
        skipped_files=skipped_files,
        errors=errors,
        chunks_added=len(all_chunks),
        index_exists=status.index_exists,
        vector_count=status.vector_count,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    if not vector_store_exists():
        raise HTTPException(
            status_code=400,
            detail="No FAISS index found. Upload and process PDFs first.",
        )

    try:
        answer = get_rag_response(question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=_format_error_message("Could not answer the question", exc),
        ) from exc

    return ChatResponse(question=question, answer=answer)


@app.delete("/documents", response_model=ClearResponse)
def clear_documents() -> ClearResponse:
    cleared = clear_vector_store()
    status = _store_status()
    return ClearResponse(
        cleared=cleared,
        index_exists=status.index_exists,
        vector_count=status.vector_count,
    )
