import gc
import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from rag_model.network_config import sanitize_dead_local_proxies
except ModuleNotFoundError:
    from network_config import sanitize_dead_local_proxies

# ── Embedding model (runs locally, no API key needed) ────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"   # folder where FAISS saves the index


def get_embeddings():
    """Returns the HuggingFace embedding model."""
    sanitize_dead_local_proxies()

    model_kwargs = {}
    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        model_kwargs["token"] = hf_token

    if _is_model_cached_locally():
        model_kwargs["local_files_only"] = True

    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
        )
    except Exception as exc:
        # If the model is already cached locally, retry in offline mode so
        # loading the vector store does not depend on a fresh network check.
        if _is_huggingface_connection_error(exc):
            return HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={**model_kwargs, "local_files_only": True},
            )
        raise


def _is_huggingface_connection_error(error: Exception) -> bool:
    """Detect common Hugging Face connectivity failures worth retrying offline."""
    message = str(error).lower()
    network_markers = (
        "connection refused",
        "actively refused it",
        "client has been closed",
        "failed to establish a new connection",
        "name or service not known",
        "temporary failure in name resolution",
        "max retries exceeded",
    )
    return any(marker in message for marker in network_markers)


def _is_model_cached_locally() -> bool:
    """Check whether the embedding model already exists in the local HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False

    cached_path = try_to_load_from_cache(
        repo_id=EMBEDDING_MODEL,
        filename="modules.json",
    )
    return isinstance(cached_path, str)


def vector_store_exists() -> bool:
    """Return True when a saved FAISS index is available on disk."""
    index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    store_file = os.path.join(FAISS_INDEX_PATH, "index.pkl")
    return os.path.exists(index_file) and os.path.exists(store_file)


def clear_vector_store() -> bool:
    """Delete the saved FAISS index and all persisted embedding history."""
    if not os.path.exists(FAISS_INDEX_PATH):
        return False

    gc.collect()
    try:
        shutil.rmtree(FAISS_INDEX_PATH)
    except OSError as exc:
        print(f"[Vector Store] Could not clear FAISS index: {exc}")
        return False

    print(f"[Vector Store] Cleared FAISS index from '{FAISS_INDEX_PATH}'")
    return True


def get_vector_store_count() -> int:
    """Return the number of stored vectors in the saved FAISS index."""
    if not vector_store_exists():
        return 0

    vector_store = load_vector_store()
    count = vector_store.index.ntotal
    del vector_store
    gc.collect()
    return count


def build_vector_store(chunks: list) -> FAISS:
    """
    Takes document chunks, embeds them, and saves a FAISS vector store.

    Args:
        chunks (list): Output from load_and_chunk_pdf()
    Returns:
        FAISS: The vector store object (also saved to disk)
    """
    print("[Vector Store] Embedding chunks... (first run may take a moment)")

    if not chunks:
        raise ValueError("No chunks provided to build the vector store.")

    embeddings = get_embeddings()

    if vector_store_exists():
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        existing_count = vector_store.index.ntotal
        vector_store.add_documents(chunks)
        print(
            f"[Vector Store] Appended {len(chunks)} chunks to existing history "
            f"({existing_count} -> {vector_store.index.ntotal})"
        )
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)
        print(f"[Vector Store] Created new FAISS index with {len(chunks)} chunks")

    # Save to disk so we don't re-embed on every run
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"[Vector Store] Saved FAISS index to '{FAISS_INDEX_PATH}'")

    return vector_store


def load_vector_store() -> FAISS:
    """
    Loads a previously saved FAISS vector store from disk.

    Returns:
        FAISS: The loaded vector store object
    """
    if not vector_store_exists():
        raise FileNotFoundError(
            "No FAISS index found. Please upload and process a PDF first."
        )

    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True   # required by LangChain for local load
    )

    print("[Vector Store] FAISS index loaded from disk")
    return vector_store

