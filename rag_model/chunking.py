import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def _legacy_load_pdfs_from_folder(folder_path:str= "D:\Bills") -> List[Document]:
    """
    Load all PDFs from a folder and split into chunks.

    Args:
        folder_path (str): Path to folder containing PDFs

    Returns:
        List[Document]: All chunks from all PDFs
    """

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    all_chunks = []

    # 🔹 Loop through all files in folder
    for file_name in os.listdir(folder_path):

        if file_name.endswith(".pdf"):
            full_path = os.path.join(folder_path, file_name)

            print(f"[INFO] Processing: {file_name}")

            # 🔹 Load PDF
            loader = PyPDFLoader(full_path)
            pages = loader.load()

            # 🔹 Clean text
            for page in pages:
                page.page_content = page.page_content.replace("\n", " ").strip()

            # 🔹 Split into chunks
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )

            chunks = splitter.split_documents(pages)

            # 🔹 Add metadata (important)
            for chunk in chunks:
                chunk.metadata["source"] = file_name

            print(f"[INFO] {file_name} → {len(chunks)} chunks")

            all_chunks.extend(chunks)

    print(f"[INFO] Total chunks from all PDFs: {len(all_chunks)}")

    return all_chunks


def _looks_unreadable(text: str) -> bool:
    """Heuristic to skip PDFs whose extracted text is mostly corrupted."""
    if not text:
        return True

    replacement_ratio = text.count("�") / max(len(text), 1)
    alnum_count = sum(char.isalnum() for char in text)

    return replacement_ratio > 0.10 or alnum_count < 20


def load_and_chunk_pdf(pdf_path: str, source_name: str | None = None) -> List[Document]:
    """
    Load a single PDF file and split it into chunks.

    Args:
        pdf_path (str): Path to a single PDF file.
        source_name (str | None): Optional display name to store in metadata.

    Returns:
        List[Document]: All chunks from the PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

    file_name = source_name or os.path.basename(pdf_path)
    print(f"[INFO] Processing: {file_name}")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    cleaned_pages = []
    unreadable_pages = 0

    for page in pages:
        page.page_content = page.page_content.replace("\n", " ").strip()
        if _looks_unreadable(page.page_content):
            unreadable_pages += 1
            continue
        cleaned_pages.append(page)

    if not cleaned_pages:
        print(
            f"[WARNING] Skipping '{file_name}' because the extracted PDF text "
            "looks unreadable. Try an OCR/text-searchable PDF."
        )
        return []

    if unreadable_pages:
        print(
            f"[WARNING] Skipped {unreadable_pages} unreadable page(s) in "
            f"'{file_name}'."
        )

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(cleaned_pages)

    for chunk in chunks:
        chunk.metadata["source"] = file_name

    print(f"[INFO] {file_name} -> {len(chunks)} chunks")
    return chunks


def load_pdfs_from_folder(folder_path: str = r"D:\Bills") -> List[Document]:
    """
    Load all PDFs from a folder and split them into chunks.

    Args:
        folder_path (str): Path to folder containing PDFs.

    Returns:
        List[Document]: All chunks from all PDFs.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    all_chunks = []

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(".pdf"):
            continue

        full_path = os.path.join(folder_path, file_name)
        chunks = load_and_chunk_pdf(full_path, source_name=file_name)
        all_chunks.extend(chunks)

    print(f"[INFO] Total chunks from all PDFs: {len(all_chunks)}")
    return all_chunks
