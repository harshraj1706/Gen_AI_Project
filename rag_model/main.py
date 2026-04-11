try:
    from rag_model.chunking import load_pdfs_from_folder
except ModuleNotFoundError:
    from chunking import load_pdfs_from_folder

if __name__ == "__main__":
    chunks = load_pdfs_from_folder(r"D:\Bills")
    print(f"Total chunks: {len(chunks)}")
