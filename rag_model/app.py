import os
import tempfile
import streamlit as st
try:
    from rag_model.chunking import load_and_chunk_pdf
    from rag_model.vector_db import (
        build_vector_store,
        clear_vector_store,
        vector_store_exists,
    )
    from rag_model.rag_chain_runnable import get_rag_response
except ModuleNotFoundError:
    from chunking import load_and_chunk_pdf
    from vector_db import build_vector_store, clear_vector_store, vector_store_exists
    from rag_chain_runnable import get_rag_response

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💰 AI Finance Assistant",
    page_icon="💰",
    layout="wide"
)

st.title("💰 AI Finance Assistant")
st.caption("Upload your monthly expense PDFs and ask anything about your finances.")

# ── Session State (stores chat history) ──────────────────────────────────
history_available = vector_store_exists()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = history_available
else:
    st.session_state.pdf_processed = (
        st.session_state.pdf_processed or history_available
    )

# ── Sidebar: PDF Upload ───────────────────────────────────────────────────
with st.sidebar:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.pdf_processed:
        st.success("Saved FAISS history found. You can ask questions immediately.")
    else:
        st.caption("No saved FAISS history yet. Upload PDFs to create it.")

    clear_history_toggle = st.toggle("Enable clear history")
    if st.button(
        "Clear FAISS history",
        disabled=not (clear_history_toggle and st.session_state.pdf_processed)
    ):
        cleared = clear_vector_store()
        st.session_state.pdf_processed = False
        st.session_state.messages = []

        if cleared:
            st.success("Stored FAISS embedding history cleared.")
        else:
            st.warning("FAISS history could not be cleared right now.")

        st.rerun()
    st.header("📂 Upload Expense PDFs")

    st.caption("After selecting files, click 'Process PDFs' to add them to FAISS.")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("⚙️ Process PDFs", disabled=not uploaded_files):
        with st.spinner("Processing your PDFs..."):
            all_chunks = []
            processed_files = []
            skipped_files = []

            for uploaded_file in uploaded_files:

                # Save uploaded file to a temp path on disk
                # This is the "uploaded_file_path" that gets passed dynamically
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    uploaded_file_path = tmp_file.name  # ← dynamic path!

                # Pass the path to the pdf processor
                chunks = load_and_chunk_pdf(
                    uploaded_file_path,
                    source_name=uploaded_file.name,
                )
                all_chunks.extend(chunks)

                if chunks:
                    processed_files.append(uploaded_file.name)
                else:
                    skipped_files.append(uploaded_file.name)

                # Clean up temp file after processing
                os.remove(uploaded_file_path)

            # Build and save FAISS vector store
            if all_chunks:
                build_vector_store(all_chunks)
                st.session_state.pdf_processed = True
                st.session_state.messages = []
            else:
                st.session_state.pdf_processed = history_available

        if processed_files:
            st.info("Processed: " + ", ".join(processed_files))

        if skipped_files:
            st.warning(
                "Skipped unreadable PDF(s): "
                + ", ".join(skipped_files)
                + ". These likely need OCR or a text-searchable PDF."
            )

        st.success(f"✅ {len(uploaded_files)} PDF(s) processed successfully!")

    if st.session_state.pdf_processed:
        st.info("✅ PDFs ready — start asking questions!")

# ── Main Chat Area ────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input(
    "Ask about your expenses... e.g. 'What was my total spend in March?'",
    disabled=not st.session_state.pdf_processed
)

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and show AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your expenses..."):
            answer = get_rag_response(user_input)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
