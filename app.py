"""
DocQuery — Retrieval-Augmented Generation (RAG) Q&A Assistant
================================================================
Upload documents (PDF/TXT), ask questions in plain English, and get
answers grounded in the actual document content — with the exact
source passages shown, so you can verify every answer.

Architecture (explain this in interviews):
  1. INGEST  -> split uploaded docs into overlapping text chunks
  2. EMBED   -> convert each chunk into a vector using a local
                sentence-transformer model (no API key needed here)
  3. STORE   -> index those vectors in FAISS for fast similarity search
  4. RETRIEVE-> on a question, embed the question and pull the
                top-k most similar chunks from FAISS
  5. GENERATE-> pass the question + retrieved chunks to an LLM (Groq)
                and ask it to answer ONLY using that context

This "retrieve then generate" pattern is what makes it RAG rather than
a plain chatbot — the model is grounded in your documents instead of
guessing from its training data, which reduces hallucination and lets
you cite sources.
"""

import os
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
st.set_page_config(page_title="DocQuery — RAG Q&A Assistant", page_icon="📄", layout="wide")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"   # llama-3.3-70b-versatile was deprecated June 2026; check console.groq.com/docs/models if this is retired too
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4

SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")


# ---------------------------------------------------------------------
# Cached resources (loaded once per session, not on every rerun)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------
# Core RAG pipeline
# ---------------------------------------------------------------------
def load_text_from_upload(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF or TXT file."""
    if uploaded_file.name.lower().endswith(".pdf"):
        tmp_path = f"/tmp/{uploaded_file.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        return "\n\n".join(p.page_content for p in pages)
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


def build_vector_store(raw_texts: list[str], embedder) -> FAISS:
    """Chunk raw text and build a searchable FAISS index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks = []
    for text in raw_texts:
        all_chunks.extend(splitter.split_text(text))

    if not all_chunks:
        raise ValueError("No text could be extracted from the uploaded document(s).")

    return FAISS.from_texts(all_chunks, embedder)


def retrieve_context(vector_store: FAISS, question: str, k: int = TOP_K):
    """Return the top-k most relevant chunks for a question."""
    return vector_store.similarity_search(question, k=k)


def generate_answer(client: Groq, question: str, retrieved_docs) -> str:
    """Ask the LLM to answer strictly from the retrieved context."""
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)
    )

    system_prompt = (
        "You are a precise document-answering assistant. Answer the user's "
        "question using ONLY the information in the provided sources. "
        "If the sources do not contain the answer, say so clearly instead "
        "of guessing. Cite which source number(s) you used, like [Source 1]."
    )
    user_prompt = f"Sources:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
def main():
    st.title("📄 DocQuery")
    st.caption("Retrieval-Augmented Generation Q&A — ask questions, get answers grounded in your documents.")

    with st.sidebar:
        st.header("Setup")
        api_key = st.text_input(
            "Groq API key",
            type="password",
            value=os.environ.get("GROQ_API_KEY", ""),
            help="Free key from console.groq.com. Never hardcode this in your code — "
                 "use environment variables or Streamlit secrets.",
        )
        st.divider()
        st.header("Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
        )
        use_sample = st.checkbox("Use bundled sample documents instead", value=not uploaded_files)
        build_clicked = st.button("Build / rebuild index", type="primary")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
        st.session_state.doc_count = 0

    if build_clicked:
        with st.spinner("Reading documents, chunking, and building the vector index..."):
            texts = []
            if use_sample or not uploaded_files:
                for fname in os.listdir(SAMPLE_DOCS_DIR):
                    with open(os.path.join(SAMPLE_DOCS_DIR, fname), "r", encoding="utf-8") as f:
                        texts.append(f.read())
                doc_count = len(os.listdir(SAMPLE_DOCS_DIR))
            else:
                texts = [load_text_from_upload(f) for f in uploaded_files]
                doc_count = len(uploaded_files)

            embedder = get_embedder()
            st.session_state.vector_store = build_vector_store(texts, embedder)
            st.session_state.doc_count = doc_count
        st.success(f"Index built from {st.session_state.doc_count} document(s). Ask a question below.")

    st.divider()

    question = st.text_input("Ask a question about your documents:")
    ask_clicked = st.button("Ask")

    if ask_clicked:
        if not api_key:
            st.error("Add your Groq API key in the sidebar first (free at console.groq.com).")
        elif not st.session_state.vector_store:
            st.error("Build the index first — click 'Build / rebuild index' in the sidebar.")
        elif not question.strip():
            st.error("Type a question first.")
        else:
            with st.spinner("Retrieving relevant passages and generating an answer..."):
                retrieved = retrieve_context(st.session_state.vector_store, question)
                client = get_groq_client(api_key)
                try:
                    answer = generate_answer(client, question, retrieved)
                except Exception as e:
                    st.error(f"Groq API call failed: {e}")
                    return

            st.subheader("Answer")
            st.write(answer)

            with st.expander("Show retrieved source passages (what the model actually saw)"):
                for i, doc in enumerate(retrieved):
                    st.markdown(f"**Source {i+1}**")
                    st.text(doc.page_content)
                    st.markdown("---")


if __name__ == "__main__":
    main()
