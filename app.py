"""
RAG Document Q&A
=================
Upload a PDF, ask questions about it, and get answers grounded in the
document's actual content using Retrieval-Augmented Generation (RAG).

How it works:
1. Extract text from the uploaded PDF
2. Split the text into overlapping chunks
3. Convert each chunk into an embedding (a vector of numbers) using
   Google's embedding model
4. When the user asks a question, embed the question too, and find the
   chunks whose embeddings are most similar (cosine similarity)
5. Send those relevant chunks + the question to Gemini, so it answers
   using the document's content instead of guessing

Run with:
    streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
from google import genai
from pypdf import PdfReader
import numpy as np


# =============================================================================
# CONFIG
# =============================================================================

CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # overlap between consecutive chunks
TOP_K = 4                # how many chunks to retrieve per question
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.6-flash"


st.set_page_config(
    page_title="RAG Document Q&A",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# STYLING
# =============================================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #faf7f0;
    }
    .block-container {
        max-width: 1400px;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li {
        color: #2d2a24;
    }
    h1, h2, h3 { color: #1c1917 !important; }

    /* Sidebar (drawer) -- deep charcoal with emerald accents */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1c1917 0%, #14201c 100%);
        border-right: 1px solid rgba(16, 185, 129, 0.15);
    }
    section[data-testid="stSidebar"] * {
        color: #e7e5df !important;
    }
    section[data-testid="stSidebar"] h1 {
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        background: rgba(16, 185, 129, 0.10);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 0.6rem;
    }
    section[data-testid="stSidebar"] h3 {
        display: inline-block;
        border: 1px solid rgba(245, 158, 11, 0.5) !important;
        background: rgba(245, 158, 11, 0.10);
        color: #fbbf24 !important;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Bordered cards -- warm cream with emerald accent */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border-radius: 16px !important;
        border: 1px solid #e7e0d0 !important;
        border-top: 3px solid #10b981 !important;
        box-shadow: 0 3px 16px rgba(28, 25, 23, 0.06);
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.6rem;
    }

    /* Section labels inside cards ("Step 1", "Step 2") */
    .step-label {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #059669;
        margin-bottom: 0.5rem;
    }
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: #10b981;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 800;
    }

    /* File uploader */
    div[data-testid="stFileUploaderDropzone"] {
        background: #fdfcf8 !important;
        border: 2px dashed #d97706 !important;
        border-radius: 14px !important;
    }
    div[data-testid="stFileUploaderDropzone"] * {
        color: #57534e !important;
    }
    /* Direct fix for the exact button class found via DevTools inspection */
    .st-emotion-cache-1uufcrr {
        background-color: #d97706 !important;
        border: 1px solid #d97706 !important;
        color: #ffffff !important;
    }
    .st-emotion-cache-1uufcrr * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    div[data-testid="stFileUploaderDropzone"] [role="button"],
    div[data-testid="stFileUploaderDropzone"] [role="button"]:disabled,
    div[data-testid="stFileUploaderDropzone"] button,
    div[data-testid="stFileUploaderDropzone"] button:disabled {
        background: #d97706 !important;
        border: 1px solid #d97706 !important;
        color: #ffffff !important;
        opacity: 1 !important;
    }
    div[data-testid="stFileUploaderDropzone"] [role="button"] *,
    div[data-testid="stFileUploaderDropzone"] button * {
        color: #ffffff !important;
        fill: #ffffff !important;
        opacity: 1 !important;
    }
    div[data-testid="stBaseButton-secondary"] {
        background: #d97706 !important;
        color: #ffffff !important;
        border: 1px solid #d97706 !important;
        opacity: 1 !important;
    }
    div[data-testid="stBaseButton-secondary"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    /* Inputs */
    div[data-baseweb="input"] {
        border-radius: 10px !important;
        border: 1px solid #d6d3cb !important;
        background: #ffffff !important;
    }
    div[data-baseweb="input"] input {
        color: #1c1917 !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15);
    }

    /* Buttons (including form submit buttons) */
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 600;
        background: #10b981;
        color: #ffffff !important;
        border: none;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #059669;
    }

    /* Success / info / error boxes */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Answer card -- amber accent, distinct from emerald cards */
    .answer-card {
        background: #fffbf0;
        border: 1px solid #fde3a7;
        border-left: 5px solid #d97706;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 18px rgba(217, 119, 6, 0.08);
        margin-top: 0.6rem;
    }
    .answer-card, .answer-card * {
        color: #292420 !important;
    }
    .answer-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 12px;
        border-radius: 999px;
        background: #d97706;
        color: #ffffff;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.6rem;
    }
    .citation-badge {
        display: inline-block;
        padding: 1px 8px;
        border-radius: 999px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #059669;
        font-size: 0.7rem;
        font-weight: 700;
        vertical-align: middle;
        margin: 0 2px;
        white-space: nowrap;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e7e0d0 !important;
        border-radius: 12px !important;
    }
    /* Streamlit top-left menu icon */
.st-emotion-cache-12bp31y {
    color: white !important;
}

.st-emotion-cache-2x5h05 {
    color: white !important;
    fill: white !important;
}
</style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR (DRAWER)
# =============================================================================

def display_sidebar():
    with st.sidebar:
        st.title("RAG Document Q&A")
        st.caption(
            "Ask questions about any PDF and get answers grounded in its "
            "actual content, powered by Retrieval-Augmented Generation."
        )

        st.subheader("Model")
        st.write("Gemini 3.6 Flash")

        st.subheader("Embeddings")
        st.write("gemini-embedding-001")

        st.subheader("How it works")
        st.write("1. Upload a PDF")
        st.write("2. It's split into chunks and embedded")
        st.write("3. Your question retrieves the most relevant chunks")
        st.write("4. Gemini answers using only that context")

        st.subheader("Tech")
        st.write("Python")
        st.write("Streamlit")
        st.write("Google Gemini API")
        st.write("NumPy (vector search)")


# =============================================================================
# API KEY SETUP
# =============================================================================

def get_api_key():
    """Get the Gemini API key from Streamlit secrets, or ask the user for it."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return st.sidebar.text_input("Enter your Gemini API key", type="password")


# =============================================================================
# PDF PROCESSING
# =============================================================================

def extract_text_from_pdf(uploaded_file) -> str:
    """Read all text out of an uploaded PDF file."""
    reader = PdfReader(uploaded_file)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks so related sentences aren't cut
    apart awkwardly at chunk boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


# =============================================================================
# EMBEDDINGS
# =============================================================================

def embed_texts(client, texts: list[str], task_type: str) -> np.ndarray:
    """Convert a list of text strings into embedding vectors using Gemini.
    task_type is 'RETRIEVAL_DOCUMENT' for chunks, 'RETRIEVAL_QUERY' for
    the user's question -- this helps the model produce better-matched
    embeddings for search."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config={
    "task_type": task_type
}
    )
    return np.array([e.values for e in result.embeddings])


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one query vector and many document
    vectors, returning a similarity score for each document."""
    query_norm = query_vec / np.linalg.norm(query_vec)
    doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    return doc_norms @ query_norm


# =============================================================================
# RETRIEVAL + GENERATION
# =============================================================================

def retrieve_relevant_chunks(client, question: str, chunks: list[str], chunk_embeddings: np.ndarray, top_k: int) -> list[str]:
    """Find the chunks most relevant to the question."""
    question_embedding = embed_texts(client, [question], task_type="RETRIEVAL_QUERY")[0]
    scores = cosine_similarity(question_embedding, chunk_embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


def generate_answer(client, question: str, context_chunks: list[str], chat_history: list[dict]) -> str:
    """Ask Gemini to answer the question using only the retrieved context,
    citing which excerpt(s) each part of the answer came from. Recent
    conversation history is included so follow-up questions make sense."""
    numbered_context = "\n\n".join(
        f"[Excerpt {i}]\n{chunk}" for i, chunk in enumerate(context_chunks, 1)
    )

    history_text = ""
    if chat_history:
        recent = chat_history[-3:]  # last 3 exchanges keep the prompt small
        history_text = "\n\n".join(
            f"Previous Q: {turn['question']}\nPrevious A: {turn['raw_answer']}" for turn in recent
        )
        history_text = f"Conversation so far:\n{history_text}\n\n"

    prompt = f"""You are a helpful assistant answering questions based ONLY on the provided document excerpts below.
If the answer isn't in the excerpts, say you don't know based on the document -- don't make anything up.

After each claim or piece of information in your answer, cite which excerpt it came from using the
format [Excerpt N] (e.g. "The company was founded in 2015 [Excerpt 1]."). If a sentence draws on multiple
excerpts, cite all of them, e.g. [Excerpt 1][Excerpt 3].

{history_text}Document excerpts:
{numbered_context}

Current question (may be a follow-up to the conversation above): {question}

Answer (with citations):"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    return response.text


def style_citations(answer: str) -> str:
    """Replace plain '[Excerpt N]' markers with small styled inline badges."""
    import re
    return re.sub(
        r"\[Excerpt (\d+)\]",
        r'<span class="citation-badge">Excerpt \1</span>',
        answer,
    )


# =============================================================================
# MAIN APP
# =============================================================================

def display_header():
    components.html(
        """
        <style>
            html, body { margin: 0; padding: 0; background: transparent; }
            .hero-box {
                position: relative;
                font-family: 'Source Sans Pro', sans-serif;
                background: linear-gradient(135deg, #1c1917 0%, #14201c 55%, #1c1917 100%);
                border: 1px solid rgba(16, 185, 129, 0.3);
                border-radius: 18px;
                padding: 1.5rem 1.7rem;
                box-sizing: border-box;
                overflow: hidden;
            }
            .hero-box::before {
                content: "";
                position: absolute;
                top: -40%; right: -10%;
                width: 260px; height: 260px;
                background: radial-gradient(circle, rgba(16, 185, 129, 0.18), transparent 70%);
                border-radius: 50%;
            }
            .hero-box::after {
                content: "";
                position: absolute;
                bottom: -50%; left: -5%;
                width: 220px; height: 220px;
                background: radial-gradient(circle, rgba(245, 158, 11, 0.14), transparent 70%);
                border-radius: 50%;
            }
            .hero-title {
                position: relative;
                margin: 0 0 0.35rem 0;
                font-size: 2.1rem;
                font-weight: 800;
                color: #ffffff;
                letter-spacing: 0.2px;
            }
            .hero-accent { color: #34d399; }
            .hero-subtitle {
                position: relative;
                color: #d6d3cb;
                margin: 0 0 0.9rem 0;
                font-size: 0.98rem;
                line-height: 1.5;
                max-width: 640px;
            }
            .badge-row { position: relative; display: flex; gap: 8px; flex-wrap: wrap; }
            .badge {
                display: inline-block;
                padding: 4px 13px;
                border-radius: 999px;
                font-size: 0.75rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.4px;
            }
            .badge-emerald {
                border: 1px solid rgba(16, 185, 129, 0.5);
                background: rgba(16, 185, 129, 0.14);
                color: #6ee7b7;
            }
            .badge-amber {
                border: 1px solid rgba(245, 158, 11, 0.5);
                background: rgba(245, 158, 11, 0.14);
                color: #fbbf24;
            }
        </style>

        <div class="hero-box">
            <h1 class="hero-title">RAG Document <span class="hero-accent">Q&amp;A</span></h1>
            <p class="hero-subtitle">
                Upload a PDF, then ask questions about its content -- answered using
                Retrieval-Augmented Generation grounded in the document itself.
            </p>
            <div class="badge-row">
                <span class="badge badge-emerald">Gemini 3.6</span>
                <span class="badge badge-amber">RAG</span>
                <span class="badge badge-emerald">Vector Search</span>
                <span class="badge badge-amber">Embeddings</span>
            </div>
        </div>
        """,
        height=175,
    )


def main():
    display_sidebar()
    display_header()

    api_key = get_api_key()
    if not api_key:
        st.info("Enter your Gemini API key in the sidebar to get started.")
        st.stop()

    genai_client = genai.Client(api_key=api_key)

    with st.container(border=True):
        st.markdown(
            '<div class="step-label"><span class="step-number">1</span> Upload Your Document</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type="pdf",
            label_visibility="collapsed",
            help="Drag and drop a PDF, or click to browse.",
        )

    if uploaded_file is not None:
        # Only reprocess if this is a new file
        if st.session_state.get("processed_filename") != uploaded_file.name:
            with st.spinner("Reading and indexing the document..."):
                text = extract_text_from_pdf(uploaded_file)
                if not text.strip():
                    st.error("Couldn't extract any text from this PDF. It might be a scanned image PDF.")
                    st.stop()

                chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
                chunk_embeddings = embed_texts(genai_client, chunks, task_type="RETRIEVAL_DOCUMENT")

                st.session_state["chunks"] = chunks
                st.session_state["chunk_embeddings"] = chunk_embeddings
                st.session_state["processed_filename"] = uploaded_file.name

            st.success(f"Indexed {len(st.session_state['chunks'])} chunks from '{uploaded_file.name}'.")

        with st.container(border=True):
            st.markdown(
                '<div class="step-label"><span class="step-number">2</span> Ask a Question</div>',
                unsafe_allow_html=True,
            )
            with st.form("ask_form", clear_on_submit=True):
                question = st.text_input(
                    "Ask a question about the document",
                    label_visibility="collapsed",
                    placeholder="e.g. What are the key findings in this document?",
                )
                submitted = st.form_submit_button("Ask")

        st.session_state.setdefault("chat_history", [])

        if submitted and question:
            chat_history = st.session_state["chat_history"]

            # For follow-up questions ("tell me more about that"), blend in
            # the previous question so retrieval still finds the right topic
            retrieval_query = question
            if chat_history:
                retrieval_query = f"{chat_history[-1]['question']} {question}"

            with st.spinner("Finding relevant sections and generating an answer..."):
                relevant_chunks = retrieve_relevant_chunks(
                    genai_client,
                    retrieval_query,
                    st.session_state["chunks"],
                    st.session_state["chunk_embeddings"],
                    TOP_K,
                )
                answer = generate_answer(genai_client, question, relevant_chunks, chat_history)

            chat_history.append({
                "question": question,
                "raw_answer": answer,
                "chunks": relevant_chunks,
            })

        # Display the full conversation, most recent first
        for turn in reversed(st.session_state.get("chat_history", [])):
            st.markdown(f"**You:** {turn['question']}")
            with st.container(border=True):
                st.markdown('<span class="answer-badge">AI Answer</span>', unsafe_allow_html=True)
                st.markdown(style_citations(turn["raw_answer"]), unsafe_allow_html=True)

            with st.expander("Show retrieved source excerpts"):
                for i, chunk in enumerate(turn["chunks"], 1):
                    st.markdown(f"**Excerpt {i}:**")
                    st.caption(chunk)

        if st.session_state.get("chat_history"):
            if st.button("Clear conversation"):
                st.session_state["chat_history"] = []
                st.rerun()


if __name__ == "__main__":
    main()