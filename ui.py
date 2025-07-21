import streamlit as st
from rag_enginex.loader import load_pdf_text
from rag_enginex.chunker import chunk_text
from rag_enginex.embedder import BGEEmbedder
from rag_enginex.vector_store import FAISSVectorestore
from rag_enginex.llm_answer import generate_answer
import os

# App title and styling
st.set_page_config(page_title="RAG-EngineX Chatbot", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .block-container { padding: 2rem 2rem 2rem; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
        border: none;
    }
    .stTextInput>div>input {
        padding: 0.6rem;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📄 RAG-EngineX PDF Chatbot")
st.write("Upload a PDF and ask questions. If context is insufficient, Gemini will fall back to its own knowledge.")

# PDF upload
pdf_file = st.file_uploader("📤 Upload a PDF", type="pdf")

# Session state for storing processed data
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embedder" not in st.session_state:
    st.session_state.embedder = BGEEmbedder()

# Process PDF
if pdf_file is not None:
    with st.spinner("🔍 Processing PDF..."):
        pdf_path = os.path.join("docs", pdf_file.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())

        text = load_pdf_text(pdf_path)
        chunks = chunk_text(text)

        embeddings = st.session_state.embedder.embed_chunks(chunks)
        vector_store = FAISSVectorestore(dim=len(embeddings[0]))
        vector_store.add_embeddings(embeddings, chunks)
        st.session_state.vector_store = vector_store
        st.success("✅ PDF processed and indexed!")

# Question input
question = st.text_input("🤔 Ask a question about the document:")

# Answer generation
if st.button("🧠 Generate Answer"):
    if not question:
        st.warning("Please enter a question.")
    elif st.session_state.vector_store is None:
        st.error("Please upload and process a PDF first.")
    else:
        with st.spinner("🔮 Generating answer..."):
            query_vec = st.session_state.embedder.embed_chunks([question])[0]
            results = st.session_state.vector_store.search(query_vec, top_k=3)
            top_chunks = [result[0] for result in results]

            final_answer = generate_answer(top_chunks, question)
            st.markdown("### 📝 Answer")
            st.success(final_answer)

            with st.expander("🔍 Retrieved Chunks"):
                for i, chunk in enumerate(top_chunks):
                    st.markdown(f"**Chunk {i+1}:**\n{chunk}")

# Footer
st.markdown("""
---
Made with ❤️ by Rahul | Powered by Gemini + FAISS + Streamlit
""")