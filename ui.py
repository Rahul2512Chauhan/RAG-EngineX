import os
import streamlit as st
import pandas as pd
import json
from rag_enginex import pipeline  # Central pipeline logic

# Page config
st.set_page_config(page_title="🧠 RAG-EngineX", layout="wide")

# Header
st.markdown(
    """
    <h1 style="text-align:center; color:#4A90E2;">📄 RAG-EngineX: Modular PDF Q&A Chatbot</h1>
    <p style="text-align:center; font-size:18px;">
        Upload a PDF. Ask smart questions. Let <b>Groq</b> answer using context retrieved by your own RAG pipeline.
    </p>
    """, unsafe_allow_html=True
)
st.markdown("---")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ RAG Settings")
    chunk_size = st.slider("🔪 Chunk Size", 100, 2000, 800, step=100)
    chunk_overlap = st.slider("🔁 Chunk Overlap", 0, 500, 100, step=50)
    top_k = st.slider("📚 Top K Chunks", 1, 10, 5)
    rerank_top_n = st.slider("🎯 Top N After Rerank", 1, top_k, 3)

    st.markdown("---")
    use_reranker = st.checkbox("🔀 Enable Reranker", value=True)
    run_evaluation = st.checkbox("📊 Show Evaluation Metrics", value=True)

    st.markdown("---")
    st.caption("Made with ❤️ for AI internships and beyond.")

# Session State Initialization
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "results_history" not in st.session_state:
    st.session_state.results_history = []

# Two-column layout
left, right = st.columns([1, 2])

with left:
    st.subheader("📤 Upload PDF")
    uploaded_pdf = st.file_uploader("Drop a PDF here", type=["pdf"])

    if uploaded_pdf:
        with st.spinner("🔍 Reading and processing..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_pdf.read())

            # 📄 Run full pipeline
            chunks, embeddings, db, embed_model = pipeline.process_pdf(
                "temp.pdf",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            st.session_state.chunks = chunks
            st.session_state.vector_store = db
            st.session_state.embedder = embed_model

        st.success(f"✅ {len(chunks)} chunks embedded and indexed!")

        with st.expander("📄 View Sample Chunks"):
            for i, chunk in enumerate(chunks[:5]):
                st.markdown(f"**Chunk {i+1}**: {chunk[:300]}...")

with right:
    st.subheader("💬 Ask a Question")
    question = st.text_input("Type your question about the PDF...", placeholder="E.g., What are the key projects mentioned?")

    if st.button("🚀 Generate Answer", use_container_width=True):
        if not question:
            st.warning("Please enter a question first.")
        elif not st.session_state.vector_store:
            st.error("Upload and process a PDF before asking.")
        else:
            with st.spinner("🧠 Thinking..."):
                answer, reranked_chunks, eval_scores = pipeline.process_query(
                    question=question,
                    vector_store=st.session_state.vector_store,
                    embedder=st.session_state.embedder,
                    top_k=top_k,
                    rerank_top_n=rerank_top_n,
                    use_reranker=use_reranker,
                    run_evaluation=run_evaluation,
                    ground_truth="Reverse Supply Chain Optimizer, Auto Researcher, Flight Delay Prediction"
                )

            st.success("✅ Answer generated!")
            st.markdown("### 📢 Final Answer")
            st.markdown(answer)

            # 🔎 Reranked chunks
            with st.expander("📚 Reranked Context Chunks Used"):
                for i, chunk in enumerate(reranked_chunks):
                    st.markdown(f"**Chunk {i+1}**: {chunk[:400]}...")

            # 📊 Evaluation Scores (Expandable Section)
            if run_evaluation and eval_scores:
                st.session_state.results_history.append({
                    "question": question,
                    "answer": answer,
                    **eval_scores
                })

                with st.expander("📊 Evaluation Metrics"):
                    for metric, score in eval_scores.items():
                        try:
                            score_float = float(score)
                            st.write(f"**{metric.replace('_', ' ').title()}**: {score_float:.2f}")
                            st.progress(min(score_float if 'score' not in metric else (score_float + 5) / 10, 1.0))

                        except (ValueError, TypeError):
                            st.write(f"**{metric.replace('_', ' ').title()}**: {score}")



# 📝 Export Answers Section
if st.session_state.results_history:
    st.markdown("### 📝 Export Answers")
    export_format = st.selectbox("Choose export format", ["CSV", "JSON"], key="export_format")

    if export_format == "CSV":
        df = pd.DataFrame(st.session_state.results_history)
        st.download_button(
            label="📥 Download CSV",
            data=df.to_csv(index=False),
            file_name="ragx_answers.csv",
            mime="text/csv"
        )
    elif export_format == "JSON":
        json_data = json.dumps(st.session_state.results_history, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name="ragx_answers.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("<center>✨ Built with modularity, style, and 💡 by RAG-EngineX ✨</center>", unsafe_allow_html=True)
