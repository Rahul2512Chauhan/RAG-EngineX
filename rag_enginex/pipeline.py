from rag_enginex.loader import load_pdf_text
from rag_enginex.chunker import chunk_text
from rag_enginex.embedder import BGEEmbedder
from rag_enginex.vector_store import FAISSVectorestore
from rag_enginex.reranker import rerank
from rag_enginex.llm_answer import generate_answer
from rag_enginex.evaluator import evaluate_sample


def process_pdf(pdf_path: str, chunk_size: int = 800, chunk_overlap: int = 100):
    """
    Load → Chunk → Embed → Store
    Returns: chunks, embeddings, vector_store, embedder
    """
    # Step 1: Load raw text from PDF
    raw_text = load_pdf_text(pdf_path)

    # Step 2: Chunk the text
    chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Step 3: Embed the chunks
    embedder = BGEEmbedder()
    embeddings = embedder.embed_chunks(chunks)

    # Step 4: Store in FAISS vector store
    dim = len(embeddings[0])
    vector_store = FAISSVectorestore(dim=dim)
    vector_store.add_embeddings(embeddings, chunks)

    return chunks, embeddings, vector_store, embedder


def search_vector_store(query: str, vector_store, embedder, top_k: int = 5):
    """
    Embed query → Search vector store → Return top-k chunks
    """
    query_vector = embedder.embed_chunks([query])[0]
    results = vector_store.search(query_vector, top_k=top_k)
    return [chunk for chunk, _ in results]


def process_query(
    question: str,
    vector_store,
    embedder,
    top_k: int = 5,
    rerank_top_n: int = 3,
    use_reranker: bool = True,
    run_evaluation: bool = True,
    ground_truth: str = ""
):
    """
    Retrieve → (optional rerank) → Answer → (optional evaluate)
    Returns: answer, reranked_chunks, evaluation_scores (dict)
    """
    # Step 1: Retrieve relevant chunks
    retrieved_chunks = search_vector_store(question, vector_store, embedder, top_k=top_k)

    # Step 2: Optional reranking
    if use_reranker:
        reranked_chunks = rerank(question, retrieved_chunks, top_n=rerank_top_n)
    else:
        reranked_chunks = retrieved_chunks[:rerank_top_n]

    # Step 3: Generate answer
    answer = generate_answer(question, reranked_chunks)

    # Step 4: Optional evaluation (ARES + Faithfulness, Relevance, Recall, Precision)
    eval_scores = {}
    if run_evaluation and ground_truth:
        eval_scores = evaluate_sample(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            contexts=reranked_chunks,
            threshold=0.7,
            use_ares=True,
            use_classic=True,
        )

    return answer, reranked_chunks, eval_scores
