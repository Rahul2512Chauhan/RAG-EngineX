from rag_enginex import loader, chunker, embedder, vector_store, llm_answer
from rag_enginex.reranker import rerank  # ✅ Import reranker

def process_pdf(pdf_path: str, chunk_size: int, chunk_overlap: int):
    text = loader.load_pdf_text(pdf_path)
    chunks = chunker.chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    embedding_model = embedder.BGEEmbedder()
    embeddings = embedding_model.embed_chunks(chunks)

    db = vector_store.FAISSVectorestore(dim=len(embeddings[0]))
    db.add_embeddings(embeddings, chunks)

    return chunks, embeddings, db, embedding_model

def process_query(
    query: str,
    db: vector_store.FAISSVectorestore,
    embedder_model,
    top_k: int,
    rerank_top_n: int
):
    query_vector = embedder_model.embed_chunks([query])[0]
    results = db.search(query_vector, top_k=top_k)
    retrieved_chunks = [chunk for chunk, _ in results]

    reranked_chunks = rerank(query, retrieved_chunks, top_n=rerank_top_n)
    answer = llm_answer.generate_answer(query, reranked_chunks)

    return answer, reranked_chunks
