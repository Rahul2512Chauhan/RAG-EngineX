"""
Reranker module for RAG-EngineX using BAAI/bge-reranker-base.
This reranker improves retrieval accuracy by scoring how relevant
each chunk is to the query using a CrossEncoder model.
"""

from sentence_transformers import CrossEncoder
from typing import List , Tuple

# Load CrossEncoder model once globally
# Can switch to a smaller model if needed for speed

model_name = "BAAI/bge-reranker-base"
reranker_model = CrossEncoder(model_name , max_length=512)

def rerank(query: str , chunks: List[str], top_n: int = 3) -> List[str]:
    """
    Rerank a list of text chunks based on relevance to the query.

    Args:
        query (str): The user query.
        chunks (List[str]): A list of retrieved document chunks.
        top_n (int): How many top-ranked chunks to return.

    Returns:
        List[str]: The top_n most relevant chunks sorted by relevance.
    """

    # Form pairs (query, chunk) for each retrieved chunk
    query_chunk_pairs: List[Tuple[str, str]] = [(query, chunk) for chunk in chunks]

    # Predict relevance scores for each pair
    scores = reranker_model.predict(query_chunk_pairs)

    # Pair scores with chunks and sort descending
    scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    # Return the top_n chunks only
    top_chunks = [chunk for chunk, _ in scored_chunks[:top_n]]
    return top_chunks