
from sentence_transformers import SentenceTransformer
from typing import List

class BGEEmbedder:
    """
    Embedder using BAAI/bge-base-en model for dense retrieval.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en"):
        """
        Initialize the SentenceTransformer model.
        """
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """
        Embed a list of text chunks.

        Args:
            chunks (List[str]): List of text chunks.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        return self.model.encode(chunks, show_progress_bar=True, convert_to_numpy=True).tolist()
