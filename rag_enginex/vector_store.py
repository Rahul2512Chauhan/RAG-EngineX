import faiss
import numpy as np
import os
import pickle
from typing import List , Tuple

class FAISSVectorestore:
    """
    Handles storing and querying embeddings using FAISS.
    """

    def __init__(self,dim: int, index_path: str = "faiss_index", save_metadata: bool = True):
        """
        Initialize the FAISS index.

        Args:
            dim (int): Dimension of embedding vectors.
            index_path (str): Folder to save/load index and metadata.
            save_metadata (bool): Whether to save/load chunk metadata.
        """
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []  # Corresponding chunks
        self.save_metadata = save_metadata


    def add_embeddings(self, embeddings: List[List[float]], chunks: List[str]):
        """
        Add embeddings and corresponding chunks to the FAISS index.

        Args:
            embeddings (List[List[float]]): Embedding vectors.
            chunks (List[str]): Corresponding text chunks.
        """
        if not embeddings or not chunks:
            print("Warning: Attempted to add empty embeddings or chunks.")
            return
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks.")

        np_embeddings = np.array(embeddings).astype("float32")
        if np_embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dim}, got {np_embeddings.shape[1]}.")

        self.index.add(np_embeddings) # type: ignore
        self.metadata.extend(chunks)


    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for top-k similar chunks given a query vector.

        Args:
            query_vector (List[float]): Embedding of the query.
            top_k (int): Number of top results to return.

        Returns:
            List of tuples: (matched_chunk, similarity_score)

        """
        if not self.index.ntotal:
            print("Warning: FAISS index is empty. No search performed.")
            return []

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query = np.array(query_vector).astype("float32").reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(f"Query vector dimension mismatch. Expected {self.dim}, got {query.shape[1]}.")

        # Ensure top_k doesn't exceed the number of indexed items
        actual_top_k = min(top_k, self.index.ntotal) 

        distances, indices = self.index.search(query, actual_top_k) # type: ignore
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            # This check `idx < len(self.metadata)` is good, though with proper
            # `add_embeddings` it should always hold.
            if idx < len(self.metadata): 
                results.append((self.metadata[idx], float(dist)))
        return results
    

    def save(self):
        """
        Save FAISS index and chunk metadata to disk.
        """
        os.makedirs(self.index_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))
        if self.save_metadata:
            with open(os.path.join(self.index_path, "chunks.pkl"), "wb") as f:
                pickle.dump(self.metadata, f)


    def load(self):
        """
        Load FAISS index and metadata from disk.
        """
        index_file = os.path.join(self.index_path, "index.faiss")
        metadata_file = os.path.join(self.index_path, "chunks.pkl")

        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
            print(f"FAISS index loaded from {index_file}")
        else:
            # Option 1: Raise error (current behavior, explicit)
            raise FileNotFoundError(f"FAISS index file not found at {index_file}!")
            # Option 2: Initialize an empty index (more flexible for "create if not exists")
            # print(f"FAISS index file not found at {index_file}. Initializing a new empty index.")
            # self.index = faiss.IndexFlatL2(self.dim)


        if self.save_metadata and os.path.exists(metadata_file):
            with open(metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"Metadata loaded from {metadata_file}")
        elif self.save_metadata and not os.path.exists(metadata_file):
            print(f"Warning: Metadata file not found at {metadata_file}. Metadata will be empty.")
            self.metadata = [] # Ensure metadata is empty if file is missing
        
