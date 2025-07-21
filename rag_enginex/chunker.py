from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def chunk_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
) -> List[str]:
    
    """
        Split text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.

    Args:
        text (str): The input text to split.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlapping characters between chunks.

    Returns:
        List[str]: List of text chunks.
    
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_text(text)
    return chunks