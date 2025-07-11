from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from tqdm import tqdm

def clean_text(text: str) -> str:
    """Remove noise like emails, footers, etc."""

    #remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    #remove multiple newlines
    text = re.sub(r'\n{2,}', '\n', text)

    #remove isolaed author names/ affiliations
    text = re.sub(r'^[A-Z][a-z]+ [A-Z][a-z]+.*\n?', '', text, flags=re.MULTILINE)

    return text.strip()

def chunk_text(text:str , chunk_size=300 , chunk_overlap=30 ,min_length =100) ->list:
    """Clean and split text into high-quality chunks."""
    text = clean_text(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    raw_chunks = splitter.split_text(text)

    #filter out very short junk chunks
    filtered = [chunk.strip() for chunk in raw_chunks if len(chunk.strip()) >= min_length]
    return filtered