from langchain_community.document_loaders import PyMuPDFLoader
from tqdm import tqdm

def load_pdf(path : str) -> str:
    """Load PDF and return combined text"""
    loader = PyMuPDFLoader(path)
    pages = loader.load()
    full_text = "\n".join([page.page_content for page in pages])
    return full_text

