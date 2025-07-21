import fitz #pymupdf
def load_pdf_text(pdf_path: str) -> str:
    """
        Extract text from a PDF using PyMuPDF.
    
    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Combined text of all pages.
    """

    text= ""

    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text() # type: ignore
    return text