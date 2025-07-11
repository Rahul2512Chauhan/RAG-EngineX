from rag_enginex.utils import load_pdf 
from rag_enginex.chunking import chunk_text

#load and chunk
pdf_path = "docs/1706.03762v7.pdf"
text = load_pdf(pdf_path)
chunks = chunk_text(text , chunk_size=300)

#show preview
print(f"✅ Loaded {len(chunks)} clean chunks.\n")
for i, chunk in enumerate(chunks[:5]):
    print(f"[{i}] {chunk[:100]}...\n")
