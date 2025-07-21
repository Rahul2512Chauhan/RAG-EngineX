from loader import load_pdf_text
from chunker import chunk_text
from embedder import BGEEmbedder
from vector_store import FAISSVectorestore
from llm_answer import generate_answer

text = load_pdf_text("docs/1706.03762v7.pdf")
chunks = chunk_text(text)


embedder = BGEEmbedder()
embeddings = embedder.embed_chunks(chunks)

vector_store = FAISSVectorestore(dim=len(embeddings[0]))
vector_store.add_embeddings(embeddings,chunks)
vector_store.save()

query = "What is encoder and decoder architecture?"
query_vec = embedder.embed_chunks([query])[0]
results = vector_store.search(query_vec , top_k = 3)

# Extract the top chunks from the search results
top_chunks = [result[0] for result in results]

final_answer = generate_answer(top_chunks , "What is encoder and decoder architecture?")
print("🔮 Final Answer:\n", final_answer)