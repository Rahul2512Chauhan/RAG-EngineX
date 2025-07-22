# test_reranker.py
from reranker import rerank

query = "How do I reset the machine if it overheats?"
chunks = [
    "Always check the fuse box if power is lost.",
    "Reset the thermal fuse located on the rear panel in case of overheating.",
    "This machine requires weekly maintenance to run efficiently.",
    "Power consumption is rated at 1500W and 10A."
]

top_chunks = rerank(query, chunks, top_n=2)
print("Top Chunks:")
for chunk in top_chunks:
    print("-", chunk)
