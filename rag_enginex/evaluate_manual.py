import os
import warnings
import logging
from rag_enginex.loader import load_pdf_text
from rag_enginex.chunker import chunk_text
from rag_enginex.embedder import BGEEmbedder
from rag_enginex.vector_store import FAISSVectorestore
from rag_enginex.reranker import rerank
from rag_enginex.llm_answer import generate_answer
from rag_enginex.evaluator import evaluate_sample

import pandas as pd

# ----------------------------
# Clean Logging Setup
# ----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="📝 [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------
# Try Rich Table Display (optional)
# ----------------------------
Table = None
Console = None
console = None
RICH_ENABLED = False

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    RICH_ENABLED = True
except ImportError:
    pass


# ----------------------------
# Core Evaluation Pipeline
# ----------------------------
def prepare_eval_dataframe(
    pdf_path: str,
    questions: list[str],
    ground_truths: list[str],
    top_k: int = 5
) -> pd.DataFrame:
    logger.info("📄 Loading and chunking document...")
    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text)

    logger.info("🔐 Embedding chunks...")
    embedder = BGEEmbedder()
    chunk_embeddings = embedder.embed_chunks(chunks)

    dim = len(chunk_embeddings[0])
    vector_store = FAISSVectorestore(dim=dim)
    vector_store.add_embeddings(chunk_embeddings, chunks)

    logger.info("📋 Preparing evaluation samples...")

    rows = []

    for idx, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
        logger.info(f"❓ Q{idx + 1}: {question}")
        query_embedding = embedder.embed_chunks([question])[0]
        retrieved = vector_store.search(query_embedding, top_k=top_k)
        top_chunks = [chunk for chunk, _ in retrieved]

        logger.info("🔁 Reranking context chunks...")
        reranked = rerank(question, top_chunks)

        logger.info("🧠 Generating answer...")
        answer = generate_answer(question, reranked)

        logger.info("🧪 Evaluating...")
        scores = evaluate_sample(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            contexts=reranked,
            use_classic=True,
            use_ares=True,  # Include ARES
        )

        rows.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "contexts": reranked,
            **scores
        })

    logger.info("✅ Evaluation complete.")
    return pd.DataFrame(rows)


# ----------------------------
# Display Results (Rich + Fallback)
# ----------------------------
def display_results(df: pd.DataFrame) -> None:
    display_columns = [
        "question", "faithfulness", "relevance",
        "context_recall", "context_precision", "ares_score"
    ]

    if console is not None and Table is not None:
        table = Table(title="Evaluation Results")
        for col in display_columns:
            table.add_column(col, justify="left")

        for _, row in df.iterrows():
            table.add_row(*[str(round(row[col], 4)) if isinstance(row[col], float) else str(row[col]) for col in display_columns])
        console.print(table)
    else:
        print("\n⚠️  'rich' not installed — showing plain table:")
        print(df[display_columns])


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    questions = [
        "What is the main idea of the document?",
        "What technologies has Rahul worked with?",
    ]

    ground_truths = [
        "The document is a resume of Rahul Chauhan, showcasing his experience, skills, and projects.",
        "Rahul has worked with front-end development, machine learning, and research tools.",
    ]

    df = prepare_eval_dataframe("sample.pdf", questions, ground_truths, top_k=5)
    display_results(df)

    df.to_csv("manual_eval_results.csv", index=False)
    logger.info("✅ Results saved to manual_eval_results.csv")
