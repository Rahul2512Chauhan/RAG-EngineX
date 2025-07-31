from typing import List, Dict, Union
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rag_enginex.llm_wrapper import get_groq_llm
from rag_enginex.ares import ARESScorer

import numpy as np
import logging

# ----------------------------
# Setup
# ----------------------------
_embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
_logger = logging.getLogger(__name__)
_llm = get_groq_llm()

# ----------------------------
# Embedding Helpers
# ----------------------------
def compute_embedding(text: str) -> np.ndarray:
    try:
        return _embedder.encode(text, convert_to_numpy=True)
    except Exception as e:
        _logger.error(f"Embedding failed for text: {text[:50]}... | {e}")
        dim = _embedder.get_sentence_embedding_dimension() or 768
        return np.zeros((dim,))


# ----------------------------
# Metric 1: Answer Relevance
# ----------------------------
def score_answer_relevance(answer: str, ground_truth: str) -> float:
    a_emb = compute_embedding(answer).reshape(1, -1)
    gt_emb = compute_embedding(ground_truth).reshape(1, -1)
    score = cosine_similarity(a_emb, gt_emb)[0][0]
    return round(float(score), 4)


# ----------------------------
# Metric 2: Faithfulness via LLM
# ----------------------------
def score_faithfulness_with_llm(answer: str, context: List[str]) -> float:
    context_joined = "\n".join(context[:5])
    prompt = f"""You are evaluating the FAITHFULNESS of an answer to a given context.

Context:
{context_joined}

Answer:
{answer}

Does the answer rely only on the information in the context?
Rate on a scale from 1 (not grounded at all) to 5 (completely grounded in context).
Just output a number from 1 to 5.

Score:"""

    try:
        response = _llm.invoke(prompt)
        score_str = getattr(response, "content", str(response)).strip().split()[0]
        score = float(score_str)
        return round(min(max(score, 1.0), 5.0), 2)
    except Exception as e:
        _logger.error(f"Faithfulness scoring failed: {e}")
        return 1.0


# ----------------------------
# Optional: ARES Score
# ----------------------------
def score_with_ares(question: str, answer: str, contexts: List[str]) -> float:
    try:
        scorer = ARESScorer()
        return round(scorer.score(question, answer, contexts), 4)
    except Exception as e:
        _logger.warning(f"ARES scoring failed: {e}")
        return 0.0


# ----------------------------
# Evaluation Wrapper
# ----------------------------
def evaluate_sample(
    question: str,
    answer: str,
    ground_truth: str,
    contexts: List[str],
    threshold: float = 0.7,
    use_ares: bool = True,
    use_classic: bool = True
) -> Dict[str, Union[str, float]]:
    result: Dict[str, Union[str, float]] = {
        "question": question,
        "answer": answer,
    }

    if use_classic:
        classic_scores: Dict[str, float] = {
            "faithfulness": score_faithfulness_with_llm(answer, contexts),
            "relevance": score_answer_relevance(answer, ground_truth),
        }
        result.update(classic_scores)

    if use_ares:
        result["ares_score"] = score_with_ares(question, answer, contexts)

    return result
