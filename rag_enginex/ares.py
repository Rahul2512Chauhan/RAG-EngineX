from sentence_transformers import CrossEncoder
import logging

logging.basicConfig(level=logging.INFO, format="📝 [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ARESScorer:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"📦 Loading ARES model: {model_name}")
        self.model = CrossEncoder(model_name)

    def score(self, question: str, answer: str, contexts: list[str]) -> float:
        if not contexts:
            logger.warning("⚠️ No context provided to ARES scorer.")
            return 0.0

        pairs = [(f"{question} {answer}", context) for context in contexts]
        scores = self.model.predict(pairs)

        best_score = max(scores)
        logger.info(f"🎯 ARES max score: {best_score:.4f}")
        return float(best_score)
