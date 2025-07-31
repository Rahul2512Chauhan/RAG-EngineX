import os
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr


def get_groq_llm(model: str = "llama3-8b-8192") -> BaseChatModel:
    """
    Returns a LangChain-compatible ChatGroq LLM for use in RAGAS evaluation.

    Requires:
        - Environment variable GROQ_API_KEY to be set
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")

    return ChatGroq(
        temperature=0.0,
        model=model,
        api_key=SecretStr(api_key),
    )
