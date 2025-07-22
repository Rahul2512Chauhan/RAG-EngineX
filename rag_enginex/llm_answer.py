import os
from typing import List
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# === Load Gemini API Key ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# === LangChain Gemini Model ===
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=SecretStr(GEMINI_API_KEY), 
    temperature=0.2
)

# === RAG Prompt Template ===
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.
First, try to answer the question using the context below. If the context does not contain the answer, use your own knowledge to answer.

Context:
{context}

Question:
{question}

Answer:"""
)

# === LangChain RAG Chain (for Gemini) ===
gemini_rag_chain = (
    RunnablePassthrough.assign(context=lambda x: "\n\n".join(x["context_chunks"]))
    | prompt_template
    | gemini_llm
    | StrOutputParser()
)

# === Main RAG Answer Generator ===
def generate_answer(
    question: str,
    context_chunks: List[str],
    llm_provider: str = "gemini"
) -> str:
    """
    Generates an answer to the question using RAG (retrieved chunks).
    Falls back to LLM's own knowledge if context is insufficient.

    Parameters:
        question (str): User query
        context_chunks (List[str]): Retrieved text chunks
        llm_provider (str): One of "gemini" or "groq"

    Returns:
        str: Final answer string
    """

    # Ensure chunks are clean strings
    cleaned_chunks = [str(chunk) for chunk in context_chunks]

    if llm_provider == "gemini":
        try:
            # Step 1: Try context-based RAG
            rag_answer = gemini_rag_chain.invoke({
                "context_chunks": cleaned_chunks,
                "question": question
            }).strip()
        except Exception as e:
            return f"[LangChain Gemini Error] {str(e)}"

        # Step 2: Fallback detection
        fallback_triggers = [
            "not available in the provided context",
            "does not contain information",
            "cannot answer using only the context",
            "insufficient context",
            "based on the context, I cannot",
        ]

        if any(trigger in rag_answer.lower() for trigger in fallback_triggers):
            print("⚠️ Insufficient context — falling back to Gemini's own knowledge...")
            try:
                fallback_response = gemini_llm.invoke(
                    f"Answer the following question using your own knowledge:\n\n{question}"
                )
                return fallback_response.strip()  # type: ignore
            except Exception as e:
                return f"[Gemini Direct Error] {str(e)}"

        return rag_answer

    else:
        raise ValueError(f"[generate_answer] Unsupported LLM provider: {llm_provider}")
