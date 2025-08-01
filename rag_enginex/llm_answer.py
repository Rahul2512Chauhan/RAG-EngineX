import os
from typing import List
from dotenv import load_dotenv
from pydantic import SecretStr
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI  # Groq-compatible wrapper
import streamlit as st

# === Load API Key ===
load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

# === LangChain LLM (Groq using OpenAI-compatible endpoint) ===
groq_llm = ChatOpenAI(
    model="llama3-8b-8192",
    api_key=SecretStr(GROQ_API_KEY),
    base_url="https://api.groq.com/openai/v1",
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

# === LangChain RAG Chain (Groq) ===
groq_rag_chain = (
    RunnablePassthrough.assign(context=lambda x: "\n\n".join(x["context_chunks"]))
    | prompt_template
    | groq_llm
    | StrOutputParser()
)

# === Main RAG Answer Generator ===
def generate_answer(
    question: str,
    context_chunks: List[str],
    llm_provider: str = "groq"
) -> str:
    """
    Generates an answer to the question using RAG (retrieved chunks).
    Falls back to LLM's own knowledge if context is insufficient.

    Parameters:
        question (str): User query
        context_chunks (List[str]): Retrieved text chunks
        llm_provider (str): Only 'groq' is supported in this version

    Returns:
        str: Final answer string
    """
    if llm_provider != "groq":
        raise ValueError(f"[generate_answer] Unsupported LLM provider: {llm_provider}")

    cleaned_chunks = [str(chunk) for chunk in context_chunks]

    try:
        rag_answer = groq_rag_chain.invoke({
            "context_chunks": cleaned_chunks,
            "question": question
        }).strip()
    except Exception as e:
        return f"[LangChain Groq Error] {str(e)}"

    fallback_triggers = [
        "not available in the provided context",
        "does not contain information",
        "cannot answer using only the context",
        "insufficient context",
        "based on the context, I cannot",
    ]

    if any(trigger in rag_answer.lower() for trigger in fallback_triggers):
        print("⚠️ Insufficient context — falling back to Groq model's own knowledge...")
        try:
            fallback_response = groq_llm.invoke(
                f"Answer the following question using your own knowledge:\n\n{question}"
            )
            return fallback_response.strip()  # type: ignore
        except Exception as e:
            return f"[Groq Direct Error] {str(e)}"

    return rag_answer
