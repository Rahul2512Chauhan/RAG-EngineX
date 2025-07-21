import os
from typing import List
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#load gemini api key 
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=SecretStr(GEMINI_API_KEY), 
    temperature=0.2
)


# LangChain PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.
First, try to answer the question using the context below. If the context does not contain the answer, use your own knowledge to answer.


Context:
{context}

Question:
{question}

Answer:
"""
)

# Create LCEL chain
rag_chain = (
    RunnablePassthrough.assign(context=lambda x: "\n\n".join(x["context_chunks"])) # Ensure context is a single string
    | prompt_template
    | llm
    | StrOutputParser() # To explicitly get the string output from the LLM
)


def generate_answer(context_chunks: List[str], question:str) -> str:
    """
    First tries to answer using context. If answer is generic or context is insufficient,
    falls back to Gemini's own knowledge without context.
    """
    # Ensure all items are strings
    cleaned_chunks = [str(chunk) for chunk in context_chunks]
    
    try:
        # Step 1: Run context-based RAG
        rag_answer = rag_chain.invoke({"context_chunks": cleaned_chunks, "question": question}).strip()
    except Exception as e:
        return f"[LangChain Gemini Error] {str(e)}"
    

    # Step 2: Fallback logic
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
            fallback_response = llm.invoke(f"Answer the following question using your own knowledge:\n\n{question}")
            return fallback_response.strip() # type: ignore
        except Exception as e:
            return f"[Gemini Direct Error] {str(e)}"

    return rag_answer