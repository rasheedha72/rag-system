from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()  # ← Add this line

# Load API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load vector store and embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)


# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Groq API function (direct HTTP call, no SDK)
def call_groq_api(context: str, question: str) -> str:
    """Call Groq API directly without using langchain_groq"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    prompt = f"""You are a helpful assistant answering questions based on the provided documents.

IMPORTANT RULES:
1. Answer ONLY using information from the provided context
2. If the answer is not in the context, say "I don't have that information"
3. Always cite which document the information comes from
4. Be concise and clear

Context from documents:
{context}

Question: {question}

Answer:"""
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling Groq API: {str(e)}"

# RAG QA function
def rag_qa(query: str):
    docs = retriever.invoke(query)
    context = "\n\n".join(
        f"{d.page_content} (Source: {d.metadata['source']})"
        for d in docs
    )
    answer = call_groq_api(context, query)
    return answer, docs

# FastAPI
app = FastAPI(
    title="RAG Chatbot API",
    description="Production-ready RAG system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list

@app.get("/")
async def root():
    return {
        "status": "✓ RAG Chatbot API is running",
        "version": "1.0.0",
        "endpoints": ["/ask", "/health"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        answer, docs = rag_qa(request.query)
        sources = list(set([doc.metadata["source"] for doc in docs]))
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources
        )
    except Exception as e:
        return QueryResponse(
            query=request.query,
            answer=f"Error: {str(e)}",
            sources=[]
        )
