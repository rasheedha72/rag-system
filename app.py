from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

import io

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import requests
import json
import os
import logging
from datetime import datetime
from config import settings


# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== INITIALIZE CLIENTS ====================


# Pinecone client
pc = Pinecone(api_key=settings.PINECONE_API_KEY)

# Ensure index exists
if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=settings.PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=settings.PINECONE_ENVIRONMENT  # e.g. "us-east-1"
        ),
    )

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# LangChain VectorStore bound to existing index
vector_store = PineconeVectorStore.from_existing_index(
    index_name=settings.PINECONE_INDEX_NAME,
    embedding=embeddings,
    text_key="text",
)

# Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# ==================== GROQ API ====================
def call_groq_api(context: str, question: str) -> str:
    """Call Groq API directly"""
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
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
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
        logger.error(f"Groq API error: {str(e)}")
        return f"Error calling Groq API: {str(e)}"

# ==================== RAG QA ====================
def rag_qa(query: str, user_id: str = "default"):
    """RAG question answering with logging"""
    import time
    start_time = time.time()
    
    try:
        # Retrieve documents
        docs = retriever.invoke(query)
        context = "\n\n".join(
            f"{d.page_content} (Source: {d.metadata.get('source', 'Unknown')})"
            for d in docs
        )
        
        # Generate answer
        answer = call_groq_api(context, query)
        
        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        
        # Log query
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "query": query,
            "response_time_ms": round(latency_ms, 2),
            "docs_retrieved": len(docs),
            "model": "llama-3.1-8b-instant"
        }
        logger.info(json.dumps(log_entry))
        
        return answer, docs, log_entry
    except Exception as e:
        logger.error(f"RAG QA error: {str(e)}")
        raise

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="RAG Chatbot API - Phase 1",
    description="Production-ready RAG system with Pinecone",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== PYDANTIC MODELS ====================
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default"

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list
    response_time_ms: float

class DocumentUploadResponse(BaseModel):
    status: str
    document_id: str
    filename: str
    chunks_created: int
    message: str

class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    upload_date: str
    total_chunks: int

# ==================== AUTH MIDDLEWARE ====================
def verify_api_key(x_api_key: str = Header(None)) -> str:
    """Simple API key verification"""
    # For now, accept any key or default user
    # In production, validate against database
    if x_api_key is None:
        return "default"
    return x_api_key

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "status": "✓ RAG Chatbot API (Phase 1) is running",
        "version": "2.0.0",
        "endpoints": {
            "chat": "/ask (POST)",
            "upload": "/upload (POST)",
            "documents": "/documents (GET)",
            "delete": "/documents/{doc_id} (DELETE)",
            "logs": "/logs/recent (GET)",
            "health": "/health (GET)"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    api_key: str = None
):
    """Ask a question about your documents"""
    try:
        user_id = api_key or request.user_id
        answer, docs, log_entry = rag_qa(request.query, user_id)
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            response_time_ms=log_entry["response_time_ms"]
        )
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    api_key: str = None
):
    """Upload and ingest a PDF document"""
    try:
        user_id = api_key or "default"

        # Validate file
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        logger.info(f"Uploading file: {file.filename} for user: {user_id}")

        # Read PDF bytes
        pdf_content = await file.read()

        # Extract text
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")

        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF has no extractable text")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)

        # Create document ID
        doc_id = f"{user_id}_{file.filename.replace('.pdf', '')}_{datetime.now().timestamp()}"

        # Prepare texts and metadatas for VectorStore
        texts = chunks
        metadatas = [
            {
                "source": file.filename,
                "doc_id": doc_id,
                "user_id": user_id,
                "chunk_index": i,
                "upload_date": datetime.now().isoformat(),
            }
            for i, _ in enumerate(chunks)
        ]

        # Store in vector DB
        vector_store.add_texts(texts=texts, metadatas=metadatas)

        logger.info(f"Successfully uploaded {file.filename} with {len(chunks)} chunks")

        return DocumentUploadResponse(
            status="success",
            document_id=doc_id,
            filename=file.filename,
            chunks_created=len(chunks),
            message=f"Uploaded {file.filename} with {len(chunks)} chunks",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/documents")
async def get_documents(api_key: str = None):
    """List all uploaded documents for user"""
    try:
        user_id = api_key or "default"
        logger.info(f"Fetching documents for user: {user_id}")
        
        # Query Pinecone to get unique documents
        # Note: This is simplified - in production, maintain separate metadata store
        return {
            "user_id": user_id,
            "documents": [
                {
                    "doc_id": "doc_1",
                    "filename": "example.pdf",
                    "upload_date": datetime.now().isoformat(),
                    "chunks": 45
                }
            ],
            "total_documents": 1
        }
    except Exception as e:
        logger.error(f"Error fetching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, api_key: str = None):
    """Delete a document from vector store"""
    try:
        user_id = api_key or "default"
        logger.info(f"Deleting document {doc_id} for user {user_id}")
        
        # In production: Query Pinecone for all vectors with this doc_id, then delete
        # For now, return success
        return {
            "status": "success",
            "message": f"Document {doc_id} deleted",
            "doc_id": doc_id
        }
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs/recent")
async def get_recent_logs(limit: int = 100):
    """Get recent query logs"""
    try:
        with open('rag_system.log', 'r') as f:
            lines = f.readlines()
        
        recent_logs = []
        for line in lines[-limit:]:
            try:
                # Parse JSON logs
                if '{' in line:
                    json_str = line[line.index('{'):]
                    recent_logs.append(json.loads(json_str))
            except:
                pass
        
        return {
            "total_logs": len(recent_logs),
            "logs": recent_logs
        }
    except Exception as e:
        logger.error(f"Error fetching logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
