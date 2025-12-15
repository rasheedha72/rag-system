# RAG Chatbot System

A production-ready Retrieval Augmented Generation (RAG) chatbot built with FastAPI, Streamlit, and LangChain.

## 🎯 Features

- **Vector Search:** FAISS-powered semantic document retrieval
- **LLM Integration:** Groq API for fast inference
- **REST API:** FastAPI backend with CORS support
- **Interactive UI:** Streamlit frontend for easy testing
- **Source Citations:** Automatic document attribution

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Groq API key: https://console.groq.com

### Installation

Clone repo
git clone https://github.com/rasheedha72/rag-system.git
cd rag-system

Create virtual environment
python -m venv .venv
.venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Add API key
echo "GROQ_API_KEY=your_key_here" > .env

text

### Run Locally

**Terminal 1 - FastAPI:**
uvicorn app:app --reload --port 8000

text

**Terminal 2 - Streamlit:**
streamlit run streamlit_app.py

text

Access at: `http://localhost:8501`

## 📁 Project Structure

rag-system/
├── app.py # FastAPI backend
├── streamlit_app.py # Streamlit frontend
├── requirements.txt # Dependencies
├── .gitignore # Git ignore rules
├── faiss_db/ # Vector database
└── documents/ # Source documents

text

## 🏗️ Architecture

User Query → Streamlit UI
↓
FastAPI Server
↓
Vector Retrieval (FAISS)
↓
LLM Generation (Groq)
↓
Answer + Sources

text

## 📊 Performance

- Retrieval: <100ms
- LLM Response: 1-2 seconds
- Total: <3 seconds per query

## 🤝 Contributing

Feel free to fork, modify, and improve!

## 📝 License

MIT

---

**Status:** Day 2 Complete ✅
**Next:** Optimize & Deploy to Hugging Face
Then commit and push:

bash
git add README.md
git commit -m "Add comprehensive README documentation"
git push origin main