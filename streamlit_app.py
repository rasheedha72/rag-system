import streamlit as st
import requests
from datetime import datetime

# Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Title
st.title("🤖 RAG Chatbot System")
st.markdown("Ask questions about your documents")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("API URL", value="http://localhost:8000")
    
    st.markdown("---")
    st.subheader("📊 Statistics")
    st.metric("Documents", "2")
    st.metric("Total Chunks", "6")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📄 View sources"):
                for source in message["sources"]:
                    st.write(f"- {source}")

# ✅ MOVED OUTSIDE COLUMNS - Chat input must be at top level
user_input = st.chat_input("Ask a question...")

# Clear button in columns (optional, doesn't need to be here)
col1, col2 = st.columns([0.9, 0.1])
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Process query
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{api_url}/ask",
                json={"query": user_input},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
                st.rerun()
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.caption("RAG Chatbot v1.0 | Powered by LangChain + Groq")
