import streamlit as st
import requests
import json
from datetime import datetime

# ==================== CONFIG ====================
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Chatbot - Phase 1",
    page_icon="🤖",
    layout="wide"
)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input("API Key (optional)", type="password", value="default")
    
    st.divider()
    
    # Page selection
    page = st.radio("Navigation", ["💬 Chat", "📚 Documents", "📊 Logs"])
    
    st.divider()
    st.markdown("### About")
    st.markdown("""
    RAG Chatbot System v2.0
    - Pinecone vector DB
    - Multi-user support
    - Document upload
    - Query logging
    """)

# ==================== PAGE: CHAT ====================
if page == "💬 Chat":
    st.title("🤖 RAG Chatbot")
    st.markdown("Ask questions about your documents")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📄 View Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")
            if "latency" in message:
                st.caption(f"⏱️ Response time: {message['latency']:.2f}ms")
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Call API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"query": prompt, "user_id": api_key},
                        headers={"X-API-Key": api_key},
                        timeout=30
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    st.markdown(data["answer"])
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data["sources"],
                        "latency": data["response_time_ms"]
                    })
                    
                    with st.expander("📄 View Sources"):
                        for source in data["sources"]:
                            st.markdown(f"- {source}")
                    
                    st.caption(f"⏱️ Response time: {data['response_time_ms']:.2f}ms")
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ API Error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ==================== PAGE: DOCUMENTS ====================
elif page == "📚 Documents":
    st.title("📚 Document Library")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload New Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file and st.button("Upload", key="upload_btn"):
            with st.spinner("Uploading and processing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = requests.post(
                        f"{API_URL}/upload",
                        files=files,
                        headers={"X-API-Key": api_key},
                        timeout=60
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    st.success(f"✅ {data['message']}")
                    st.json({
                        "Document ID": data["document_id"],
                        "Chunks": data["chunks_created"]
                    })
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
    
    with col2:
        st.subheader("Your Documents")
        try:
            response = requests.get(
                f"{API_URL}/documents",
                headers={"X-API-Key": api_key},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data["documents"]:
                for doc in data["documents"]:
                    col_doc, col_delete = st.columns([4, 1])
                    with col_doc:
                        st.markdown(f"📄 **{doc['filename']}**")
                        st.caption(f"Chunks: {doc['chunks']} | Uploaded: {doc['upload_date'][:10]}")
                    with col_delete:
                        if st.button("🗑️", key=f"del_{doc['doc_id']}"):
                            # Delete logic here
                            st.rerun()
            else:
                st.info("No documents uploaded yet. Upload one above!")
        except Exception as e:
            st.error(f"❌ Error fetching documents: {str(e)}")

# ==================== PAGE: LOGS ====================
elif page == "📊 Logs":
    st.title("📊 Query Logs & Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        limit = st.slider("Recent logs to show", 10, 100, 50)
    
    try:
        response = requests.get(f"{API_URL}/logs/recent?limit={limit}", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data["logs"]:
            st.subheader(f"📈 Last {len(data['logs'])} Queries")
            
            # Display as table
            logs_data = []
            for log in data["logs"]:
                logs_data.append({
                    "Time": log.get("timestamp", "N/A")[:19],
                    "Query": log.get("query", "N/A")[:50],
                    "Response (ms)": log.get("response_time_ms", 0),
                    "Docs": log.get("docs_retrieved", 0),
                    "User": log.get("user_id", "N/A")
                })
            
            st.dataframe(logs_data, use_container_width=True)
            
            # Metrics
            st.subheader("📊 Metrics")
            col1, col2, col3 = st.columns(3)
            
            avg_latency = sum([log.get("response_time_ms", 0) for log in data["logs"]]) / len(data["logs"])
            avg_docs = sum([log.get("docs_retrieved", 0) for log in data["logs"]]) / len(data["logs"])
            total_queries = len(data["logs"])
            
            with col1:
                st.metric("Total Queries", total_queries)
            with col2:
                st.metric("Avg Latency (ms)", f"{avg_latency:.0f}")
            with col3:
                st.metric("Avg Docs Retrieved", f"{avg_docs:.1f}")
        else:
            st.info("No logs yet. Start chatting!")
    except Exception as e:
        st.error(f"❌ Error fetching logs: {str(e)}")
