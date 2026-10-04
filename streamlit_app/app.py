import os
import streamlit as st
import httpx

from streamlit_app.pages.upload import render_upload_page
from streamlit_app.pages.chat import render_chat_page
from streamlit_app.pages.inventory import render_inventory_page
from streamlit_app.pages.analysis import render_analysis_page
from streamlit_app.pages.audit import render_audit_page

# Page Configuration
st.set_page_config(
    page_title="AUTOSAR HLD Document Analysis Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL from env or default
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Fetch available documents from backend
def get_documents():
    try:
        res = httpx.get(f"{API_BASE_URL}/documents", timeout=5.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []


def check_system_health():
    try:
        res = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


# Sidebar Navigation
st.sidebar.markdown(
    """
    <div style="padding-bottom: 8px;">
        <h2 style="margin: 0; color: #0F172A; font-size: 20px;">🚗 AUTOSAR HLD</h2>
        <span style="color: #64748B; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Document Analysis Assistant</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

# Active Document Selection
docs = get_documents()
doc_options = {d["id"]: f"#{d['id']} — {d['filename']} ({d['page_count']} pages)" for d in docs}

selected_doc_id = None
if docs:
    st.sidebar.markdown("**Active Document:**")
    selected_idx = 0
    if "active_document_id" in st.session_state and st.session_state["active_document_id"] in doc_options:
        selected_idx = list(doc_options.keys()).index(st.session_state["active_document_id"])
    
    selected_doc_id = st.sidebar.selectbox(
        "Select Document",
        options=list(doc_options.keys()),
        format_func=lambda x: doc_options[x],
        index=selected_idx,
        label_visibility="collapsed"
    )
    st.session_state["active_document_id"] = selected_doc_id
    active_doc_name = next((d["filename"] for d in docs if d["id"] == selected_doc_id), "Unknown")
else:
    st.sidebar.info("No documents indexed yet. Start by uploading an HLD PDF.")
    active_doc_name = ""

st.sidebar.markdown("---")

# Main Navigation
nav_selection = st.sidebar.radio(
    "Navigation",
    [
        "Upload HLD",
        "Ask the HLD",
        "Architecture Inventory",
        "Analysis & Inconsistencies",
        "Audit Trail",
        "System Settings"
    ],
    index=0 if not docs else 1
)

st.sidebar.markdown("---")

# Sidebar Health Indicator
health_data = check_system_health()
if health_data:
    ollama_stat = health_data.get("ollama", {})
    ollama_color = "🟢" if ollama_stat.get("available") else "🟡"
    st.sidebar.markdown(
        f"""
        <div style="font-size: 11px; color: #475569; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 4px; padding: 8px;">
            <div><strong>Backend API:</strong> 🟢 Connected</div>
            <div><strong>ChromaDB:</strong> 🟢 {health_data.get('chromadb', {}).get('total_indexed_chunks', 0)} chunks</div>
            <div><strong>Ollama LLM:</strong> {ollama_color} {ollama_stat.get('model', 'llama3.2:3b')}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.warning("⚠️ FastAPI Backend offline. Please ensure 'uvicorn app.main:app' is running.")


# Route to Pages
if nav_selection == "Upload HLD":
    render_upload_page(API_BASE_URL)

elif nav_selection == "Ask the HLD":
    if not selected_doc_id:
        st.warning("Please upload or select an active AUTOSAR HLD document first from the sidebar or Upload page.")
        render_upload_page(API_BASE_URL)
    else:
        render_chat_page(API_BASE_URL, selected_doc_id, active_doc_name)

elif nav_selection == "Architecture Inventory":
    if not selected_doc_id:
        st.warning("Please upload or select an active AUTOSAR HLD document first.")
    else:
        render_inventory_page(API_BASE_URL, selected_doc_id, active_doc_name)

elif nav_selection == "Analysis & Inconsistencies":
    if not selected_doc_id:
        st.warning("Please upload or select an active AUTOSAR HLD document first.")
    else:
        render_analysis_page(API_BASE_URL, selected_doc_id, active_doc_name)

elif nav_selection == "Audit Trail":
    render_audit_page(API_BASE_URL, selected_doc_id or 1)

elif nav_selection == "System Settings":
    st.header("⚙️ System Configuration & Diagnostics")
    st.markdown("Inspect backend services, local models, storage directories, and grounding configurations.")

    if health_data:
        st.subheader("Operational Health")
        st.json(health_data)
    else:
        st.error("Backend API is currently unreachable at " + API_BASE_URL)

    st.subheader("Environment Settings")
    st.code(
        f"""
APP_NAME: AUTOSAR HLD Document Analysis Assistant
EMBEDDING_MODEL: BAAI/bge-small-en-v1.5
OLLAMA_BASE_URL: http://localhost:11434
OLLAMA_MODEL: llama3.2:3b
CHROMA_DIR: ./data/chroma
UPLOAD_DIR: ./data/uploads
EXPORT_DIR: ./data/exports
DATABASE_URL: sqlite:///./data/autosar.db
        """,
        language="yaml"
    )

    st.subheader("Grounding Guarantees")
    st.markdown(
        """
        - **Local Only:** Documents and vector embeddings never leave the local machine.
        - **Document Isolation:** Vector searches are strictly partitioned by document and project IDs.
        - **Zero Hallucination Principle:** Queries with insufficient evidence explicitly return `INSUFFICIENT EVIDENCE`.
        - **Deterministic Citations:** Answers include exact `[Page X | Section: Y]` citations extracted from retrieved text.
        """
    )
