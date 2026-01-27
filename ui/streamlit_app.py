"""
Streamlit UI
Document Intelligence Platform

Professional chat interface for document Q&A.
"""

import os
import time
import requests
from typing import Dict, List, Optional
from datetime import datetime

import streamlit as st

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")


# ============================================================================
# Helper Functions
# ============================================================================

def api_request(endpoint: str, method: str = "GET", **kwargs) -> Dict:
    """Make API request with error handling."""
    url = f"{API_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            if "files" in kwargs:
                response = requests.post(url, files=kwargs["files"], timeout=180)
            else:
                response = requests.post(url, json=kwargs.get("json"), timeout=180)
        elif method == "DELETE":
            response = requests.delete(url, timeout=30)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Please ensure the server is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def format_confidence(confidence: float) -> str:
    """Format confidence score with color coding."""
    if confidence >= 0.8:
        return f"High ({confidence:.0%})"
    elif confidence >= 0.5:
        return f"Medium ({confidence:.0%})"
    else:
        return f"Low ({confidence:.0%})"


def format_citation(citation: Dict) -> str:
    """Format a citation for display."""
    parts = []
    if citation.get("company"):
        parts.append(citation["company"])
    if citation.get("filing_type"):
        parts.append(citation["filing_type"])
    if citation.get("fiscal_year"):
        parts.append(str(citation["fiscal_year"]))
    if citation.get("page"):
        parts.append(f"Page {citation['page']}")
    if citation.get("section"):
        parts.append(citation["section"].replace("_", " ").title())
    
    return ", ".join(parts)


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Document Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .citation-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        font-size: 0.85rem;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []


# ============================================================================
# Sidebar - Document Management
# ============================================================================

with st.sidebar:
    st.markdown("## Documents")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf"],
        help="Upload documents in PDF format"
    )
    
    if uploaded_file is not None:
        if st.button("Process Document", use_container_width=True):
            with st.spinner("Processing document..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                result = api_request("/upload", method="POST", files=files)
                
                if result:
                    st.success(f"{result['message']}")
                    st.info(f"Processing time: {result['processing_time']:.2f}s")
                    # Refresh document list
                    st.session_state.documents = api_request("/documents") or []
    
    st.divider()
    
    # Document list
    st.markdown("### Available Documents")
    
    # Refresh button
    if st.button("Refresh", use_container_width=True):
        st.session_state.documents = api_request("/documents") or []
    
    # Display documents
    docs = st.session_state.documents or api_request("/documents") or []
    
    if docs:
        for doc in docs:
            with st.expander(f"{doc.get('filename', 'Unknown')[:30]}..."):
                st.write(f"**Company:** {doc.get('company', 'N/A')}")
                st.write(f"**Type:** {doc.get('filing_type', 'N/A')}")
                st.write(f"**Year:** {doc.get('fiscal_year', 'N/A')}")
                st.write(f"**Pages:** {doc.get('total_pages', 'N/A')}")
                st.write(f"**Chunks:** {doc.get('total_chunks', 'N/A')}")
                
                if st.button("Delete", key=f"del_{doc['document_id']}"):
                    result = api_request(f"/documents/{doc['document_id']}", method="DELETE")
                    if result:
                        st.success("Document deleted")
                        st.rerun()
    else:
        st.info("No documents uploaded yet")
    
    st.divider()
    
    # System stats
    st.markdown("### System Stats")
    stats = api_request("/stats")
    if stats:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats.get("total_documents", 0))
        with col2:
            st.metric("Chunks", stats.get("total_chunks", 0))


# ============================================================================
# Main Content - Chat Interface
# ============================================================================

# Header
st.markdown('<p class="main-header">Document Intelligence</p>', unsafe_allow_html=True)


st.divider()


    


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display citations if present
        if message["role"] == "assistant" and "citations" in message:
            citations = message["citations"]
            if citations:
                st.markdown("---")
                st.markdown("**Sources:**")
                for i, citation in enumerate(citations, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="citation-box">
                            <strong>[{i}]</strong> {format_citation(citation)}<br>
                            <em>Relevance: {citation.get('relevance_score', 0):.0%}</em><br>
                            <small>{citation.get('text_snippet', '')[:200]}...</small>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Display confidence if present
        if message["role"] == "assistant" and "confidence" in message:
            st.caption(f"Confidence: {format_confidence(message['confidence'])}")

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Query the API
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            # Build request
            request_data = {
                "question": prompt,
                "top_k": 5,
                "include_citations": True,
                "include_reasoning": True
            }
            

            
            # Make request
            result = api_request("/query", method="POST", json=request_data)
            
            if result:
                # Display answer
                st.markdown(result["answer"])
                
                # Display reasoning steps if present
                if result.get("reasoning_steps"):
                    st.markdown("---")
                    st.markdown("**Calculation:**")
                    for step in result["reasoning_steps"]:
                        st.markdown(f"- {step}")
                
                # Display citations
                citations = result.get("sources", [])
                if citations:
                    st.markdown("---")
                    st.markdown("**Sources:**")
                    for i, citation in enumerate(citations, 1):
                        st.markdown(f"""
                        <div class="citation-box">
                            <strong>[{i}]</strong> {format_citation(citation)}<br>
                            <em>Relevance: {citation.get('relevance_score', 0):.0%}</em><br>
                            <small>{citation.get('text_snippet', '')[:200]}...</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display confidence
                confidence = result.get("confidence", 0)
                st.caption(f"Confidence: {format_confidence(confidence)} | "
                          f"Time: {result.get('processing_time', 0):.2f}s")
                
                # Save to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": citations,
                    "confidence": confidence,
                    "reasoning_steps": result.get("reasoning_steps")
                })
            else:
                error_msg = "Sorry, I couldn't process your question. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Clear chat button
if st.session_state.messages:
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()



