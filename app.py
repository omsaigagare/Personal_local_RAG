import os
import tempfile
import streamlit as st
from backend import create_in_memory_vector_db, retrieve_context_from_db, generate_rag_response

st.set_page_config(page_title="RAG Document Assistant", layout="wide")
st.title("📚 Local RAG Document Assistant")

# ==========================================
# CREDENTIAL VALIDATION
# ==========================================
# Verify API Key availability before proceeding
api_key_set = False
if "GOOGLE_API_KEY" in os.environ:
    api_key_set = True
elif hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
    api_key_set = True

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("📂 Document Ingestion")
    uploaded_file = st.file_uploader("Upload document (PDF)", type=["pdf"])

    if not api_key_set:
        st.error("⚠️ `GOOGLE_API_KEY` is missing! Configure it in `.streamlit/secrets.toml` or Cloud Secrets.")

    if st.button("🗑️ Reset Session & Memory"):
        st.session_state.clear()
        st.success("RAM and Chat History cleared!")
        st.rerun()

    st.divider()
    st.header("⚙️ Hyperparameters")
    top_k = st.slider("Top K Chunks to Retrieve", min_value=1, max_value=10, value=5)
    threshold = st.slider("Similarity Score Threshold", min_value=0.0, max_value=1.0, value=0.15, step=0.05)

# ==========================================
# IN-MEMORY FILE PROCESSING
# ==========================================
if uploaded_file is not None:
    if "active_file_name" not in st.session_state or st.session_state.active_file_name != uploaded_file.name:
        with st.spinner(f"Processing '{uploaded_file.name}' into RAM..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                vector_db, chunk_count, preview_text = create_in_memory_vector_db(tmp_file_path)
                st.session_state.vector_db = vector_db
                st.session_state.active_file_name = uploaded_file.name
                st.session_state.preview_text = preview_text
                st.session_state.messages = []  # Clear chat history when a new document is uploaded
                st.sidebar.success(f"✅ Ingested **{uploaded_file.name}** into RAM ({chunk_count} chunks)")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
    
    st.sidebar.info(f"⚡ Active Document: **{st.session_state.active_file_name}**")
    with st.expander("📄 Extracted Text Sample (Sanity Check)"):
        st.write(st.session_state.get("preview_text", "No preview available."))
else:
    st.info("👈 Upload a document in the sidebar to begin.")

# ==========================================
# CONTINUOUS CHAT ENGINE
# ==========================================
# Initialize chat memory buffer
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Render existing chat history on screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If the assistant provided RAG evidence, render the inspection expander
        if "evidence" in message and message["evidence"]:
            with st.expander("🔍 View Retrieved Chunks & Evidence"):
                st.text(message["evidence"])

# 2. Listen for new chat inputs
if prompt := st.chat_input("Ask a question or follow-up about the document..."):
    if "vector_db" not in st.session_state:
        st.warning("⚠️ Please upload a PDF document first.")
    elif not api_key_set:
        st.error("❌ Cannot process query: `GOOGLE_API_KEY` is not set.")
    else:
        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and thinking..."):
                evidence_chunks = retrieve_context_from_db(
                    st.session_state.vector_db, 
                    prompt, 
                    top_k=top_k, 
                    threshold=threshold
                )
                
                # Pass prior conversation history (excluding the prompt just submitted)
                prior_history = st.session_state.messages[:-1]
                response = generate_rag_response(prompt, evidence_chunks, prior_history)
                
                st.markdown(response)
                with st.expander("🔍 View Retrieved Chunks & Evidence"):
                    st.text(evidence_chunks)

        # Save assistant answer and evidence to session memory
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response, 
            "evidence": evidence_chunks
        })