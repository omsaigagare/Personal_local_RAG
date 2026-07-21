import os
import tempfile
import streamlit as st
from backend import create_in_memory_vector_db, retrieve_context_from_db, generate_rag_response

st.set_page_config(page_title="Local RAG Resume Evaluator", layout="wide")
st.title("📚 Local RAG Resume Evaluator")

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("📂 Document Ingestion")
    uploaded_file = st.file_uploader("Upload Candidate Resume (PDF)", type=["pdf"])

    if st.button("🗑️ Reset Session Memory"):
        st.session_state.clear()
        st.success("RAM cleared!")
        st.rerun()

    st.divider()
    st.header("⚙️ Hyperparameters")
    top_k = st.slider("Top K Chunks to Retrieve", min_value=1, max_value=10, value=5)
    threshold = st.slider("Similarity Score Threshold", min_value=0.0, max_value=1.0, value=0.15, step=0.05)

# ==========================================
# IN-MEMORY FILE PROCESSING
# ==========================================
if uploaded_file is not None:
    # Process only if file changed or memory is empty
    if "active_file_name" not in st.session_state or st.session_state.active_file_name != uploaded_file.name:
        with st.spinner(f"Processing '{uploaded_file.name}' into RAM..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # Build pure in-memory vector store
                vector_db, chunk_count, preview_text = create_in_memory_vector_db(tmp_file_path)
                
                # Store directly in Streamlit session memory
                st.session_state.vector_db = vector_db
                st.session_state.active_file_name = uploaded_file.name
                st.session_state.preview_text = preview_text
                
                st.sidebar.success(f"✅ Ingested **{uploaded_file.name}** into RAM ({chunk_count} chunks)")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
    
    # Display Active Status and Text Preview
    st.sidebar.info(f"⚡ Active Document: **{st.session_state.active_file_name}**")
    with st.expander("📄 Extracted Text Sample (Sanity Check)"):
        st.write(st.session_state.get("preview_text", "No preview available."))
else:
    st.info("👈 Upload a candidate resume (PDF) in the sidebar to begin.")

# ==========================================
# QUERY ENGINE
# ==========================================
user_query = st.text_input("Ask a question about the uploaded resume:")

if user_query:
    if "vector_db" not in st.session_state:
        st.warning("⚠️ Please upload a PDF document first.")
    else:
        with st.spinner("Retrieving from RAM and running Llama 3.2..."):
            evidence_chunks = retrieve_context_from_db(
                st.session_state.vector_db, 
                user_query, 
                top_k=top_k, 
                threshold=threshold
            )
            response = generate_rag_response(user_query, evidence_chunks)

        st.subheader("💡 Answer")
        st.markdown(response)

        with st.expander("🔍 View Retrieved Chunks & Evidence"):
            st.text(evidence_chunks)