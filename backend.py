import pypdf
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# CONFIGURATION
# ==========================================
LLM_MODEL_NAME = "llama3.2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def create_in_memory_vector_db(file_path: str):
    """
    Reads the uploaded PDF and builds a pure IN-MEMORY Chroma vector store.
    No files are saved to disk; no stale data can persist.
    """
    # 1. Extract text from PDF
    reader = pypdf.PdfReader(file_path)
    raw_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    if not raw_text.strip():
        raise ValueError("❌ OCR Required: The uploaded PDF contains no selectable text (it might be an image/scan).")

    # 2. Chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(raw_text)

    # 3. Create Ephemeral (RAM-Only) Chroma Client
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    ephemeral_client = chromadb.EphemeralClient()
    
    vector_db = Chroma(
        client=ephemeral_client,
        collection_name="active_resume",
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    vector_db.add_texts(texts=chunks)
    return vector_db, len(chunks), raw_text[:300]


def retrieve_context_from_db(vector_db, query: str, top_k: int = 5, threshold: float = 0.15) -> str:
    """
    Queries the RAM vector store directly.
    """
    if vector_db is None:
        return "⚠️ No active document loaded in memory."

    results = vector_db.similarity_search_with_relevance_scores(query, k=top_k)

    valid_chunks = []
    for idx, (doc, score) in enumerate(results, 1):
        if score >= threshold:
            valid_chunks.append(f"[Chunk {idx} | Relevance: {score:.2f}]\n{doc.page_content.strip()}")

    if not valid_chunks:
        return "No relevant context chunks found exceeding the similarity score threshold."

    return "\n\n".join(valid_chunks)


def generate_rag_response(query: str, retrieved_chunks: str) -> str:
    if "No relevant context chunks found" in retrieved_chunks or "⚠️ No active document" in retrieved_chunks:
        return f"⚠️ **Grounded Analytics Aborted:** {retrieved_chunks}"

    local_llm = OllamaLLM(model=LLM_MODEL_NAME)

    system_rules = (
        "You are an expert resume evaluator. Analyze the provided context and answer the user's question based STRICTLY and EXCLUSIVELY on the provided data.\n"
        "1. Do not invent or assume facts.\n"
        "2. If a detail (like candidate name or college) is absent from the text, state that it is not present in the document.\n\n"
        "Context Evidence:\n{context}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_rules),
        ("human", "{input}")
    ])

    chain = prompt_template | local_llm
    return chain.invoke({"context": retrieved_chunks, "input": query})