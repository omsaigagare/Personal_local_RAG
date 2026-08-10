import pypdf
import chromadb
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_in_memory_vector_db(file_path: str):
    reader = pypdf.PdfReader(file_path)
    raw_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    if not raw_text.strip():
        raise ValueError("❌ OCR Required: The uploaded PDF contains no selectable text.")

    # Using Cohere for high-quality, free embeddings
    embedding_model = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=st.secrets["COHERE_API_KEY"]
    )
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)

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
    if vector_db is None:
        return "⚠️ No active document loaded in memory."

    results = vector_db.similarity_search_with_relevance_scores(query, k=top_k)
    valid_chunks = [f"[Chunk {idx} | Relevance: {score:.2f}]\n{doc.page_content.strip()}" for idx, (doc, score) in enumerate(results, 1) if score >= threshold]

    if not valid_chunks:
        return "No relevant context chunks found."
    return "\n\n".join(valid_chunks)

def generate_rag_response(query: str, retrieved_chunks: str, chat_history: list = None) -> str:
    if "No relevant context chunks found" in retrieved_chunks or "⚠️ No active document" in retrieved_chunks:
        return f"⚠️ **Grounded Analytics Aborted:** {retrieved_chunks}"

    # Using Groq's Llama 3 for fast, free inference
    llm = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    history_str = "No previous conversation history."
    if chat_history and len(chat_history) > 0:
        history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history[-4:]])

    system_rules = (
       "You are an expert analytical document evaluator.\n"
        "1. FACTUAL BOUNDARY: Do not invent facts.\n"
        "2. EVIDENCE-FIRST REASONING: Output a bulleted list of exact quotes before concluding.\n"
        "Conversation History:\n{history}\n\n"
        "Context Evidence:\n{context}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_rules),
        ("human", "{input}")
    ])

    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"context": retrieved_chunks, "history": history_str, "input": query})