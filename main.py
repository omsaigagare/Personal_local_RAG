import os
import tempfile
from typing import List, Optional
import pypdf
import chromadb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize FastAPI App
app = FastAPI(
    title="RAG Document Evaluator API",
    description="Backend API for processing PDFs and answering grounded questions."
)

# 2. Enable CORS (Allows your HTML/Firebase frontend to talk to Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your Firebase URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global reference for the in-memory vector database
current_vector_db = None


# Pydantic Schema for incoming queries
class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[dict]] = None


def get_cohere_key() -> str:
    key = os.getenv("COHERE_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="Server Error: COHERE_API_KEY environment variable is missing.")
    return key


def get_groq_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="Server Error: GROQ_API_KEY environment variable is missing.")
    return key


def create_in_memory_vector_db(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name

    try:
        reader = pypdf.PdfReader(tmp_file_path)
        raw_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

        if not raw_text.strip():
            raise ValueError("OCR Required: The uploaded PDF contains no selectable text.")

        embedding_model = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=get_cohere_key()
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
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


def retrieve_context_from_db(vector_db, query: str, top_k: int = 5, threshold: float = 0.15) -> str:
    if vector_db is None:
        return "⚠️ No active document loaded in memory."

    results = vector_db.similarity_search_with_relevance_scores(query, k=top_k)
    valid_chunks = [
        f"[Chunk {idx} | Relevance: {score:.2f}]\n{doc.page_content.strip()}" 
        for idx, (doc, score) in enumerate(results, 1) if score >= threshold
    ]

    if not valid_chunks:
        return "No relevant context chunks found."
    return "\n\n".join(valid_chunks)


def generate_rag_response(query: str, retrieved_chunks: str, chat_history: list = None) -> str:
    if "No relevant context chunks found" in retrieved_chunks or "⚠️ No active document" in retrieved_chunks:
        return f"⚠️ Grounded Analytics Aborted: {retrieved_chunks}"

    llm = ChatGroq(
        api_key=get_groq_key(),
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


# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "online", "message": "RAG API is operational"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Receives a PDF file from the frontend and indexes it into Chroma vector store."""
    global current_vector_db
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are permitted.")
    
    try:
        contents = await file.read()
        vector_db, chunk_count, preview = create_in_memory_vector_db(contents)
        current_vector_db = vector_db
        return {
            "message": "PDF uploaded and vectorized successfully.",
            "filename": file.filename,
            "chunks_created": chunk_count,
            "preview": preview
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")


@app.post("/query")
async def query_document(request: QueryRequest):
    """Receives user questions from frontend and generates a RAG answer."""
    global current_vector_db
    
    if current_vector_db is None:
        raise HTTPException(status_code=400, detail="No active document found. Upload a PDF via /upload first.")

    try:
        retrieved_chunks = retrieve_context_from_db(current_vector_db, request.query)
        answer = generate_rag_response(request.query, retrieved_chunks, request.chat_history)
        return {
            "query": request.query,
            "answer": answer,
            "context": retrieved_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")