import pypdf
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# CONFIGURATION
# ==========================================
LLM_MODEL_NAME = "gemini-1.5-flash"
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


def generate_rag_response(query: str, retrieved_chunks: str, chat_history: list = None) -> str:
    if "No relevant context chunks found" in retrieved_chunks or "⚠️ No active document" in retrieved_chunks:
        return f"⚠️ **Grounded Analytics Aborted:** {retrieved_chunks}"

    # Initialize Gemini model (temperature set low for accurate factual retrieval)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.2)

    # Format the last 4 messages of conversation history to prevent context window overflow
    history_str = "No previous conversation history."
    if chat_history and len(chat_history) > 0:
        recent_history = chat_history[-4:] 
        history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in recent_history])

    system_rules = (
       "You are an expert analytical document evaluator. Your job is to analyze the provided context, synthesize patterns, and form evaluations.\n\n"
        "1. FACTUAL BOUNDARY (NO INVENTION): You are strictly forbidden from inventing facts, metrics, or details not explicitly written in the context. If a fact is absent, reply: 'This detail is not present in the provided document.'\n"
        "2. ANALYTICAL PERMISSION: You ARE expected to evaluate core arguments, identify document weaknesses, and assess structural completeness. Apply logical deduction to the provided text without fabricating additional information.\n"
        "3. EVIDENCE-FIRST REASONING: For any critique or evaluation, you must first output a bulleted list of the exact quotes/facts from the text that serve as your evidence, followed by your analytical conclusion.\n"
        "4. MISSING DATA: If asked to evaluate a topic with zero supporting evidence in the text, state clearly: 'The document contains no evidence to evaluate [topic].'\n"
        "Conversation History:\n{history}\n\n"
        "Context Evidence:\n{context}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_rules),
        ("human", "{input}")
    ])

    # Chain now includes StrOutputParser to return raw text string instead of AIMessage object
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"context": retrieved_chunks, "history": history_str, "input": query})