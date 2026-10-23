from pydantic import BaseModel
import os
import pypdf
import chromadb
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
class QueryRequest(BaseModel):
    question: str

# Load local environment variables from .env
load_dotenv()

app = FastAPI(title="DocAnalyzer AI")

# CORS setup for Firebase frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WARNING: Global state. This will cause collisions if multiple users upload at the same time.
current_vector_db = None

def get_api_key(env_var: str) -> str:
    key = os.getenv(env_var)
    if not key:
        raise HTTPException(status_code=500, detail=f"Server Error: {env_var} is missing. Check .env or Render dashboard.")
    return key

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global current_vector_db
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        reader = pypdf.PdfReader(file_path)
        raw_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

        if not raw_text.strip():
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="OCR Required: The uploaded PDF contains no selectable text.")

        embedding_model = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=get_api_key("COHERE_API_KEY")
        )
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(raw_text)

        ephemeral_client = chromadb.EphemeralClient()
        current_vector_db = Chroma(
            client=ephemeral_client,
            collection_name="active_document",
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )

        current_vector_db.add_texts(texts=chunks)
        os.remove(file_path)
        
        return {"message": "Document processed successfully", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel

# Define the JSON structure FastAPI should expect
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_document(request: QueryRequest):
    global current_vector_db
    if current_vector_db is None:
        raise HTTPException(status_code=400, detail="No active document loaded in memory. Upload a PDF first.")

    try:
        # Extract the question from the validated JSON request
        question = request.question
        
        results = current_vector_db.similarity_search_with_relevance_scores(question, k=5)
        valid_chunks = [f"[Chunk {idx} | Relevance: {score:.2f}]\n{doc.page_content.strip()}" for idx, (doc, score) in enumerate(results, 1) if score >= 0.15]

        retrieved_chunks = "\n\n".join(valid_chunks) if valid_chunks else "No relevant context chunks found."

        if "No relevant context chunks found" in retrieved_chunks:
            return {"answer": "I could not find relevant information in the document to answer your question."}

        llm = ChatGoogleGenerativeAI(
            google_api_key=get_api_key("GOOGLE_API_KEY"),
            model="gemini-2.5-flash",
            temperature=0
        )
        
        # ... (keep the rest of your LLM execution code below here exactly as it was) ...

        system_rules = (
           "You are an expert analytical document evaluator.\n"
            "1. FACTUAL BOUNDARY: Do not invent facts.\n"
            "2. EVIDENCE-FIRST REASONING: Output a bulleted list of exact quotes before concluding.\n"
            "Context Evidence:\n{context}"
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_rules),
            ("human", "{input}")
        ])

        chain = prompt_template | llm | StrOutputParser()
        answer = chain.invoke({"context": retrieved_chunks, "input": question})
        
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))