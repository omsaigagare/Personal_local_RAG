# Serverless RAG Document Assistant (Zero-Cost Cloud Architecture)

This repository contains a Retrieval-Augmented Generation (RAG) application designed to process PDF documents and answer user queries using context-grounded AI. It is built to be deployed entirely for free on Streamlit Community Cloud.

## The Architecture (And Engineering Trade-offs)

Initially, this project attempted to run local embedding models (`sentence-transformers`) and a local vector database. That approach failed in deployment. Streamlit Community Cloud imposes a strict ~1GB RAM memory ceiling on its free tier. Downloading PyTorch and pulling local embedding models into memory instantly crashed the container with Out-Of-Memory (OOM) errors.

To achieve a production-ready, 100% free deployment without requiring a credit card or paid cloud infrastructure, the backend was refactored to a serverless API architecture:

*   **Frontend / Hosting:** Streamlit (Community Cloud)
*   **Vector Database:** ChromaDB (Ephemeral / RAM-only to prevent stale data persistence)
*   **Embeddings:** Cohere (`embed-english-v3.0`) via `langchain-cohere`
*   **Text Generation (LLM):** Groq (`llama-3.1-8b-instant`) via `langchain-groq` for high-speed, zero-cost inference.
*   **Document Parsing:** `pypdf`
*   **Orchestration:** LangChain

## Local Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <https://github.com/omsaigagare/Personal_local_RAG>
   cd <main>