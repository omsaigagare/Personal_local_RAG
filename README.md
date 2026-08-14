# ​Serverless RAG Document Assistant (Zero-Cost Cloud Architecture)
​This repository contains a Retrieval-Augmented Generation (RAG) application designed to process PDF documents and answer user queries using context-grounded AI. It is built to be deployed entirely for free using decoupled frontend and backend services.
​The Architecture (And Engineering Trade-offs)
​Initially, this project attempted to run local embedding models (sentence-transformers) and a local vector database on Streamlit Community Cloud. That approach failed because pulling local embedding models into memory instantly crashed the container with Out-Of-Memory (OOM) errors under their strict ~1GB RAM ceiling.
​To achieve a production-ready, 100% free deployment without requiring a credit card, the architecture was decoupled and migrated:
​Frontend / Hosting: Firebase Hosting (HTML, CSS, Vanilla JS)
​Backend API: Render (Python Web Service)
​Vector Database: ChromaDB (Ephemeral / RAM-only to prevent stale data persistence)
​Embeddings: Cohere (embed-english-v3.0) via langchain-cohere
​Text Generation (LLM): Groq (llama-3.1-8b-instant) via langchain-groq for high-speed, zero-cost inference.
​Document Parsing: pypdf
​Orchestration: LangChain
​Note on Latency: Because the Python backend runs on Render's free tier, the web service spins down after 15 minutes of inactivity. The first API request after a period of inactivity will experience a 30–60 second cold start delay while the container wakes up.

​Local Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/omsaigagare/Personal_local_RAG
cd Personal_local_RAG
