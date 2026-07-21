# 📚 Local RAG Resume Evaluator

A fully local, privacy-first Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **LangChain**, **ChromaDB**, and **Ollama (Llama 3.2)** to evaluate PDF resumes without leaking data to cloud providers.

## 🚀 Features
- **In-Memory Vector Search**: Uses `chromadb.EphemeralClient` to eliminate stale disk database bugs.
- **Strict Grounding**: System prompts force Llama 3.2 to answer strictly from retrieved context.
- **Glass-Box Inspection**: View retrieved vector chunks and relevance scores directly in the UI.

## 🛠️ Prerequisites
1. Install [Ollama](https://ollama.com/).
2. Pull the Llama 3.2 model:
   ```bash
   ollama pull llama3.2