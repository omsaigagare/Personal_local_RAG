# 📚 RAG Document Assistant

A cloud-ready Retrieval-Augmented Generation (RAG) web application built to analyze and extract information from dense PDF documents using **Streamlit**, **LangChain**, **ChromaDB**, and **Google Gemini**.

## 🚀 Architecture & Features
- **Cloud-Powered LLM**: Integrates `gemini-1.5-flash` via Google Generative AI for rapid, high-context document reasoning.
- **In-Memory Vector Search**: Utilizes `chromadb.EphemeralClient` for fast, temporary session storage, ensuring the vector space resets cleanly on every run without stale data bugs.
- **Generalized Document Ingestion**: Capable of processing diverse PDF formats (reports, textbooks, contracts) beyond standard resumes.
- **Glass-Box Inspection**: Includes UI expanders to view the exact retrieved vector chunks, ensuring the AI's answers are grounded in the provided text.

## 🛠️ Prerequisites
1. **Python 3.9+** installed.
2. A **Google Gemini API Key** (Get one from [Google AI Studio](https://aistudio.google.com/)).

## ⚙️ Local Setup & Execution

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/Personal_local_RAG.git](https://github.com/omsaigagare/Personal_local_RAG)
   cd Personal_local_RAG