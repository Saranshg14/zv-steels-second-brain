# ZV Steels — Second Brain

AI-powered knowledge retrieval engine and Second Brain for ZV Steels, built with ChromaDB semantic search, local embeddings, and Groq/OpenAI compatible LLM inference. Includes role-based access control and web UI.

---

## ⚡ Features

- **Semantic Search & Retrieval**: Powered by ChromaDB and `BAAI/bge-small-en-v1.5` embeddings.
- **Role-Based Access Control**: Filter answers by role (`employee`, `manager`, `founder`).
- **Semantic Caching**: Zero-cost immediate answers for identical or near-identical questions.
- **Interactive Web Interface**: Modern Flask UI for asking questions and ingesting notes.
- **Knowledge Questionnaire**: Comprehensive structured questionnaire for capturing organizational knowledge.

---

## 🔑 API Keys & Configuration

> **Note on API Keys:**
> API keys are **NOT** hardcoded in the codebase.
> They are loaded dynamically at runtime from environment variables using `python-dotenv`.

### Setting Up Your API Key

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill in your Groq or xAI API key:
   ```env
   XAI_API_KEY=your_actual_api_key_here
   GROK_MODEL=llama-3.3-70b-versatile
   EMBED_MODEL=BAAI/bge-small-en-v1.5
   TOP_K=5
   MAX_DISTANCE=0.55
   CACHE_DISTANCE=0.08
   ```

*(The `.env` file is protected by `.gitignore` to prevent secret leakage).*

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ingest Knowledge Notes
```bash
python brain.py ingest
```

### 3. Run the Web App
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

### 4. CLI Usage (Optional)
```bash
# Ask a question as employee
python brain.py ask "What is our shipping policy?" --role employee

# Ask as manager
python brain.py ask "What is our margin rule?" --role manager

# Interactive chat as founder
python brain.py chat --role founder
```
