# Attimo Test Case (AI - Assistant)

RAG application that answers questions using uploaded documents (PDF / TXT / MD).
Users upload documents via Streamlit, the system indexes them with embeddings in ChromaDB, retrieves the most relevant chunks for a question, and returns grounded answers with sources.

---

## Features

- Upload documents (pdf/txt/md)
- Ingestion status + loaded document list
- Chunking + embeddings + similarity search (ChromaDB)
- Question answering with sources (file + chunk id + snippet)
- Reset / re-upload support
- Local fast embeddings for performance
- LLM integration with fallback (retrieval-only if quota unavailable)
- Basic logging and error handling

### Measured Performance (local machine)

Using a single academic PDF (~26 pages, ~100+ chunks):

- Ingestion (indexing): ~3 seconds (one-time cost)
- Retrieval + answer: ~3.2 seconds average
- End-to-end inference (Ask → Answer): consistently under 5 seconds

Performance was achieved by:
- Using fast local embeddings (fastembed)
- Limiting retrieval to the top 3 chunks
- Keeping context size bounded
- Avoiding re-ingestion on every query

### Setup

Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate

2. Install Dependencies
pip install -r requirements.txt

3. Environment variables (optional LLM)

Create a .env file:
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
TOP_K=3
MAX_CONTEXT_CHARS=6500
LLM_TIMEOUT_SECONDS=4
CHROMA_DIR=./data/chroma
COLLECTION=docs

If the OpenAI key has no quota, the app still works using retrieval-only excerpts.

#Run

streamlit run main.py


