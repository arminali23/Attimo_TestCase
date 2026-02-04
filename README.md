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
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Environment variables (optional LLM)
```bash
Create a .env file:
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
TOP_K=3
MAX_CONTEXT_CHARS=6500
LLM_TIMEOUT_SECONDS=4
CHROMA_DIR=./data/chroma
COLLECTION=docs
```
If the OpenAI key has no quota, the app still works using retrieval-only excerpts.

#Run
```bash
streamlit run main.py
```
### How Ingestion Works

	1.	User uploads PDF / TXT / MD files from the Streamlit sidebar.
	2.	Documents are parsed:
	•	PDF: page-by-page text extraction
	•	TXT / MD: UTF-8 decode (fallback latin-1)
	3.	Text is split into overlapping chunks.
	4.	Each chunk is embedded locally (fastembed).
	5.	Chunks are stored in ChromaDB with metadata:
	•	source (filename)
	•	chunk_id (integer)

Ingestion happens once per upload and is persisted locally

### How Retrieval Works
	1.	The user question is embedded using the same local embedding model.
	2.	ChromaDB performs similarity search.
	3.	Top-K relevant chunks (default K=3) are retrieved.
	4.	Retrieved chunks are displayed as sources in the UI (file + chunk id + snippet).

⸻

### Prompt Approach

Retrieved chunks are provided as context to the LLM with explicit chunk headers.

Rules enforced:
	•	Answer ONLY using retrieved context
	•	If information is missing, respond:
“I don’t know based on the uploaded documents.”
	•	Always include sources

If the LLM is unavailable (quota/timeout/ auth), the system falls back to deterministic retrieval-only excerpts.

This guarantees grounded answers and avoids hallucinations.

⸻

### Performance Strategy (< 5 seconds)

To ensure inference stays under 5 seconds:
	•	Local fast embeddings using fastembed (no heavy models at query time)
	•	Retrieval limited to top 3 chunks
	•	Context size capped (MAX_CONTEXT_CHARS)
	•	Persistent Chroma index (no re-ingestion per query)

Measured Performance (local machine)

Using a single academic PDF (~26 pages, ~100+ chunks):
	•	Ingestion (one-time): ~3 seconds
	•	Retrieval + answer: ~3.2 seconds average
	•	End-to-end inference: consistently under 5 seconds

⸻

### Reliability & Engineering
	•	Modular structure: core/, rag/, app/
	•	Input validation (empty question, missing documents)
	•	Error handling for:
	•	LLM quota
	•	timeouts
	•	connection issues
	•	Retrieval-only fallback when LLM fails
	•	Basic logging for ingestion, indexing, and querying

⸻

### Limitations and Future Improvements

Limitations
	•	No OCR for scanned PDFs
	•	No conversation history
	•	Retrieval quality depends on chunking and embeddings

Future Improvements
	•	OCR support (e.g., Tesseract)
	•	Cross-encoder reranking
	•	Streaming responses
	•	Conversation memory
	•	Deployment (Docker / Streamlit Cloud)



