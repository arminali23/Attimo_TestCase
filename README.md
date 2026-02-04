# Attimo Test Case (AI - Assistant)

RAG application that answers questions using uploaded documents (PDF / TXT / MD).
Users upload documents via Streamlit, the system indexes them with embeddings in ChromaDB, retrieves the most relevant chunks for a question, and returns grounded answers with sources.

---

## Setup

### Create a virtual environment

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


