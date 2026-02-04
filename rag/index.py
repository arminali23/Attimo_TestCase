from typing import List, Tuple
import chromadb
from fastembed import TextEmbedding
from core.config import settings
from core.logging import setup_logger
from rag.schemas import Chunk

logger = setup_logger("index")
_embedder = TextEmbedding("BAAI/bge-small-en-v1.5")

def get_collection():
    client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    return client.get_or_create_collection(
        name=settings.COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

def reset_index():
    client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    try:
        client.delete_collection(settings.COLLECTION)
    except Exception:
        pass

    client.get_or_create_collection(
        name=settings.COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Index reset complete")

def _embed_texts(texts: List[str]) -> List[List[float]]:
    return [vec.tolist() for vec in _embedder.embed(texts)]

def add_chunks(chunks: List[Chunk]) -> int:
    if not chunks:
        return 0

    col = get_collection()

    ids = []
    docs = []
    metas = []

    for c in chunks:
        ids.append(f"{c.source}_{c.chunk_id}")
        docs.append(c.text)
        metas.append({"source": c.source, "chunk_id": int(c.chunk_id)})

    embeddings = _embed_texts(docs)
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    logger.info(f"Added {len(ids)} chunks")
    return len(ids)

def query_chunks(question: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
    
    col = get_collection()
    q_emb = _embed_texts([question])[0]
    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        score = max(0.0, 1.0 - float(dist))
        out.append((Chunk(text=doc, source=meta["source"], chunk_id=int(meta["chunk_id"])), score))

    return out