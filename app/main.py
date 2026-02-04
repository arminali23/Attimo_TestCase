import time
import streamlit as st
from core.logging import setup_logger
from rag.ingest import ingest_file
from rag.index import reset_index, add_chunks, query_chunks
from rag.llm import grounded_answer

def run():
    logger = setup_logger("app")

    st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")
    st.title("AI Knowledge Assistant (RAG)")
    st.caption("Upload documents (pdf/txt/md) and ask questions grounded in those documents.")

    if "loaded_files" not in st.session_state:
        st.session_state.loaded_files = []

    if "ingested" not in st.session_state:
        st.session_state.ingested = False

    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""

    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []

    if "last_latency" not in st.session_state:
        st.session_state.last_latency = None

    with st.sidebar:
        st.header("Documents")

        uploaded = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            key="uploader",
        )

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("Ingest", use_container_width=True):
                if not uploaded:
                    st.error("Please upload at least one document.")
                else:
                    try:
                        reset_index()
                        st.session_state.loaded_files = []
                        total_chunks = 0

                        with st.spinner("Ingesting documents..."):
                            for f in uploaded:
                                chunks = ingest_file(f.name, f.read())
                                added = add_chunks(chunks)
                                total_chunks += added
                                st.session_state.loaded_files.append(f.name)

                        st.session_state.ingested = True
                        st.success(f"Ingestion complete. Total chunks indexed: {total_chunks}")

                    except Exception as e:
                        st.session_state.ingested = False
                        logger.exception(f"Ingestion failed: {e}")
                        st.error(f"Ingestion failed: {e}")

        with col_b:
            if st.button("Reset", use_container_width=True):
                try:
                    reset_index()
                    st.session_state.loaded_files = []
                    st.session_state.ingested = False
                    st.session_state.last_answer = ""
                    st.session_state.last_sources = []
                    st.session_state.last_latency = None
                    st.success("Reset complete.")
                except Exception as e:
                    logger.exception(f"Reset failed: {e}")
                    st.error(f"Reset failed: {e}")

        st.divider()

        st.subheader("Loaded documents")
        if st.session_state.loaded_files:
            for name in st.session_state.loaded_files:
                st.write(f"- {name}")
        else:
            st.info("No documents loaded yet.")

    st.subheader("Ask a question")
    question = st.text_input("Question", placeholder="e.g., What does the document say about ...?")
    ask = st.button("Ask", type="primary")

    if ask:
        if not question.strip():
            st.error("Please enter a question.")
        elif not st.session_state.ingested:
            st.error("Please ingest documents first.")
        else:
            t0 = time.perf_counter()
            try:
                with st.spinner("Retrieving and answering..."):
                    t_retr = time.perf_counter()
                    hits = query_chunks(question)
                    retr_s = time.perf_counter() - t_retr

                    t_ans = time.perf_counter()
                    answer, citations, llm_latency = grounded_answer(question, hits)
                    ans_s = time.perf_counter() - t_ans

                total_latency = time.perf_counter() - t0
                st.caption(f"Retrieval: {retr_s:.2f}s | Answer: {ans_s:.2f}s")
                st.session_state.last_answer = answer
                st.session_state.last_sources = hits
                st.session_state.last_latency = total_latency

            except Exception as e:
                logger.exception(f"Ask failed: {e}")
                st.error(f"Failed to answer: {e}")

    if st.session_state.last_answer:
        st.divider()
        st.subheader("Answer")
        st.write(st.session_state.last_answer)

        if st.session_state.last_latency is not None:
            st.caption(f"Inference time: {st.session_state.last_latency:.2f}s")

        st.subheader("Sources")
        hits = st.session_state.last_sources

        if hits:
            for chunk, score in hits:
                snippet = chunk.text[:220].replace("\n", " ")
                st.markdown(
                    f"- **{chunk.source}** | chunk `{chunk.chunk_id}` | score `{score:.3f}`  \n"
                    f"  _{snippet}..._"
                )
        else:
            st.info("No sources retrieved.")