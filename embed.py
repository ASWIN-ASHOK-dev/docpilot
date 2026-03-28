from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import hashlib
import store


def _is_context_length_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "context" in msg and "length" in msg
    ) or "input length" in msg or "too many tokens" in msg


def _split_text_by_chars(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    step = max(1, max_chars - overlap)
    start = 0
    while start < len(text):
        part = text[start:start + max_chars].strip()
        if part:
            parts.append(part)
        start += step
    return parts


def _safe_add_documents(batch_docs: list[Document], batch_ids: list[str]) -> tuple[int, int]:
    """Try batch insert first; on context overflow, split individual docs and retry."""
    try:
        vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
        return len(batch_docs), 0
    except Exception as e:
        if not _is_context_length_error(e):
            raise

    inserted = 0
    skipped = 0
    for doc, doc_id in zip(batch_docs, batch_ids):
        queue: list[tuple[str, str]] = [(doc.page_content, doc_id)]
        while queue:
            text, current_id = queue.pop(0)
            try:
                vectorstore.add_documents(
                    documents=[Document(page_content=text, metadata=doc.metadata, id=current_id)],
                    ids=[current_id],
                )
                inserted += 1
                continue
            except Exception as inner_exc:
                if not _is_context_length_error(inner_exc):
                    raise

                # Keep splitting until chunks are small enough; skip pathological inputs.
                if len(text) <= 220:
                    skipped += 1
                    print(f"Skipped chunk due to context limit: {current_id}")
                    continue

                next_size = max(220, len(text) // 2)
                overlap = min(80, max(20, next_size // 5))
                sub_parts = _split_text_by_chars(text, max_chars=next_size, overlap=overlap)
                if len(sub_parts) <= 1:
                    skipped += 1
                    print(f"Skipped chunk due to repeated overflow: {current_id}")
                    continue

                for idx, sub in enumerate(sub_parts):
                    queue.append((sub, f"{current_id}-s{idx}"))

    return inserted, skipped

def _print_progress(label: str, current: int, total: int, width: int = 28) -> None:
    if total <= 0:
        return
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if current >= total else "\r"
    print(f"{label}: [{bar}] {current}/{total}", end=end, flush=True)

embeddings = OllamaEmbeddings(model=store.load_config().get("default_embed_model", "mxbai-embed-large:335m"))
db_location =  store.path / "chroma_langchain_db"

vectorstore = Chroma(
    collection_name="documents",
    persist_directory=db_location,
    embedding_function=embeddings
)


def chunk_text(
    text: str,
    chunk_size: int = 120,
    overlap: int = 20,
    max_chars: int = 1200,
) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size]).strip()
        if len(chunk) > max_chars:
            chunk = chunk[:max_chars].rsplit(" ", 1)[0].strip() or chunk[:max_chars]
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks

def embed_texts(texts: list[str], source: str = "web", batch_size: int = 32):
    documents = []
    ids = []
    chunk_counter = 0
    seen_chunks = set()  # Track chunk content to avoid duplicates
    for text in texts:
        for chunk in chunk_text(text):
            # Skip duplicate chunks
            if chunk in seen_chunks:
                continue
            seen_chunks.add(chunk)
            
            doc_id = hashlib.md5(f"{source}:{chunk_counter}:{chunk}".encode()).hexdigest()
            doc = Document(
                page_content=chunk+"source"+source,
                metadata={"source": source},
                id=doc_id
            )
            documents.append(doc)
            ids.append(doc_id)
            chunk_counter += 1

    if not documents:
        print(f"No chunks to embed from {source}")
        return

    total_batches = (len(documents) + batch_size - 1) // batch_size
    inserted_total = 0
    skipped_total = 0
    for batch_num, i in enumerate(range(0, len(documents), batch_size), start=1):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        inserted, skipped = _safe_add_documents(batch_docs, batch_ids)
        inserted_total += inserted
        skipped_total += skipped
        _print_progress("Embedding batches", batch_num, total_batches)

    print(f"Embedded {inserted_total} chunks from {source} (skipped {skipped_total})")

retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
