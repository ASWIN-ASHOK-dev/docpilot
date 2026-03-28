from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import hashlib

def _print_progress(label: str, current: int, total: int, width: int = 28) -> None:
    if total <= 0:
        return
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if current >= total else "\r"
    print(f"{label}: [{bar}] {current}/{total}", end=end, flush=True)

embeddings = OllamaEmbeddings(model="mxbai-embed-large:335m")
db_location = "./chroma_langchain_db"

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
    for text in texts:
        for chunk in chunk_text(text):
            doc_id = hashlib.md5(chunk.encode()).hexdigest()
            doc = Document(
                page_content=chunk,
                metadata={"source": source},
                id=doc_id
            )
            documents.append(doc)
            ids.append(doc_id)

    if not documents:
        print(f"No chunks to embed from {source}")
        return

    total_batches = (len(documents) + batch_size - 1) // batch_size
    for batch_num, i in enumerate(range(0, len(documents), batch_size), start=1):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
        _print_progress("Embedding batches", batch_num, total_batches)

    print(f"Embedded {len(documents)} chunks from {source}")

retriever = vectorstore.as_retriever(search_kwargs={"k": 20})