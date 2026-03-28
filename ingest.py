import subprocess, tempfile, shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

EXTENSIONS = {".md", ".py"}
DB_LOCATION = "./chroma_langchain_db"
COLLECTION  = "docpilot"
EMBEDDINGS  = OllamaEmbeddings(model="mxbai-embed-large:335m")

def clone(url):
    tmp = tempfile.mkdtemp()
    subprocess.run(["git", "clone", "--depth=1", url, tmp], check=True)
    return tmp

def read_files(repo_dir):
    for path in Path(repo_dir).rglob("*"):
        if any(s in path.parts for s in [".git","node_modules","__pycache__"]):
            continue
        if path.suffix in EXTENSIONS and path.is_file():
            text = path.read_text(errors="ignore").strip()
            if text:
                yield str(path.relative_to(repo_dir)), text

def chunk(text, size=800, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return [c for c in chunks if len(c) > 30]

def ingest(github_url):
    tmp = clone(github_url)
    try:
        store = Chroma(collection_name=COLLECTION,
                       persist_directory=DB_LOCATION,
                       embedding_function=EMBEDDINGS)
        docs, ids = [], []
        for filename, content in read_files(tmp):
            for i, chunk_text in enumerate(chunk(content)):
                doc_id = f"{filename}:{i}"
                docs.append(Document(
                    page_content=chunk_text,
                    metadata={"source": filename, "repo": github_url},
                    id=doc_id
                ))
                ids.append(doc_id)
        print(f"Storing {len(docs)} chunks...")
        store.add_documents(documents=docs, ids=ids)
        print("Done!")
    finally:
        
        shutil.rmtree(tmp, ignore_errors=True)
       

if __name__ == "__main__":
    import sys
    ingest(sys.argv[1])