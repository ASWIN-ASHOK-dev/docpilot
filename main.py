import ollama
import numpy as np
from datetime import datetime

# Simple knowledge base about time
documents = [
    "The current time can be retrieved using datetime.now() in Python.",
    "Time zones affect how time is displayed globally.",
    "Daylight saving time shifts clocks forward or backward seasonally.",
    "Unix timestamp represents seconds since 1970-01-01 00:00:00 UTC.",
    "To get current time in UTC: datetime.utcnow()."
]

# Compute embeddings for documents
doc_embeddings = []
for doc in documents:
    response = ollama.embeddings(model="deepseek-coder-v2", prompt=doc)
    print(response)
    doc_embeddings.append(np.array(response["embedding"]))

doc_embeddings = np.array(doc_embeddings)
def retrieve(query, top_k=2):
    # Compute query embedding
    query_emb = ollama.embeddings(model="deepseek-coder-v2", prompt=query)
    query_emb = np.array(query_emb["embedding"])
    
    # Compute cosine similarities
    similarities = np.dot(doc_embeddings, query_emb) / (np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_emb))
    
    # Get top_k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return retrieved documents
    return [documents[i] for i in top_indices]

while True:
    inp = input("enter a prompt: ")
    retrieved_docs = retrieve(inp)
    context = "\n".join(retrieved_docs)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    augmented_prompt = f"Context:\n{context}\n\nCurrent time: {current_time}\n\nQuestion: {inp}"
    
    stream = ollama.chat(
        model="deepseek-coder-v2",
        messages=[{"role": "user", "content": augmented_prompt}],
        stream=True
    )

    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print()