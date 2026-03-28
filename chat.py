from langchain_ollama.llms import OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate
from embed import retriever
import store

config = store.load_config()
model = Ollama(model=config.get("default_model", "deepseek-coder-v2"))
template = """
You are an expert in answering questions about a pizza restaurant
q
Here are some relevant reviews: {reviews}
Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


def _build_bounded_context(docs, max_total_chars: int = 6000, max_doc_chars: int = 1200) -> str:
    lines = []
    total = 0
    for doc in docs:
        doc_text = doc.page_content.strip()
        if len(doc_text) > max_doc_chars:
            doc_text = doc_text[:max_doc_chars].rsplit(" ", 1)[0].strip() or doc_text[:max_doc_chars]
        line = f"[ID: {doc.id}] {doc_text}"
        if total + len(line) > max_total_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)

def askai(question):
    rag = retriever.invoke(question)
    rag_text = "\n".join(f"[{doc.metadata['source']}]\n{doc.page_content}" for doc in rag)
    result = chain.invoke({"reviews": rag_text, "question": question})
    print(result, "\n\n")

if __name__ == "__main__":
    while True:
        question = input("Ask a question (or type 'exit'): ")
        if question.lower() == "exit":
            break
        askai(question)
