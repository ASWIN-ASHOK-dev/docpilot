from langchain_ollama.llms import OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

vectorstore = Chroma(
    collection_name="docpilot",
    persist_directory="./chroma_langchain_db",
    embedding_function=OllamaEmbeddings(model="mxbai-embed-large:335m")
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

model = Ollama(model="llama3.2")

template = """
You are a helpful assistant that answers questions using documentation.
Use only the context below to answer. If you don't know, say so.

Context:
{reviews}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question = input("Enter a question (or type 'exit'): ")
    if question.lower() == "exit":
        break
    docs = retriever.invoke(question)
    reviews_text = "\n".join(f"[{doc.metadata['source']}]\n{doc.page_content}" for doc in docs)
    result = chain.invoke({"reviews": reviews_text, "question": question})
    print(result, "\n\n")