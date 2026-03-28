from langchain_ollama.llms import OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate
from embed import retriever
import store

config = store.load_config()
model = Ollama(model=config.get("default_model", "deepseek-coder-v2"))

template = """
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}
Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

def askai(question):
    rag = retriever.invoke(question)
    rag_text = "\n".join(f"[ID: {doc.id}] {doc.page_content}" for doc in rag)
    result = chain.invoke({"reviews": rag_text, "question": question + "always end with source whenever possible"})
    print(result,"\n \n")