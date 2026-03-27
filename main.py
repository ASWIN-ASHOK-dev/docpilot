from langchain_ollama.llms import OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = Ollama(model="deepseek-coder-v2")

template = """
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}
Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model
while True:
    question = input("Enter a question about the restaurant: ")
    reviews = retriever.invoke(question)
    print("Retrieved reviews:",reviews)
    reviews_text = "\n".join(f"[ID: {doc.id}] {doc.page_content}" for doc in reviews)
    result = chain.invoke({"reviews": reviews_text, "question": question})
    print(result,"\n \n")