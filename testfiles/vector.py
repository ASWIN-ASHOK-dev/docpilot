from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import os
import pandas as pd

df = pd.read_csv("reviews_numbered.tsv", delimiter='\t')
embeddings = OllamaEmbeddings(model="mxbai-embed-large:335m")

db_location = "./chroma_langchain_db"
add_documents = True
#add_documents = not os.path.exists(db_location)

# Create or load the vector store
vectorstore = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)
'''if add_documents:
    documents = []
    ids = []
    for row in df.itertuples(index=False):
        doc_id = str(row.Rownumber)
        document = Document(
            page_content=row.Review + " " + str(row.Liked),
            metadata={"Review": row.Review, "Liked": row.Liked},
            id=doc_id
        )
        documents.append(document)
        ids.append(doc_id)

    vectorstore.add_documents(documents=documents, ids=ids)
'''
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})