from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma

# you can also use all-MiniLM-L12-v2 
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# the file structure has been modified to load each category as separate document
loader = JSONLoader(file_path="./menu.json", jq_schema=".data[]", text_content=False)
documents = loader.load()

db = Chroma.from_documents(documents, embedding_function)
query = "burgers ?"
docs = db.similarity_search(query)

# can print n number of docs
print(docs[0].page_content)
