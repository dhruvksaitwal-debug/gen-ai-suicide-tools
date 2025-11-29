import chromadb
from langchain_chroma import Chroma

class VectorStoreManager:
    def __init__(self, embedding_model, db_path="./db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_model = embedding_model
        self.db_path = db_path

    def create_collection(self, name, documents):
        store = Chroma(
            collection_name=name,
            embedding_function=self.embedding_model,
            client=self.client,
            persist_directory=self.db_path
        )
        store.add_documents(documents=documents)
        return store