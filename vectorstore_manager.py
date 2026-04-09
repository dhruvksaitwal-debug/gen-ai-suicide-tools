import chromadb
from langchain_chroma import Chroma
import re


class VectorStoreManager:
    def __init__(self, embedding_model, db_path="./db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_model = embedding_model
        self.db_path = db_path

    def create_collection(self, name, documents):
        # safe_name = self.sanitize_collection_name(name)
        store = Chroma(
            collection_name=name,
            embedding_function=self.embedding_model,
            client=self.client,
            persist_directory=self.db_path
        )
        store.add_documents(documents=documents)
        return store
    
    @staticmethod
    def sanitize_collection_name(name: str) -> str:
        # Lowercase
        name = name.lower()

        # Replace spaces with underscores
        name = name.replace(" ", "_")

        # Remove invalid characters (keep only a-z, 0-9, ., _, -)
        name = re.sub(r"[^a-z0-9._-]", "", name)

        # Ensure it starts with alphanumeric
        if not re.match(r"^[a-z0-9]", name):
            name = "c_" + name

        # Ensure it ends with alphanumeric
        if not re.match(r".*[a-z0-9]$", name):
            name = name + "0"

        # Enforce max length (Chroma allows up to 512)
        return name[:200]  # 200 is safe and readable