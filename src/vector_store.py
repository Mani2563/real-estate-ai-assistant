from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import VECTOR_STORE_DIR
from src.embeddings import EmbeddingModel


class VectorStore:

    def __init__(self):

        self.embeddings = EmbeddingModel.get_embeddings()

        self.store_path = Path(VECTOR_STORE_DIR)

        self.store_path.mkdir(
            parents=True,
            exist_ok=True
        )

    def create(
        self,
        documents: list[Document]
    ) -> FAISS:

        vector_store = FAISS.from_documents(
            documents,
            self.embeddings
        )

        return vector_store

    def save(
        self,
        vector_store: FAISS
    ) -> None:

        vector_store.save_local(
            str(self.store_path)
        )

    def load(self) -> FAISS:

        return FAISS.load_local(
            str(self.store_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
