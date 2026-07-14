from pathlib import Path
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import VECTOR_STORE_DIR
from src.embeddings import EmbeddingModel


class VectorStore:

    def __init__(self):

        st.write("🔸 Creating Embeddings...")
        self.embeddings = EmbeddingModel.get_embeddings()
        st.success("✅ Embeddings Loaded")

        self.store_path = Path(VECTOR_STORE_DIR)

        self.store_path.mkdir(
            parents=True,
            exist_ok=True
        )

    def create(self, documents: list[Document]) -> FAISS:

        vector_store = FAISS.from_documents(
            documents,
            self.embeddings
        )

        return vector_store

    def save(self, vector_store: FAISS) -> None:

        vector_store.save_local(str(self.store_path))

    def load(self) -> FAISS:

        st.write("🔸 Loading FAISS Index...")

        db = FAISS.load_local(
            str(self.store_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        st.success("✅ FAISS Loaded")

        return db
