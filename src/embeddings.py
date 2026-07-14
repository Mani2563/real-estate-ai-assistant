from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

from config import EMBEDDING_MODEL


class EmbeddingModel:

    @staticmethod
    def get_embeddings():

        st.write(f"Loading embedding model: {EMBEDDING_MODEL}")

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        st.success("Embedding model loaded!")

        return embeddings
