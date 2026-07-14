from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL



class EmbeddingModel:
    """
    Creates and returns the embedding model.
    """

    @staticmethod
    def get_embeddings() -> HuggingFaceEmbeddings:

        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )
