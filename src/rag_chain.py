import streamlit as st

from src.llm import LLMModel
from src.prompts import RAG_PROMPT
from src.retriever import Retriever
from src.memory import ConversationMemory


class RAGChain:

    def __init__(self):

        try:
            st.info("🔹 Step 1: Initializing LLM...")
            self.llm = LLMModel.get_llm()
            st.success("✅ LLM Loaded")
        except Exception as e:
            st.exception(e)
            raise

        try:
            st.info("🔹 Step 2: Initializing Retriever...")
            self.retriever = Retriever()
            st.success("✅ Retriever Loaded")
        except Exception as e:
            st.exception(e)
            raise

        try:
            st.info("🔹 Step 3: Initializing Memory...")
            self.memory = ConversationMemory()
            st.success("✅ Memory Loaded")
        except Exception as e:
            st.exception(e)
            raise

        st.success("🎉 RAGChain Initialized Successfully")

    def ask(self, question: str):

        history = self.memory.get_history()

        docs = self.retriever.retrieve(question)

        if not docs:
            answer = (
                "I couldn't find enough relevant information in the provided knowledge base "
                "for that question. I can help with real-estate topics such as properties, "
                "builders, pricing, amenities, policies, and legal documents."
            )

            self.memory.add_user_message(question)
            self.memory.add_ai_message(answer)

            return {
                "answer": answer,
                "sources": []
            }

        context = ""
        sources = []

        for doc in docs:
            context += (
                f"Source: {doc.metadata['source']}\n\n"
                f"{doc.page_content}\n\n"
                "----------------------------------\n\n"
            )
            sources.append(doc.metadata["source"])

        prompt = RAG_PROMPT.invoke(
            {
                "history": history,
                "context": context,
                "question": question,
            }
        )

        response = self.llm.invoke(prompt)

        self.memory.add_user_message(question)
        self.memory.add_ai_message(response.content)

        return {
            "answer": response.content,
            "sources": list(set(sources))
        }
