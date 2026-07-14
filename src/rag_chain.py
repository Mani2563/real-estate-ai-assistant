from src.llm import LLMModel
from src.prompts import RAG_PROMPT
from src.retriever import Retriever
from src.memory import ConversationMemory

class RAGChain:

    def __init__(self):

        self.llm = LLMModel.get_llm()

        self.retriever = Retriever()

        self.memory = ConversationMemory()

    def ask(self, question: str):

        history = self.memory.get_history()

        docs = self.retriever.retrieve(question)

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
