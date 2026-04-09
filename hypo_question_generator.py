from langchain_core.documents import Document

class HypotheticalQuestionGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate(self, documents, system_message):
        questions = []
        for doc in documents:
            response = self.llm_client.chat_completion([
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"<Document>{doc.page_content}</Document>"}
            ])
            for q in response.split("\n"):
                questions.append(Document(page_content=q, metadata={"parent_chunk_id": doc.id}))
        return questions