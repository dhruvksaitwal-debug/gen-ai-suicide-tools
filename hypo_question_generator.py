import concurrent.futures
import logging

from langchain_core.documents import Document

from constants import DEFAULT_MAX_WORKERS

logger = logging.getLogger(__name__)


class HypotheticalQuestionGenerator:
    """Generates hypothetical questions per document chunk, for HyDE-style retrieval."""

    def __init__(self, llm_client, max_workers: int = DEFAULT_MAX_WORKERS):
        self.llm_client = llm_client
        self.max_workers = max_workers

    def _generate_for_doc(self, doc: Document, system_message: str) -> list[Document]:
        try:
            response = self.llm_client.chat_completion([
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"<Document>{doc.page_content}</Document>"}
            ])
        except Exception as e:
            logger.warning("Failed to generate hypothetical questions for chunk %s: %s", doc.id, e, exc_info=True)
            return []
        return [
            Document(page_content=q, metadata={"parent_chunk_id": doc.id})
            for q in response.split("\n") if q.strip()
        ]

    def generate(self, documents: list[Document], system_message: str) -> list[Document]:
        """
        Return one Document per generated question, tagged with its parent chunk id.
        Each source document is independent, so generation is dispatched to a thread pool.
        """
        questions: list[Document] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._generate_for_doc, doc, system_message) for doc in documents]
            for future in futures:
                questions.extend(future.result())
        return questions