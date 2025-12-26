import os
from dotenv import load_dotenv
from itertools import count
from langchain_core.documents import Document
from embeddings_with_backoff import EmbeddingsWithBackoff
from llm_client import LLMClient
from pdf_extractor import PDFExtractor
from vectorstore_manager import VectorStoreManager
from hypo_question_generator import HypotheticalQuestionGenerator
from query_expander import QueryExpander
from rag_answerer import RAGAnswerer
from answer_normalizer import QueryScopedNormalizer, AnswerAccumulator, FinalRecordAssembler, AuditLogger

class DocRAGPipelineOrchestrator:
    def __init__(self, file_name, data_folder="Data", model_name="gpt-4o-mini"):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        # Core components
        self.llm_client = LLMClient(api_key, base_url, model_name)
        self.pdf_extractor = PDFExtractor(self.llm_client)
        self.embedding_model = EmbeddingsWithBackoff(api_key=api_key, base_url=base_url, model="text-embedding-3-small") 
        self.vector_manager = VectorStoreManager(self.embedding_model, db_path="./doc_rag_db")
        self.hypo_gen = HypotheticalQuestionGenerator(self.llm_client)
        self.query_expander = QueryExpander(self.llm_client)
        self.q_normalizer = QueryScopedNormalizer(self.llm_client, debug=True)
        self.assembler = FinalRecordAssembler()
        self.audit_logger = AuditLogger()

        # Answerer setup
        system_prompt = "You are an assistant who answers user queries using provided context only."
        user_prompt_template = "<Context>{context}</Context><Question>{question}</Question>"
        self.answerer = RAGAnswerer(self.llm_client, system_prompt, user_prompt_template)

        # File paths
        self.file_name = file_name
        self.pdf_path = os.path.join(data_folder, file_name + ".pdf")

        # State
        self.chunks_vectorstore = None
        self.hypo_vectorstore = None

    def setup(self):
        # 1. Extract PDF
        print("Extracting PDF contents...")
        extracted_contents = self.pdf_extractor.extract(self.pdf_path)

        # 2. Build vectorstore
        print("Building vectorstore...")
        documents = [Document(id=i, page_content=str(chunk)) for i, chunk in zip(count(1), extracted_contents)]
        article_collection_name = self.vector_manager.sanitize_collection_name(self.file_name + "_article")
        self.chunks_vectorstore = self.vector_manager.create_collection(article_collection_name, documents)
        self.chunked_documents = documents   # keep the Document objects

        # 3. Generate hypothetical questions
        print("Generating hypothetical questions...")
        hypo_questions = self.hypo_gen.generate(
            self.chunked_documents,   # pass Document objects, not strings
            "Generate up to 10 hypothetical questions about suicide screening/assessment tools."
        )
        hypo_collection_name = self.vector_manager.sanitize_collection_name(self.file_name + "_hypo")
        self.hypo_vectorstore = self.vector_manager.create_collection(hypo_collection_name, hypo_questions)

    def run_queries(self, queries):
        acc = AnswerAccumulator(doc_id=self.file_name)
        per_query_contexts = {}  # query -> list[str] contexts

        for q in queries:
            print(f"\nQuery: {q}")

            # --- Retrieval ---
            expanded = self.query_expander.expand(q)
            chunk_ctx, hypo_ctx = [], []
            for eq in expanded:
                chunk_ctx.extend(
                    self.chunks_vectorstore.as_retriever(
                        search_type="similarity", search_kwargs={"k": 5}
                    ).invoke(eq)
                )
                hypo_ctx.extend(
                    self.hypo_vectorstore.as_retriever(
                        search_type="similarity", search_kwargs={"k": 8}
                    ).invoke(eq)
                )

            # Deduplicate contexts to strings
            chunk_ctx = list({d.page_content for d in chunk_ctx})
            hypo_ctx = list({d.page_content for d in hypo_ctx})
            contexts = chunk_ctx + hypo_ctx

            # --- Answering ---
            chunk_answer = self.answerer.answer(q, chunk_ctx)
            hypo_answer = self.answerer.answer(q, hypo_ctx)
            final_answer = self.answerer.combine(chunk_answer, hypo_answer)
            print(f"GenAI Answer: {final_answer}")

            # --- Normalization ---
            partial = self.q_normalizer.normalize_query(q, final_answer)
            updated_fields = acc.update(q, partial)  # now uses query

            # --- Audit log ---
            self.audit_logger.log(self.file_name, q, final_answer, partial)

            # Save contexts per original query
            per_query_contexts[q] = contexts

            # Short-circuit
            if q == "Does the article study any suicide screening/assessment tools?" \
            and partial.get("studies_tool") == "no":
                records = self.assembler.assemble(self.file_name, acc)
                return self._flatten_with_alignment(records, acc.get_field_provenance(), per_query_contexts)

        # --- Assemble full record(s) ---
        records = self.assembler.assemble(self.file_name, acc)

        # --- Flatten with aligned contexts ---
        return records, acc.get_field_provenance(), per_query_contexts # self._flatten_with_alignment(records, acc.get_field_provenance(), per_query_contexts)


    def _flatten_with_alignment(self, records, field_provenance: dict, per_query_contexts: dict):
        """
        records: assembled records from FinalRecordAssembler (base_fields present)
        field_provenance: {field -> [queries]} from AnswerAccumulator
        per_query_contexts: {query -> [context strings]}
        """
        flattened = []
        for record in records:
            doc_id = record["doc_id"]
            studies_tool = record.get("studies_tool")
            tool_name = record.get("tool_name")
            tool_type = record.get("tool_type")

            for field, answer in record.items():
                if field == "doc_id":
                    continue

                # Gather contexts from all queries that contributed to this field
                contexts = []
                for q in field_provenance.get(field, []):
                    contexts.extend(per_query_contexts.get(q, []))

                # Dedup final contexts
                contexts = list(dict.fromkeys(contexts))

                flattened.append({
                    "doc_id": doc_id,
                    "studies_tool": studies_tool,
                    "tool_name": tool_name,
                    "tool_type": tool_type,
                    "question": field,   # base_field name
                    "answer": answer,    # normalized answer
                    "contexts": contexts
                })
        return flattened