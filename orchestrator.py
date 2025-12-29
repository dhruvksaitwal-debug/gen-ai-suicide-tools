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
        per_query_contexts = []
        query_order = []

        for q in queries:
            query_order.append(q)
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
            updated_fields = acc.update(q, partial)

            # --- Audit log ---
            self.audit_logger.log(self.file_name, q, final_answer, partial)

            # --- Append contexts in order ---
            per_query_contexts.append(contexts)

            # Short-circuit
            if q == "Does the article study any suicide screening/assessment tools?" \
            and partial.get("studies_tool") == "no":
                records = self.assembler.assemble(self.file_name, acc)
                flattened = self._flatten_with_alignment(records, acc.get_field_provenance(), per_query_contexts, query_order, FinalRecordAssembler.BASE_FIELDS)
                return flattened

        # --- Assemble full record(s) ---
        records = self.assembler.assemble(self.file_name, acc)

        # --- Flatten with aligned contexts ---
        flattened = self._flatten_with_alignment(records, acc.get_field_provenance(), per_query_contexts, query_order, FinalRecordAssembler.BASE_FIELDS)
        return flattened
    

    def _flatten_with_alignment(self, records, field_provenance, per_query_contexts, query_order, base_fields):
        """
        Clean, deterministic flattening:
        - One row per (record, field)
        - Contexts aligned by query index
        - No duplication
        - No context leakage
        - Works for unspecified_tool and no-tool cases
        """

        flattened = []
        query_to_index = {q: i for i, q in enumerate(query_order)}

        # Flatten each record
        for record in records:
            doc_id = record["doc_id"]
            studies_tool = record.get("studies_tool")
            tool_name = record.get("tool_name")
            tool_type = record.get("tool_type")

            for field in base_fields:
                answer = record.get(field)
                if answer is None:
                    continue
                if field == "doc_id":
                    continue

                # Gather contexts ONLY from queries that contributed to this field
                contexts = []
                contributing_queries = field_provenance.get(field, [])

                for q in contributing_queries:
                    idx = query_to_index.get(q)
                    if idx is not None and idx < len(per_query_contexts):
                        contexts.extend(per_query_contexts[idx])

                # Deduplicate contexts
                contexts = list(dict.fromkeys(contexts))

                # Append clean row
                flattened.append({
                    "doc_id": doc_id,
                    "studies_tool": studies_tool,
                    "tool_name": tool_name,
                    "tool_type": tool_type,
                    "question": field,
                    "answer": answer,
                    "contexts": contexts,
                })

        return flattened