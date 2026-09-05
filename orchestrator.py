import logging
import os
from itertools import count

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cache import content_fingerprint, file_fingerprint, load_or_compute
from config import load_openai_config
from constants import (
    CHROMA_DB_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_MODEL_NAME,
    MIN_CHUNK_CHARS_FOR_HYPO,
    MIN_RELEVANCE_SCORE,
)
from embeddings_with_backoff import EmbeddingsWithBackoff
from llm_client import LLMClient
from pdf_extractor import PDFExtractor
from vectorstore_manager import VectorStoreManager, chroma_lock
from hypo_question_generator import HypotheticalQuestionGenerator
from query_expander import QueryExpander
from rag_answerer import RAGAnswerer
from answer_normalizer import QueryScopedNormalizer, AnswerAccumulator, FinalRecordAssembler, AuditLogger

logger = logging.getLogger(__name__)


class _DocIdLoggerAdapter(logging.LoggerAdapter):
    """Prefixes log messages with the doc_id, so interleaved concurrent-PDF logs stay legible."""

    def process(self, msg, kwargs):
        return f"[{self.extra['doc_id']}] {msg}", kwargs


class DocRAGPipelineOrchestrator:
    def __init__(
        self,
        file_name: str,
        data_folder: str = "Data",
        model_name: str = DEFAULT_MODEL_NAME,
        use_cache: bool = True,
    ):
        self.use_cache = use_cache
        config = load_openai_config()
        api_key, base_url = config.api_key, config.base_url

        self.logger = _DocIdLoggerAdapter(logger, {"doc_id": file_name})

        # Core components
        self.llm_client = LLMClient(api_key, base_url, model_name)
        self.pdf_extractor = PDFExtractor(self.llm_client)
        self.embedding_model = EmbeddingsWithBackoff(api_key=api_key, base_url=base_url, model="text-embedding-3-small")
        self.vector_manager = VectorStoreManager(self.embedding_model, db_path=CHROMA_DB_PATH)
        self.hypo_gen = HypotheticalQuestionGenerator(self.llm_client)
        self.query_expander = QueryExpander(self.llm_client)
        self.q_normalizer = QueryScopedNormalizer(self.llm_client)
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


    def setup(self) -> None:
        """Extract the PDF and build the chunk + hypothetical-question vectorstores."""
        # 1. Extract PDF (cached: table/image LLM calls are expensive and the PDF rarely changes)
        self.logger.info("Extracting PDF contents...")
        extracted_contents = load_or_compute(
            self.file_name, "extraction", file_fingerprint(self.pdf_path),
            lambda: self.pdf_extractor.extract(self.pdf_path),
            force=not self.use_cache,
        )

        # 2. Split into retrieval-sized chunks (a whole page as one chunk is too coarse:
        # it dilutes the embedding signal and bloats every downstream answer-call prompt)
        self.logger.info("Building vectorstore...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        split_docs = splitter.create_documents([str(c) for c in extracted_contents])
        documents = [Document(id=i, page_content=d.page_content) for i, d in zip(count(1), split_docs)]
        article_collection_name = self.vector_manager.sanitize_collection_name(self.file_name + "_article")
        self.chunks_vectorstore = self.vector_manager.create_collection(article_collection_name, documents)
        self.chunked_documents = documents   # keep the Document objects
        self._chunk_content_by_id = {d.id: d.page_content for d in documents}

        # 3. Generate hypothetical questions (cached; skip chunks too short to carry signal)
        self.logger.info("Generating hypothetical questions...")
        hypo_source_docs = [d for d in documents if len(d.page_content.strip()) >= MIN_CHUNK_CHARS_FOR_HYPO]
        hypo_fingerprint = content_fingerprint(*(d.page_content for d in hypo_source_docs))

        def _generate_hypo_questions():
            docs = self.hypo_gen.generate(
                hypo_source_docs,
                "Generate up to 10 hypothetical questions about suicide screening/assessment tools."
            )
            return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]

        hypo_raw = load_or_compute(
            self.file_name, "hypo_questions", hypo_fingerprint,
            _generate_hypo_questions,
            force=not self.use_cache,
        )
        hypo_questions = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in hypo_raw]
        hypo_collection_name = self.vector_manager.sanitize_collection_name(self.file_name + "_hypo")
        self.hypo_vectorstore = self.vector_manager.create_collection(hypo_collection_name, hypo_questions)

    def _search_above_relevance_floor(self, store: Chroma, embedding: list[float], k: int) -> list[Document]:
        """
        Retrieve top-k matches and drop any below MIN_RELEVANCE_SCORE, using langchain_chroma's
        own distance-metric-aware normalization function (works whether the collection was
        built with L2 or cosine distance). Falls back to unfiltered top-k if that (private,
        version-pinned) API is ever unavailable.
        """
        with chroma_lock:
            results = store.similarity_search_by_vector_with_relevance_scores(embedding, k=k)
        try:
            score_fn = store._select_relevance_score_fn()
        except Exception as e:
            self.logger.warning("Relevance-score normalization unavailable (%s); skipping the floor.", e)
            return [doc for doc, _ in results]
        return [doc for doc, raw_score in results if score_fn(raw_score) >= MIN_RELEVANCE_SCORE]

    def run_queries(self, queries: list[str]) -> list[dict]:
        """Run each query through retrieval, answering, and normalization; return flattened records."""
        acc = AnswerAccumulator(doc_id=self.file_name)
        per_query_contexts = []
        query_order = []

        for q in queries:
            query_order.append(q)
            self.logger.info("Query: %s", q)

            # --- Retrieval ---
            # Embed all expanded query variants in a single batched call (rather than one
            # embed_query round-trip per variant per store), then reuse each embedding
            # against both stores directly instead of rebuilding a retriever wrapper per call.
            expanded = self.query_expander.expand(q)
            expanded_embeddings = self.embedding_model.embed_documents(expanded)
            chunk_ctx, hypo_ctx = [], []
            for embedding in expanded_embeddings:
                chunk_ctx.extend(self._search_above_relevance_floor(self.chunks_vectorstore, embedding, k=5))
                hypo_ctx.extend(self._search_above_relevance_floor(self.hypo_vectorstore, embedding, k=8))

            # Chunk-store matches are already real source text. Hypo-store matches are
            # hypothetical *questions* used only as a retrieval index (questions embed more
            # like other questions than raw prose does) — swap each match back to its parent
            # chunk's real content via parent_chunk_id, rather than feeding the answering LLM
            # a list of questions as if they were evidence.
            chunk_texts = {d.page_content for d in chunk_ctx}
            hypo_parent_ids = {d.metadata.get("parent_chunk_id") for d in hypo_ctx}
            hypo_texts = {
                self._chunk_content_by_id[cid]
                for cid in hypo_parent_ids
                if cid in self._chunk_content_by_id
            }
            contexts = list(dict.fromkeys(list(chunk_texts) + list(hypo_texts)))

            # --- Answering ---
            # One call over the combined, deduplicated context instead of answering each
            # retrieval path separately and then combining the two answers: cheaper, and
            # avoids losing information by summarizing two independent summaries.
            final_answer = self.answerer.answer(q, contexts)
            self.logger.info("GenAI Answer: %s", final_answer)

            # --- Normalization ---
            partial = self.q_normalizer.normalize_query(q, final_answer)
            updated_fields = acc.update(q, partial)

            # --- Audit log ---
            self.audit_logger.log(self.file_name, q, final_answer, partial)

            # --- Append contexts in order ---
            per_query_contexts.append(contexts)

            # Short-circuit: only the "studies_tool" query's QUERY_FIELDS mapping ever
            # populates this key in `partial`, so this check is scoped to that query
            # without needing to duplicate its literal text here.
            if partial.get("studies_tool") == "no":
                records = self.assembler.assemble(self.file_name, acc)
                flattened = self._flatten_with_alignment(records, acc.get_field_provenance(), per_query_contexts, query_order, FinalRecordAssembler.BASE_FIELDS)
                return flattened

        # --- Assemble full record(s) ---
        records = self.assembler.assemble(self.file_name, acc)

        # --- Flatten with aligned contexts ---
        flattened = self._flatten_with_alignment(records, acc.get_field_provenance(), per_query_contexts, query_order, FinalRecordAssembler.BASE_FIELDS)
        return flattened
    

    def _flatten_with_alignment(
        self,
        records: list[dict],
        field_provenance: dict,
        per_query_contexts: list[list[str]],
        query_order: list[str],
        base_fields: list[str],
    ) -> list[dict]:
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