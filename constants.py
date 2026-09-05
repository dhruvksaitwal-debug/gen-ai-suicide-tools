# Maximum length for shortened PDF stems used in filenames
PDF_STEM_MAXLEN = 30

# Chat/completion model used for both answer generation (orchestrator) and RAGAS evaluation
# (evaluator) — single source of truth so the two never diverge silently.
DEFAULT_MODEL_NAME = "gpt-4o-mini"

# Default thread pool size for fanning out independent, per-item LLM calls
# (table/image summarization within a PDF, hypothetical-question generation per chunk)
DEFAULT_MAX_WORKERS = 8

# Shared Chroma persistent-store path
CHROMA_DB_PATH = "./doc_rag_db"

# Text-splitting parameters (characters) applied to extracted PDF content before embedding,
# so a chunk is a retrieval-sized unit rather than a whole page.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Chunks shorter than this are skipped for hypothetical-question generation (boilerplate,
# headers, page-number artifacts aren't worth an LLM call).
MIN_CHUNK_CHARS_FOR_HYPO = 40

# Disk cache for expensive per-PDF LLM outputs (extraction, hypothetical questions)
CACHE_DIR = "pipeline_cache"

# Minimum normalized relevance score (langchain_chroma's distance-metric-aware relevance
# function; higher = more similar) a retrieved chunk must clear to be used as context.
# Deliberately lenient: this only filters out matches that are essentially uncorrelated
# with the query, not moderately-relevant ones. Calibrated analytically against unit
# embeddings, not against real OpenAI embeddings on real documents — treat as a starting
# point to tune, not a validated cutoff.
MIN_RELEVANCE_SCORE = -0.3

# ragas.evaluate()'s own internal fan-out for scoring rows within a single document —
# unrelated to DEFAULT_MAX_WORKERS above, which governs our own extraction/hypo-gen calls.
# ragas's default (16) reliably pins a 200K-tokens/minute account ceiling on its own,
# observed empirically via sustained 429s and job timeouts even with a single document
# being scored. Lower trades peak parallelism for far fewer wasted retry-backoff cycles.
RAGAS_MAX_WORKERS = 4

# Process-wide cap on concurrent OpenAI chat-completion calls in flight at once, regardless
# of which document or pipeline stage issues them. DEFAULT_MAX_WORKERS (per-document fan-out)
# and main.py's --workers (documents processed at once) each independently multiply demand on
# the same shared 200K-tokens/minute account ceiling; their product can exceed it even when
# each is individually reasonable. This is the single choke point that keeps true concurrent
# demand bounded regardless of how those two are configured, mirroring the same TPM-ceiling
# lesson learned from RAGAS_MAX_WORKERS above.
LLM_CONCURRENCY_LIMIT = 4