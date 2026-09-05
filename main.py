import argparse
import concurrent.futures
import logging
import os
import time

import pandas as pd

from answer_normalizer import QUERIES
from constants import CHROMA_DB_PATH, DEFAULT_MODEL_NAME
from naming import safe_doc_id_stem
from orchestrator import DocRAGPipelineOrchestrator
from evaluator import Evaluator, load_gold_examples
from vectorstore_manager import get_shared_chroma_client, VectorStoreManager, drop_collections

logger = logging.getLogger(__name__)

DEFAULT_PDF_WORKERS = 4
MAX_DOC_ATTEMPTS = 3


def _run_pipeline_for_pdf_once(
    doc_id: str,
    data_folder: str,
    queries: list[str],
    evaluate: bool,
    gold_examples: dict,
    model_name: str,
) -> pd.DataFrame:
    # Initialize pipeline
    pipeline = DocRAGPipelineOrchestrator(file_name=doc_id, data_folder=data_folder, model_name=model_name)
    pipeline.setup()

    # Run pipeline → get answers for all KPI's
    records = pipeline.run_queries(queries)

    # Attach doc_id
    for row in records:
        row["doc_id"] = doc_id

    df = pd.DataFrame(records)

    # If evaluation disabled → return KPI-only CSV
    if not evaluate:
        return df

    # Otherwise run RAGAS evaluation. Pass the same model_name used to generate the answers,
    # so evaluation is always scored with the same model that produced them, not just
    # whichever model happens to share the same default right now.
    evaluator = Evaluator(gold_examples=gold_examples, model_name=model_name)
    results = evaluator.evaluate(df.to_dict(orient="records"))

    # Clean up columns
    results = results.rename(columns={"response": "GenAI_answer"})
    results = results.drop(columns=["retrieved_contexts"], errors="ignore")
    results["doc_id"] = df["doc_id"].values

    # Reorder
    desired_order = [
        "doc_id", "user_input", "GenAI_answer",
        "faithfulness", "answer_relevancy", "context_recall"
    ]
    results = results[[col for col in desired_order if col in results.columns]]

    return results


def run_pipeline_for_pdf(
    doc_id: str,
    data_folder: str,
    queries: list[str],
    evaluate: bool,
    gold_examples: dict,
    model_name: str = DEFAULT_MODEL_NAME,
) -> pd.DataFrame:
    """
    Runs the RAG pipeline for a single PDF and returns a DataFrame, retrying up to
    MAX_DOC_ATTEMPTS times. A failed attempt can leave a vectorstore collection registered in
    Chroma's catalog with incomplete on-disk data (crashed mid-write); retrying without
    cleanup just repeats the same failure, so each retry force-drops this doc_id's collections
    first to guarantee the next attempt starts from a clean slate. Also covers transient
    network errors (DNS blips, connection resets), which aren't retried at the request level
    the way rate limits are.
    """
    logger.info("Processing %s...", doc_id)

    last_exc = None
    for attempt in range(1, MAX_DOC_ATTEMPTS + 1):
        try:
            return _run_pipeline_for_pdf_once(doc_id, data_folder, queries, evaluate, gold_examples, model_name)
        except Exception as e:
            last_exc = e
            if attempt < MAX_DOC_ATTEMPTS:
                logger.warning(
                    "Attempt %d/%d failed for %s (%s); clearing vectorstore state and retrying.",
                    attempt, MAX_DOC_ATTEMPTS, doc_id, e
                )
                client = get_shared_chroma_client(CHROMA_DB_PATH)
                drop_collections(client, [
                    VectorStoreManager.sanitize_collection_name(doc_id + "_article"),
                    VectorStoreManager.sanitize_collection_name(doc_id + "_hypo"),
                ])
                time.sleep(2 * attempt)

    raise last_exc


def save_results(doc_id: str, df: pd.DataFrame, results_folder: str) -> None:
    """Save the pipeline output for one PDF, splitting into one CSV per tool when applicable."""
    df = df.drop(columns=["contexts"], errors="ignore")
    safe_doc_id = safe_doc_id_stem(doc_id)

    if df.empty:
        logger.warning("No records produced for %s. Saving EMPTY CSV.", doc_id)
        output_csv = os.path.join(results_folder, f"{safe_doc_id}_EMPTY.csv")
        df.to_csv(output_csv, index=False)
        return

    if "studies_tool" not in df.columns:
        logger.warning("Missing studies_tool column for %s. Saving RAW CSV.", doc_id)
        output_csv = os.path.join(results_folder, f"{safe_doc_id}_RAW.csv")
        df.to_csv(output_csv, index=False)
        return

    if df["studies_tool"].iloc[0] == "no":
        df = df.drop(columns=["studies_tool", "tool_name", "tool_type"], errors="ignore")
        output_csv = os.path.join(results_folder, f"{safe_doc_id}_no_tool_results.csv")
        df.to_csv(output_csv, index=False, float_format="%.2f")
        logger.info("Saved results to %s", output_csv)
        return

    # Group by tool_name and save one CSV per tool
    for tool_name, group in df.groupby("tool_name"):
        safe_tool = (
            tool_name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace(":", "_")
            .replace(".", "_")
            .replace("(", "_")
            .replace(")", "_")
        )
        group = group.drop(columns=["studies_tool", "tool_name", "tool_type"], errors="ignore")
        output_csv = os.path.join(results_folder, f"{safe_doc_id}_{safe_tool}_results.csv")
        group.to_csv(output_csv, index=False, float_format="%.2f")
        logger.info("Saved results to %s", output_csv)


def main():
    parser = argparse.ArgumentParser(description="Run DocRAG pipeline on PDFs.")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Enable RAGAS evaluation (default: disabled)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_PDF_WORKERS,
        help=f"Number of PDFs to process concurrently (default: {DEFAULT_PDF_WORKERS})"
    )
    parser.add_argument(
        "--doc-ids",
        type=str,
        default=None,
        help=(
            "Pipe-separated doc_ids to process, e.g. 'Gold16|Gold17' (default: all PDFs in the "
            "data folder). Pipe-separated rather than comma-separated because '|' can never "
            "appear in a Windows filename, while article-title-derived doc_ids routinely contain commas."
        )
    )
    args = parser.parse_args()

    evaluate = args.evaluate
    logger.info("RAGAS Evaluation Enabled: %s", evaluate)

    # Load gold examples only if evaluation is enabled and choose folders based on evaluation flag
    if evaluate:
        data_folder = "gold_data"
        gold_examples_path = os.path.join("gold_data", "gold_examples.json")
        GOLD_EXAMPLES = load_gold_examples(gold_examples_path)
        results_folder = "gold_results"
    else:
        data_folder = "test_data"
        GOLD_EXAMPLES = {}
        results_folder = "test_results"

    if not os.path.isdir(data_folder):
        raise FileNotFoundError(
            f"Data folder '{data_folder}' does not exist. Create it and add PDFs before running."
        )
    os.makedirs(results_folder, exist_ok=True)

    queries = QUERIES

    # Process each PDF
    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".pdf")]

    if args.doc_ids:
        requested = {d.strip() for d in args.doc_ids.split("|") if d.strip()}
        pdf_files = [f for f in pdf_files if os.path.splitext(f)[0] in requested]
        found = {os.path.splitext(f)[0] for f in pdf_files}
        missing = requested - found
        if missing:
            logger.warning("Requested doc_ids not found in '%s': %s", data_folder, sorted(missing))

    if not pdf_files:
        logger.warning("No PDF files found in '%s'.", data_folder)
        return

    # Pre-create the shared Chroma persistent client once, serially: concurrently
    # constructing PersistentClient instances against a not-yet-initialized directory races
    # on tenant setup and fails for all but one caller. Every orchestrator's
    # VectorStoreManager reuses this same cached client (see
    # vectorstore_manager.get_shared_chroma_client) rather than creating its own — separate
    # client instances against the same directory also race on reading each other's
    # freshly-written HNSW segments once processing starts.
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    get_shared_chroma_client(CHROMA_DB_PATH)

    total = len(pdf_files)
    completed = 0
    batch_start = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_pipeline_for_pdf,
                doc_id=os.path.splitext(file_name)[0],
                data_folder=data_folder,
                queries=queries,
                evaluate=evaluate,
                gold_examples=GOLD_EXAMPLES,
            ): (os.path.splitext(file_name)[0], time.monotonic())
            for file_name in pdf_files
        }

        # as_completed() yields one at a time on this thread, so the counter needs no lock.
        for future in concurrent.futures.as_completed(futures):
            doc_id, submitted_at = futures[future]
            elapsed = time.monotonic() - submitted_at
            completed += 1

            try:
                df = future.result()
            except Exception:
                logger.exception(
                    "[%d/%d] Pipeline failed for %s after %.1fs; skipping.",
                    completed, total, doc_id, elapsed
                )
                continue

            save_results(doc_id, df, results_folder)
            logger.info("[%d/%d] Completed %s in %.1fs", completed, total, doc_id, elapsed)

    logger.info("Done: %d/%d PDFs processed in %.1fs total.", completed, total, time.monotonic() - batch_start)


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )
    logging.getLogger(__name__).info("Logging to console and %s", log_path)
    main()