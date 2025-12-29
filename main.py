import os
import argparse
import pandas as pd
from constants import PDF_STEM_MAXLEN
from orchestrator import DocRAGPipelineOrchestrator
from evaluator import Evaluator, load_gold_examples


def run_pipeline_for_pdf(doc_id, data_folder, queries, evaluate, gold_examples):
    """
    Runs the RAG pipeline for a single PDF and returns a DataFrame.
    If evaluate=True, also runs RAGAS and returns metrics.
    """

    print(f"\nProcessing {doc_id}...")

    # Initialize pipeline
    pipeline = DocRAGPipelineOrchestrator(file_name=doc_id, data_folder=data_folder)
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

    # Otherwise run RAGAS evaluation
    evaluator = Evaluator(gold_examples=gold_examples)

    # Add gold references
    df["reference"] = df.apply(
        lambda row: gold_examples.get(row["doc_id"], {}).get(row["question"], None),
        axis=1
    )

    # Run evaluation
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


def main():
    parser = argparse.ArgumentParser(description="Run DocRAG pipeline on PDFs.")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Enable RAGAS evaluation (default: disabled)"
    )
    args = parser.parse_args()

    evaluate = args.evaluate
    print(f"\nRAGAS Evaluation Enabled: {evaluate}")

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
    os.makedirs(results_folder, exist_ok=True)

    # Define queries
    queries = [
        "Does the article study any suicide screening/assessment tools?",
        "Which suicide screening/assessment tool is studied?",
        "Classify if the tool is screening or assessment.",
        "Discuss the study outcome with the tool analyzed.",
        "Discuss clinical settings where the tool is used.",
        "Discuss demographics of participants for whom the tool is used.",
        "The geographic locations or countries where the study was conducted.",
        "Discuss intended medical conditions of the patients in the study.",
        "Discuss the study duration and population size."
    ]

    # Process each PDF
    for file_name in os.listdir(data_folder):
        if not file_name.lower().endswith(".pdf"):
            continue

        doc_id = os.path.splitext(file_name)[0]

        df = run_pipeline_for_pdf(
            doc_id=doc_id,
            data_folder=data_folder,
            queries=queries,
            evaluate=evaluate,
            gold_examples=GOLD_EXAMPLES
        )

        # Remove contexts column if present
        df = df.drop(columns=["contexts"], errors="ignore")
        
        # Handle empty dataframe OR missing studies_tool
        if df.empty:
            print(f"WARNING: No records produced for {doc_id}. Saving EMPTY CSV.")
            output_csv = os.path.join(results_folder, f"{safe_doc_id}_EMPTY.csv")
            df.to_csv(output_csv, index=False)
            return
        
        if "studies_tool" not in df.columns:
            print(f"WARNING: Missing studies_tool column for {doc_id}. Saving RAW CSV.")
            output_csv = os.path.join(results_folder, f"{safe_doc_id}_RAW.csv")
            df.to_csv(output_csv, index=False)
            return

        # Save one CSV per tool per PDF
        safe_doc_id = doc_id[:PDF_STEM_MAXLEN].lower().replace(" ", "_")
        if df["studies_tool"].iloc[0] == "no":
            df = df.drop(columns=["studies_tool", "tool_name", "tool_type"], errors="ignore")
            output_csv = os.path.join(results_folder, f"{safe_doc_id}_no_tool_results.csv")
            df.to_csv(output_csv, index=False, float_format="%.2f")
            print(f"Saved results to {output_csv}")
        else:
            print("[DEBUG] About to save CSV:", not df.empty)
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
                print(f"Saved results to {output_csv}")
        

if __name__ == "__main__":
    main()