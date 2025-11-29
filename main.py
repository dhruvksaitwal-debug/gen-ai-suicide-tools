import os
import pandas as pd
from orchestrator import DocRAGPipelineOrchestrator
from evaluator import Evaluator, load_gold_examples

def main():
    # Load gold examples
    GOLD_EXAMPLES = load_gold_examples()

    # Define queries
    queries = [
        "Does the article study any suicide screening/assessment tools?",
        "Which suicide screening/assessment tool is studied?",
        "Classify if the tool is screening or assessment.",
        "Discuss the study outcome.",
        "Discuss clinical settings where the tool is used.",
        "Discuss demographics of participants.",
        "Where was the study conducted?",
        "Discuss majority medical conditions.",
        "Discuss study duration and population size."
    ]

    all_flattened = []

    # Loop over all PDF files in Data folder
    data_folder = "Data"
    for file_name in os.listdir(data_folder):
        if file_name.lower().endswith(".pdf"):
            doc_id = os.path.splitext(file_name)[0]  # e.g. "Gold1", "Gold2"
            print(f"\nProcessing {doc_id}...")

            # Initialize pipeline for this file
            pipeline = DocRAGPipelineOrchestrator(file_name=doc_id, data_folder=data_folder)
            pipeline.setup()

            # Run pipeline → flattened records
            records, field_provenance, per_query_contexts = pipeline.run_queries(queries)
            flattened = pipeline._flatten_with_alignment(records, field_provenance, per_query_contexts)

            # Attach doc_id explicitly
            for row in flattened:
                row["doc_id"] = doc_id

            all_flattened.extend(flattened)

    # Convert to DataFrame
    flattened_df = pd.DataFrame(all_flattened)

    # Add gold references
    flattened_df["reference"] = flattened_df.apply(
        lambda row: GOLD_EXAMPLES.get(row["doc_id"], {}).get(row["question"], None),
        axis=1
    )

    # Evaluate with RAGAS
    evaluator = Evaluator(gold_examples=GOLD_EXAMPLES)
    results = evaluator.evaluate(flattened_df.to_dict(orient="records"))

    # Reorder columns
    results = results.rename(columns={"response": "GenAI_answer"})
    results = results.drop(columns=["retrieved_contexts"], errors="ignore")
    results["doc_id"] = flattened_df["doc_id"].values

    print(f'\nColumns in results dataframe: {results.columns.tolist()}')
    desired_order = [
        "doc_id", "user_input", "GenAI_answer",
        "faithfulness", "answer_relevancy", "context_recall"
        # "context_precision", "context_recall", "answer_correctness"
    ]
    results = results[[col for col in desired_order if col in results.columns]]

    # Save evaluation results to CSV
    results.to_csv("ragas_evaluation_results.csv", index=False, float_format="%.2f")
    print("Evaluation results saved to ragas_evaluation_results.csv")

if __name__ == "__main__":
    main()