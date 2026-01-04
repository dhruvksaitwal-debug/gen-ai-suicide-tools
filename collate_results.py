import os
import pandas as pd

# Folder containing test_results for all PDF files
INPUT_FOLDER = "test_results"
OUTPUT_FILE = "data/collated_results.csv"

# Final schema (13 columns)
FINAL_COLUMNS = [
    "doc_id",
    "studies_tool",
    "tool_name",
    "tool_type",
    "outcome_summary",
    "clinical_settings",       # normalized name
    "demographics_summary",
    "location",
    "duration_value",
    "duration_text",
    "population_size",
    "population_text",
    "medical_conditions",
]

def load_and_flatten_csv(file_path):
    """Reads one CSV and returns a dict with all 13 fields."""
    df = pd.read_csv(file_path)

    # Extract doc_id (same for all rows)
    doc_id = df["doc_id"].iloc[0]

    # Build a mapping: question -> answer
    record = {row["question"]: row["answer"] for _, row in df.iterrows()}

    # Normalize clinical_setting → clinical_settings
    if "clinical_setting" in record:
        record["clinical_settings"] = record.pop("clinical_setting")

    # Build final row with all 13 fields
    row = {"doc_id": doc_id}
    for col in FINAL_COLUMNS:
        if col == "doc_id":
            continue
        row[col] = record.get(col, None)

    return row


def combine_all_csvs():
    rows = []

    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".csv"):
            file_path = os.path.join(INPUT_FOLDER, filename)
            row = load_and_flatten_csv(file_path)
            rows.append(row)

    # Create final DataFrame
    df = pd.DataFrame(rows, columns=FINAL_COLUMNS)

    # Save combined CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Combined CSV saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    combine_all_csvs()