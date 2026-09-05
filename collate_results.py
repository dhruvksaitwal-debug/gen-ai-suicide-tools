import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

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

def load_and_flatten_csv(file_path: str) -> dict:
    """Reads one CSV and returns a dict with all 13 fields."""
    df = pd.read_csv(file_path)
    if df.empty or "doc_id" not in df.columns:
        raise ValueError(f"{file_path} has no usable rows (empty or missing 'doc_id' column).")

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


def combine_all_csvs() -> None:
    if not os.path.isdir(INPUT_FOLDER):
        raise FileNotFoundError(f"Input folder '{INPUT_FOLDER}' does not exist.")

    rows = []
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith(".csv"):
            continue
        file_path = os.path.join(INPUT_FOLDER, filename)
        try:
            rows.append(load_and_flatten_csv(file_path))
        except (ValueError, pd.errors.ParserError) as e:
            logger.warning("Skipping %s: %s", file_path, e)

    if not rows:
        logger.warning("No usable CSVs found in '%s'. Nothing to write.", INPUT_FOLDER)
        return

    # Create final DataFrame
    df = pd.DataFrame(rows, columns=FINAL_COLUMNS)

    # Save combined CSV
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info("Combined CSV saved to: %s", OUTPUT_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    combine_all_csvs()