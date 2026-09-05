import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

INPUT_FILE = "data/collated_results.csv"
OUTPUT_FILE = "data/collated_results_v1.csv"


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file '{INPUT_FILE}' does not exist.")

    df = pd.read_csv(INPUT_FILE)

    # Create a mapping: doc_id → unique integer
    unique_ids = (
        df["doc_id"]
        .drop_duplicates()
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "unique_doc_id"})
    )

    # Merge back into the main dataframe
    df_with_ids = df.merge(unique_ids, on="doc_id", how="left")
    cols = ["unique_doc_id"] + [c for c in df_with_ids.columns if c != "unique_doc_id"]
    df_with_ids = df_with_ids[cols]

    # Save the updated CSV
    df_with_ids.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved updated file with unique_doc_id column to: %s", OUTPUT_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
