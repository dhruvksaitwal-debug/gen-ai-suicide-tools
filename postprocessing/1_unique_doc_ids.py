import pandas as pd

INPUT_FILE = "data/collated_results.csv"
OUTPUT_FILE = "data/collated_results_v1.csv"

# Load your collated results
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

print(f"Saved updated file with unique_doc_id column to: {OUTPUT_FILE}")