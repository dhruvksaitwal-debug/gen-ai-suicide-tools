import pandas as pd
import json
from pathlib import Path

INPUT_FILE = "data/collated_results.csv"
OUTPUT_MAP = "data/tool_name_map.raw.json"

def main():
    df = pd.read_csv(INPUT_FILE)

    tool_names = (
        df["tool_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )

    # initial identity mapping
    mapping = {name: name for name in sorted(tool_names)}

    Path(OUTPUT_MAP).write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Initial tool map written to {OUTPUT_MAP} with {len(mapping)} entries")

if __name__ == "__main__":
    main()