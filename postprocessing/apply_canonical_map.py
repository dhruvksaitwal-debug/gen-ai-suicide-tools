import pandas as pd
import json
from pathlib import Path

INPUT_FILE = "data/collated_results_v1.csv"
INPUT_MAP = "data/tool_name_map.canonical.json"
OUTPUT_FILE = "data/collated_results_with_canonical.csv"

def main():
    df = pd.read_csv(INPUT_FILE)
    mapping = json.loads(Path(INPUT_MAP).read_text(encoding="utf-8"))

    def map_name(x):
        if not isinstance(x, str):
            return x
        x_norm = x.strip()
        return mapping.get(x_norm, x_norm)

    # Create canonical column
    df["tool_name_canonical"] = df["tool_name"].apply(map_name)

    # Reorder columns: put canonical right after original
    cols = list(df.columns)
    if "tool_name" in cols and "tool_name_canonical" in cols:
        cols.remove("tool_name_canonical")
        insert_at = cols.index("tool_name") + 1
        cols.insert(insert_at, "tool_name_canonical")
        df = df[cols]

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved updated dataset with canonical tool names to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

