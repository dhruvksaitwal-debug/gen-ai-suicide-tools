import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

INPUT_FILE = "data/collated_results.csv"
OUTPUT_MAP = "data/tool_name_map.raw.json"


def main() -> None:
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"Input file '{INPUT_FILE}' does not exist.")

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

    logger.info("Initial tool map written to %s with %d entries", OUTPUT_MAP, len(mapping))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()