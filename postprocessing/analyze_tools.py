import pandas as pd
from pathlib import Path

INPUT_FILE = "data/collated_results_with_canonical.csv"
OUTPUT_TOOL_COUNTS = "data/tool_usage_counts.csv"
OUTPUT_SUMMARY = "data/tool_summary_stats.txt"

# ------------------------------------------------------------
# Helper: classify screening vs assessment tools
# ------------------------------------------------------------

def classify_tool(name: str):
    """
    Simple heuristic:
    - If name contains 'screen' → screening tool
    - If name contains 'assess' → assessment tool
    - Otherwise: other
    """
    if not isinstance(name, str):
        return "other"

    n = name.lower()
    if "screen" in n:
        return "screening"
    if "assess" in n:
        return "assessment"
    return "other"


# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------

def main():
    df = pd.read_csv(INPUT_FILE)

    # Normalize canonical names
    df["tool_name_canonical"] = (
        df["tool_name_canonical"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
    )

    # --------------------------------------------------------
    # 1. Total number of unique articles
    # --------------------------------------------------------
    total_articles = df["unique_doc_id"].nunique()

    # --------------------------------------------------------
    # 2. Articles WITH tools
    # --------------------------------------------------------
    df_tools = df.dropna(subset=["tool_name_canonical"])
    df_tools = df_tools.query("tool_name_canonical != 'unspecified_tool'")

    articles_with_tools = df_tools["unique_doc_id"].nunique()

    # --------------------------------------------------------
    # 3. Articles WITHOUT tools
    # --------------------------------------------------------
    articles_without_tools = total_articles - articles_with_tools

    # --------------------------------------------------------
    # 4. Unique canonical tools
    # --------------------------------------------------------
    unique_tools = (
        df_tools["tool_name_canonical"]
        .dropna()
        .unique()
        .tolist()
    )

    total_unique_tools = len(unique_tools)

    # --------------------------------------------------------
    # 4a. Unique screening tools
    # --------------------------------------------------------
    screening_tools = [
        t for t in unique_tools
        if classify_tool(t) == "screening"
    ]
    total_screening_tools = len(screening_tools)

    # --------------------------------------------------------
    # 4b. Unique assessment tools
    # --------------------------------------------------------
    assessment_tools = [
        t for t in unique_tools
        if classify_tool(t) == "assessment"
    ]
    total_assessment_tools = len(assessment_tools)

    # --------------------------------------------------------
    # 5. GLOBAL frequency of each tool across ALL articles
    # --------------------------------------------------------
    tool_counts = (
        df_tools["tool_name_canonical"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "tool_name_canonical", "tool_name_canonical": "count"})
    )

    tool_counts.to_csv(OUTPUT_TOOL_COUNTS, index=False)

    # --------------------------------------------------------
    # Save summary stats
    # --------------------------------------------------------
    summary = f"""
SUMMARY STATISTICS
------------------
Total unique articles: {total_articles}
Articles with tools: {articles_with_tools}
Articles without tools: {articles_without_tools}

Total unique canonical tools: {total_unique_tools}
  - Screening tools: {total_screening_tools}
  - Assessment tools: {total_assessment_tools}

Global tool usage counts saved to: {OUTPUT_TOOL_COUNTS}
"""

    Path(OUTPUT_SUMMARY).write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()