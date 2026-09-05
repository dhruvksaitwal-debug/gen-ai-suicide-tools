"""
Builds the per-article-tool wide KPI table used for the statistical LOINC-candidate
ranking in the new (BMC) manuscript. Each of the 1,012 tool-level CSVs under
test_results/Done/Batch-1|2/ already represents one (article, tool) pair with its 12 KPI
question/answer rows in long format; this script pivots each to one wide row, maps its raw
tool_name to the consolidated canonical name (see build_raw_to_canonical_map.py), and keeps
only the canonical tools retained as suicide-related in the domain-expert-reviewed
Screening Tool Analysis.xlsx (scope != 'No'/'Unclear').

Unlike the earlier Frontiers submission, tool frequency and all other statistics here are
computed fresh from the live 1,012-CSV corpus rather than carried over from the Excel's own
frequency column, which was found (see raw_to_canonical_frequency_check.csv) to disagree
with the live corpus by a small margin for several tools -- most likely because it reflects
an earlier corpus snapshot. The Excel's per-tool scope classification (suicide-specific vs.
comprehensive-mental-health-with-suicide-item) is still treated as authoritative, since that
is a human clinical judgment call, not a corpus count.
"""
import glob
import json

import openpyxl
import pandas as pd

from build_raw_to_canonical_map import DUPLICATE_TARGET_MERGES, EXCEL_PATH

RAW_TO_CANONICAL_PATH = "docs/BMC_Submission/Supporting_Material/raw_to_canonical.json"
OUT_JOINED = "docs/BMC_Submission/Supporting_Material/tool_kpi_joined.csv"
OUT_TOOL_LIST = "docs/BMC_Submission/Supporting_Material/tool_list_consolidated.csv"

KPI_FIELDS = [
    "studies_tool", "tool_type", "outcome_summary", "clinical_setting",
    "demographics_summary", "location", "duration_value", "duration_text",
    "population_size", "population_text", "medical_conditions",
]


def load_scope_map():
    """canonical_tool_name (post-merge) -> normalized scope, restricted to kept tools."""
    wb = openpyxl.load_workbook(EXCEL_PATH, data_only=True)
    ws = wb["Sheet1"]
    scope_by_name = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        name, suicide = row[1], row[3]
        if not name:
            continue
        canon = DUPLICATE_TARGET_MERGES.get(str(name).strip(), str(name).strip())
        s = str(suicide).strip().lower()
        if s in ("no", "unclear"):
            continue
        normalized = "suicide-specific" if s == "x" else \
            "comprehensive mental health tool (suicide-related item/subscale)"
        # A merged pair could in principle carry different scope calls; keep whichever
        # was already recorded first rather than silently overwrite.
        scope_by_name.setdefault(canon, normalized)
    return scope_by_name


def main():
    with open(RAW_TO_CANONICAL_PATH, encoding="utf-8") as f:
        raw_to_canonical = json.load(f)

    scope_by_name = load_scope_map()

    all_files = glob.glob("test_results/Done/Batch-1/*.csv") + glob.glob("test_results/Done/Batch-2/*.csv")
    tool_files = [f for f in all_files if "no_tool_results" not in f]

    records = []
    skipped_no_tool_name = 0
    skipped_unmapped = 0
    skipped_out_of_scope = 0

    for path in tool_files:
        df = pd.read_csv(path)
        answers = dict(zip(df["question"], df["answer"]))
        tool_name = str(answers.get("tool_name", "")).strip()
        if not tool_name or tool_name.lower() in ("nan", "unspecified_tool"):
            skipped_no_tool_name += 1
            continue

        canon = raw_to_canonical.get(tool_name)
        if canon is None:
            skipped_unmapped += 1
            continue

        scope = scope_by_name.get(canon)
        if scope is None:
            skipped_out_of_scope += 1
            continue

        doc_id = df["doc_id"].iloc[0] if "doc_id" in df.columns and not df.empty else path
        record = {"doc_id": doc_id, "canonical_tool_name": canon, "scope": scope, "source_file": path}
        for field in KPI_FIELDS:
            record[field] = answers.get(field)
        records.append(record)

    joined = pd.DataFrame(records)
    joined.to_csv(OUT_JOINED, index=False, encoding="utf-8")

    tool_list = (
        joined.groupby(["canonical_tool_name", "scope"]).size()
        .reset_index(name="frequency")
        .sort_values("frequency", ascending=False)
    )
    tool_list.to_csv(OUT_TOOL_LIST, index=False, encoding="utf-8")

    print(f"Tool-level CSVs scanned: {len(tool_files)}")
    print(f"Skipped (no tool_name / unspecified): {skipped_no_tool_name}")
    print(f"Skipped (raw tool_name not in raw_to_canonical map): {skipped_unmapped}")
    print(f"Skipped (mapped tool is out of scope / No or Unclear): {skipped_out_of_scope}")
    print(f"Joined rows written: {len(joined)}")
    print(f"Distinct kept canonical tools: {joined['canonical_tool_name'].nunique()}")
    print(f"Wrote {OUT_JOINED} and {OUT_TOOL_LIST}")


if __name__ == "__main__":
    main()
