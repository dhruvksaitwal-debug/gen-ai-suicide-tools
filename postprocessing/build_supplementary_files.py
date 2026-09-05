"""
Builds the two Excel supplementary files ("Additional files" in BMC's terminology) for the
BMC manuscript:
  - Additional file 1: the full 210-tool list with frequency and scope classification
    (referenced in Methods, Tool-name resolution methodology).
  - Additional file 2: the full composite-score LOINC-candidate ranking across all 210
    tools, with the underlying per-criterion values and percentile ranks that feed the
    composite score -- not just the top-15 excerpt shown in Table 5 of the main text.
    Submitting the full ranking (not just the excerpt) lets a reader verify the ranking
    directly rather than trusting the main-text table alone.
"""
import pandas as pd

TOOL_LIST_CSV = "docs/BMC_Submission/Supporting_Material/tool_list_consolidated.csv"
RANKING_CSV = "docs/BMC_Submission/Supporting_Material/loinc_candidate_ranking.csv"

OUT_DIR = "docs/BMC_Submission"
OUT_FILE1 = f"{OUT_DIR}/Additional_file_1_Tool_List.xlsx"
OUT_FILE2 = f"{OUT_DIR}/Additional_file_2_LOINC_Candidate_Ranking.xlsx"


def build_additional_file_1():
    df = pd.read_csv(TOOL_LIST_CSV)
    df = df.sort_values("frequency", ascending=False).reset_index(drop=True)
    df.insert(0, "rank_by_frequency", df.index + 1)
    df = df.rename(columns={
        "canonical_tool_name": "Canonical Tool Name",
        "scope": "Scope",
        "frequency": "Frequency",
        "rank_by_frequency": "Rank (by Frequency)",
    })

    readme = pd.DataFrame({
        "Column": ["Rank (by Frequency)", "Canonical Tool Name", "Scope", "Frequency"],
        "Description": [
            "Rank among the 210 retained tools, ordered by frequency (descending).",
            "The human-verified canonical name for the tool, after two-stage resolution "
            "of raw tool-name mentions (see Methods, Tool-name resolution methodology).",
            "\"suicide-specific\" or \"comprehensive mental health tool (suicide-related "
            "item/subscale)\", as classified by two domain experts (a physician and a "
            "terminology/standardization expert).",
            "Number of article-tool records for this canonical tool across the "
            "769-article, 1,012-output-file corpus.",
        ],
    })

    with pd.ExcelWriter(OUT_FILE1, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        df.to_excel(writer, sheet_name="Tool List", index=False)
    print(f"Wrote {OUT_FILE1} ({len(df)} tools)")


def build_additional_file_2():
    df = pd.read_csv(RANKING_CSV)

    column_order = [
        "rank", "canonical_tool_name", "scope", "frequency", "geographic_breadth",
        "setting_breadth", "condition_breadth", "population_total", "population_median",
        "studies_with_population_data", "composite_score", "already_in_loinc",
        "open_access", "candidate_tier", "loinc_verification_note",
        "pct_frequency", "pct_geographic", "pct_setting", "pct_population",
        "pct_condition", "pct_scope", "countries", "settings",
    ]
    df = df[column_order]

    rename_map = {
        "rank": "Rank", "canonical_tool_name": "Canonical Tool Name", "scope": "Scope",
        "frequency": "Frequency", "geographic_breadth": "Geographic Breadth (# countries)",
        "setting_breadth": "Setting Breadth (# categories)",
        "condition_breadth": "Condition Breadth (# conditions)",
        "population_total": "Population Coverage (total participants)",
        "population_median": "Population Coverage (median per study)",
        "studies_with_population_data": "# Studies with Population Data",
        "composite_score": "Composite Score",
        "already_in_loinc": "Already in LOINC? (verified, top-ranked tools only)",
        "open_access": "Open Access Status (verified, top-ranked tools only)",
        "candidate_tier": "LOINC Candidate Tier (verified / preliminary)",
        "loinc_verification_note": "LOINC/Open-Access Verification Note",
        "pct_frequency": "Percentile Rank: Frequency",
        "pct_geographic": "Percentile Rank: Geographic Breadth",
        "pct_setting": "Percentile Rank: Setting Breadth",
        "pct_population": "Percentile Rank: Population Coverage",
        "pct_condition": "Percentile Rank: Condition Breadth",
        "pct_scope": "Percentile Rank: Suicide-Specificity",
        "countries": "Countries (semicolon-separated)",
        "settings": "Care Settings (semicolon-separated)",
    }
    df = df.rename(columns=rename_map)

    readme = pd.DataFrame({
        "Section": [
            "Overview", "", "Composite score", "", "LOINC/open-access verification", "",
            "Candidate tier",
        ],
        "Notes": [
            "Full statistical ranking of all 210 suicide-related tools retained from the "
            "769-article corpus (see manuscript Methods, Statistical ranking of LOINC "
            "candidates), not just the top-15 excerpt shown in the main text (Table 5).",
            "",
            "Composite Score = the unweighted mean of six percentile-rank columns "
            "(Frequency, Geographic Breadth, Setting Breadth, Population Coverage, "
            "Condition Breadth, Suicide-Specificity), each computed across all 210 tools. "
            "Equal weighting was pre-specified and not tuned to favor any tool.",
            "",
            "\"Already in LOINC?\" and \"Open Access Status\" were verified by hand "
            "against loinc.org and each instrument's publisher/copyright holder for the "
            "top-ranked tools only (see manuscript Methods and Future Work); most rows "
            "are blank, meaning not independently checked for this analysis -- a blank "
            "cell is neither a confirmation nor an exclusion.",
            "",
            "\"verified\" = confirmed absent from LOINC and confirmed freely available. "
            "\"preliminary\" = confirmed absent from LOINC, no commercial publisher "
            "identified, but open availability not independently confirmed. Blank = not "
            "checked.",
        ],
    })

    with pd.ExcelWriter(OUT_FILE2, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        df.to_excel(writer, sheet_name="Full Ranking", index=False)
    print(f"Wrote {OUT_FILE2} ({len(df)} tools)")


if __name__ == "__main__":
    build_additional_file_1()
    build_additional_file_2()
