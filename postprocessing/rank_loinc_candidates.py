"""
Statistical ranking of the 211 suicide-related tools (see build_tool_kpi_join.py) as
candidates for LOINC standardization, using all 12 extracted KPI fields rather than tool
frequency alone -- the central methodological upgrade over the earlier (Frontiers) version,
which picked five candidates by frequency plus unquantified assertions about openness and
format.

Composite score design (documented here for reproducibility, since this is what a reviewer
will scrutinize first):
  - Six equally-weighted criteria, each converted to a 0-1 percentile rank across all 211
    tools before averaging, so no single criterion's raw scale (e.g. population counts in
    the thousands vs. a country count under 20) dominates the others:
      1. frequency               - number of article-tool records (adoption in literature)
      2. geographic breadth      - distinct countries the tool was studied in
      3. clinical-setting breadth - distinct care-setting categories (ED, inpatient,
                                    outpatient/primary care, school/community, military/
                                    veteran, correctional, specialty/other)
      4. population coverage     - total participants across all studies of the tool
      5. condition breadth       - distinct medical/psychiatric conditions studied
      6. suicide-specificity     - 1.0 if classified suicide-specific, 0.5 if a
                                    comprehensive mental-health tool with a suicide-related
                                    item/subscale (both are in-scope; this only expresses
                                    that a dedicated suicide tool is the more natural fit
                                    for a suicide-domain LOINC panel)
  - Equal weighting is a deliberate choice: it is the least gameable option and does not
    presuppose which criterion should decide the outcome (in particular, it was NOT tuned
    to put any specific tool on top).
  - This produces a ranked list, not a hypothesis test -- no p-values are reported, since
    the 211 tools are not a random sample and standard significance testing does not apply
    to a full-population literature-mapping corpus like this one.

Two further checks are applied only to the top of the ranked list, by hand, since they are
not derivable from the KPI fields: (a) whether the tool is already represented in LOINC,
and (b) whether its format is structurally suited to LOINC's discrete-question/answer-list
model versus a clinical-judgment/decision-support framework. Both are recorded directly in
the manuscript-build step, not this script.
"""
import re

import pandas as pd

from extract_countries import extract_countries

JOINED_PATH = "docs/BMC_Submission/Supporting_Material/tool_kpi_joined.csv"
OUT_RANKED = "docs/BMC_Submission/Supporting_Material/loinc_candidate_ranking.csv"

# Manually verified against loinc.org search results and each instrument's own licensing
# terms (session date: see manuscript methods for the verification date). Only the
# top-ranked tools were checked by hand -- this is not run automatically against a live
# LOINC API, so it is deliberately scoped to the tools that matter for the candidate
# shortlist rather than claimed for all 211 rows.
LOINC_VERIFICATION = {
    "Columbia-Suicide Severity Rating Scale (C-SSRS)": {
        "already_in_loinc": True, "open_access": "open",
        "note": "Multiple existing LOINC codes (e.g. 93245-9, 93373-9).",
    },
    "Columbia Suicide Severity Rating Scale (C-SSRS) Screen version": {
        "already_in_loinc": True, "open_access": "open",
        "note": "Covered by the C-SSRS screener LOINC panel (93373-9).",
    },
    "Patient Health Questionnaire-9 (PHQ-9)": {
        "already_in_loinc": True, "open_access": "open",
        "note": "Existing LOINC panel 44249-1 and item-level codes.",
    },
    "Beck Scale for Suicide Ideation (BSS)": {
        "already_in_loinc": False, "open_access": "proprietary",
        "note": "No LOINC panel found; commercially licensed via Pearson Assessments, "
                "not freely accessible.",
    },
    "Mini International Neuropsychiatric Interview (MINI)": {
        "already_in_loinc": False, "open_access": "licensed",
        "note": "No LOINC panel found; requires a use license from the copyright holder "
                "for any use, including research.",
    },
    "Ask Suicide-Screening Questions (ASQ)": {
        "already_in_loinc": False, "open_access": "open",
        "note": "No LOINC panel found; freely available NIMH/SAMHSA toolkit.",
    },
    "Suicidal Behaviors Questionnaire-Revised (SBQ-R)": {
        "already_in_loinc": False, "open_access": "open",
        "note": "No LOINC panel found; freely published academic instrument.",
    },
    "Adult Suicidal Ideation Questionnaire (ASIQ)": {
        "already_in_loinc": False, "open_access": "proprietary",
        "note": "No LOINC panel found; commercially published by Psychological "
                "Assessment Resources (PAR), Inc., not freely accessible.",
    },
    "Computerized Adaptive Screen for Suicidal Youth (CASSY)": {
        "already_in_loinc": False, "open_access": "licensed",
        "note": "No LOINC panel found; licensed exclusively through Adaptive Testing "
                "Technologies.",
    },
    "Suicide Ideation and Behavior Assessment Tool (SIBAT)": {
        "already_in_loinc": False, "open_access": "licensed",
        "note": "No LOINC panel found; distributed under license via Mapi Research "
                "Trust/ePROVIDE and developed in connection with a branded "
                "pharmaceutical product.",
    },
    "Beck Suicide Intent Scale (SIS)": {
        "already_in_loinc": False, "open_access": "open",
        "note": "No LOINC panel found; the full instrument is freely hosted by the "
                "Beck Institute, unlike the commercially distributed BSS.",
    },
    "Self-rating Idea of Suicide Scale (SIOSS)": {
        "already_in_loinc": False, "open_access": "preliminary-open",
        "note": "No LOINC panel found; no commercial publisher identified, but open "
                "availability was not independently confirmed beyond its reproduction "
                "in academic articles.",
    },
    "Suicide Crisis Inventory-2": {
        "already_in_loinc": False, "open_access": "preliminary-open",
        "note": "No LOINC panel found; no commercial publisher identified, but open "
                "availability was not independently confirmed beyond its reproduction "
                "in academic articles.",
    },
    "Depressive Symptom Inventory-Suicidality Subscale": {
        "already_in_loinc": False, "open_access": "preliminary-open",
        "note": "No LOINC panel found; no commercial publisher identified, but open "
                "availability was not independently confirmed beyond its reproduction "
                "in academic articles.",
    },
    "Suicidal Ideation Attributes Scale (SIDAS)": {
        "already_in_loinc": False, "open_access": "preliminary-open",
        "note": "No LOINC panel found; no commercial publisher identified, but open "
                "availability was not independently confirmed beyond its reproduction "
                "in academic articles.",
    },
}

SETTING_CATEGORIES = [
    ("Emergency", r"emergency|urgent care|acute care|acute psychiatric"),
    ("Inpatient/Psychiatric hospital", r"inpatient|psychiatric hospital|hospitali|general hospital"),
    ("Outpatient/Primary care", r"outpatient|primary care|ambulatory|clinic(?!al settings)"),
    ("School/Community", r"school|college|university counsel|community"),
    ("Military/Veteran", r"military|veteran|\bVHA\b"),
    ("Correctional/Justice", r"incarcerat|prison|correctional|justice"),
    ("Specialty/Other", r"palliative|oncology|maternity|geriatric|substance (use|abuse)|"
                         r"occupational health|research institution|nursing"),
]
SETTING_PATTERNS = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in SETTING_CATEGORIES]


def extract_settings(text: str) -> set:
    if not isinstance(text, str):
        return set()
    return {name for name, pat in SETTING_PATTERNS if pat.search(text)}


def split_conditions(text: str) -> set:
    if not isinstance(text, str) or not text.strip():
        return set()
    parts = re.split(r",|;| and ", text)
    return {p.strip().lower() for p in parts if p.strip()}


def percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average")


def main():
    df = pd.read_csv(JOINED_PATH)

    rows = []
    for canon, g in df.groupby("canonical_tool_name"):
        countries = set()
        for loc in g["location"].dropna():
            countries |= extract_countries(str(loc))

        settings = set()
        for cs in g["clinical_setting"].dropna():
            settings |= extract_settings(str(cs))

        conditions = set()
        for mc in g["medical_conditions"].dropna():
            conditions |= split_conditions(str(mc))

        pop = pd.to_numeric(g["population_size"], errors="coerce").dropna()

        rows.append({
            "canonical_tool_name": canon,
            "scope": g["scope"].iloc[0],
            "frequency": len(g),
            "geographic_breadth": len(countries),
            "countries": "; ".join(sorted(countries)),
            "setting_breadth": len(settings),
            "settings": "; ".join(sorted(settings)),
            "condition_breadth": len(conditions),
            "population_total": int(pop.sum()) if len(pop) else 0,
            "population_median": float(pop.median()) if len(pop) else 0.0,
            "studies_with_population_data": int(len(pop)),
        })

    stats = pd.DataFrame(rows)

    stats["scope_weight"] = stats["scope"].map(
        lambda s: 1.0 if s == "suicide-specific" else 0.5
    )

    stats["pct_frequency"] = percentile_rank(stats["frequency"])
    stats["pct_geographic"] = percentile_rank(stats["geographic_breadth"])
    stats["pct_setting"] = percentile_rank(stats["setting_breadth"])
    stats["pct_population"] = percentile_rank(stats["population_total"])
    stats["pct_condition"] = percentile_rank(stats["condition_breadth"])
    stats["pct_scope"] = percentile_rank(stats["scope_weight"])

    stats["composite_score"] = stats[
        ["pct_frequency", "pct_geographic", "pct_setting",
         "pct_population", "pct_condition", "pct_scope"]
    ].mean(axis=1)

    stats = stats.sort_values("composite_score", ascending=False).reset_index(drop=True)
    stats.insert(0, "rank", stats.index + 1)

    stats["already_in_loinc"] = stats["canonical_tool_name"].map(
        lambda n: LOINC_VERIFICATION.get(n, {}).get("already_in_loinc")
    )
    stats["open_access"] = stats["canonical_tool_name"].map(
        lambda n: LOINC_VERIFICATION.get(n, {}).get("open_access")
    )
    stats["loinc_verification_note"] = stats["canonical_tool_name"].map(
        lambda n: LOINC_VERIFICATION.get(n, {}).get("note")
    )
    def candidate_tier(n):
        if n not in LOINC_VERIFICATION:
            return None
        v = LOINC_VERIFICATION[n]
        if v["already_in_loinc"] is not False:
            return None
        if v["open_access"] == "open":
            return "verified"
        if v["open_access"] == "preliminary-open":
            return "preliminary"
        return None

    stats["candidate_tier"] = stats["canonical_tool_name"].map(candidate_tier)

    stats.to_csv(OUT_RANKED, index=False, encoding="utf-8")

    print(f"Ranked {len(stats)} tools. Wrote {OUT_RANKED}")
    print()
    print(stats[["rank", "canonical_tool_name", "frequency", "geographic_breadth",
                 "setting_breadth", "condition_breadth", "population_total",
                 "composite_score", "already_in_loinc", "open_access"]]
          .head(15).to_string(index=False))

    verified = stats[stats["candidate_tier"] == "verified"]
    preliminary = stats[stats["candidate_tier"] == "preliminary"]
    print()
    print("Verified candidates (not in LOINC, confirmed openly available), by rank:")
    print(verified[["rank", "canonical_tool_name", "composite_score"]].to_string(index=False))
    print()
    print("Preliminary candidates (not in LOINC, openness not independently confirmed), by rank:")
    print(preliminary[["rank", "canonical_tool_name", "composite_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
