"""
Builds the raw tool_name string -> final canonical tool name map, anchored against the
domain-expert-reviewed target list in Screening Tool Analysis.xlsx (274 canonical names,
of which 216 are retained after excluding 'No'/'Unclear' scope rows -- see
Supplementary_Table_S1_tool_frequencies.csv).

Re-running normalize_tools.py's build_clusters() from scratch does NOT reproduce this
target list exactly: that function's greedy clustering depends on Python's per-process
string-hash-randomized set iteration order, and a fresh run also skips the manual
consolidation the domain experts applied when the Excel was produced (a fresh run here
gave 318 canonical tools vs. the Excel's 274). So instead of re-deriving clusters, this
script matches every raw tool_name string directly against the fixed, already-approved
274-name target list -- the source of truth stays the human-reviewed spreadsheet, not a
new automated pass.
"""
import difflib
import json

import openpyxl
import pandas as pd

from normalize_tools import (
    FUZZY_THRESHOLD,
    extract_abbr_and_full,
    has_distinguishing_modifier,
    load_raw_tool_names,
    norm_abbr,
)

EXCEL_PATH = "docs/BMC_Submission/Supporting_Material/Screening Tool Analysis.xlsx"
OUT_MAP = "docs/BMC_Submission/Supporting_Material/raw_to_canonical.json"
OUT_UNMATCHED = "docs/BMC_Submission/Supporting_Material/raw_to_canonical_unmatched.csv"
OUT_FREQ_CHECK = "docs/BMC_Submission/Supporting_Material/raw_to_canonical_frequency_check.csv"
OUT_EXCLUDED = "docs/BMC_Submission/Supporting_Material/raw_to_canonical_excluded.csv"

# Raw strings the automated abbreviation/fuzzy matcher could not confidently place against
# the 274-name target list, resolved by manual inspection (each checked against the full
# target list and, where relevant, the source CSV context). "None" means the raw string is
# excluded from the tool-ranking analysis entirely -- either because it names something
# that is not one of the 274 reviewed canonical tools (e.g. a distinct psychosocial
# interview mnemonic like HEADSS that was never part of the reviewed list, or a non-scale
# artifact like a chatbot name), or because it packs two tools into one field
# ("DPI C-SSRS + HDRS") with no reliable way to attribute it to a single canonical tool.
MANUAL_OVERRIDES = {
    "Beck Suicidal Ideas Scale (SSI)": "Beck Scale for Suicide Ideation (BSS)",
    "C-SSRS Screener": "Columbia Suicide Severity Rating Scale (C-SSRS) Screen version",
    "Columbia Brief Suicide Severity Rating Scale (C-BSSRS)": None,
    "Columbia Scale": "Columbia-Suicide Severity Rating Scale (C-SSRS)",
    "Columbia Suicide History Form": None,
    "Columbia Suicide Rating Scale (CSRS)": "Columbia-Suicide Severity Rating Scale (C-SSRS)",
    "Columbia Suicide Severity Scale": "Columbia-Suicide Severity Rating Scale (C-SSRS)",
    "Columbia-Suicide Severity Rating Scale (Posner et al., 2011)":
        "Columbia-Suicide Severity Rating Scale (C-SSRS)",
    "Columbia-Suicide Severity Rating Scale Screen Version (C-SSRS Screen)":
        "Columbia Suicide Severity Rating Scale (C-SSRS) Screen version",
    "Columbia-Suicide Severity Rating Scale, Screening Version":
        "Columbia Suicide Severity Rating Scale (C-SSRS) Screen version",
    "DPI C-SSRS + HDRS": None,
    "Depression Symptom Index-Suicide Subscale (DSI-SS)":
        "Depressive Symptom Inventory-Suicidality Subscale",
    "ERS Suicide Risk Scale": "ERS Suicide Risk Scale (Emergency Room Suicide Risk Scale)",
    "G.T. Mixed States Rating Scale (G.T. MSRS scale)":
        "G.T. Mixed States Rating Scale (G.T. MSRS scale) (Giovanni Tundo’s)",
    "HEADSS": None,
    "HEADSS assessment": None,
    "HEADSS psychosocial screening tool": None,
    "HEADSSS": None,
    "HEEADSSS": None,
    "Hamilton Rating Scale for Depression (HAM-D)": "Hamilton Depression Rating Scale (HDRS)",
    "Hamilton Rating Scale for Depression (HRSD)": "Hamilton Depression Rating Scale (HDRS)",
    "Item 3 on the 17-item Hamilton Depression Rating Scale (HAM-D)":
        "Hamilton Depression Rating Scale (HDRS)",
    "MHSAFE probes":
        "MHSAFE probes (Mental Health, Suicide ideation, Suicide attempts, Affect, "
        "Family/Friends, Events)",
    "Mini International Neuropsychiatric Interview (MINI 5.0-MZ)":
        "Mini International Neuropsychiatric Interview (MINI)",
    "MyHEARTSMAP digital psychosocial self-assessment":
        "MyHEARTSMAP digital psychosocial self-assessment ( a digital youth "
        "mental‑health assessment that evaluates 10 domains: Home, Education, "
        "Activities, Relationships, Thoughts/Emotions, Substances, Safety, Sexual Health, "
        "Medical, and Services.)",
    "Okasha assessment tool": "Okasha Suicidality Scale",
    "OxSATS": "Oxford Suicide Assessment Tool after Self-harm (OxSATS)",
    "PROVE-SR": None,
    "Pallis 18-item + Beck Suicide Intent Scale (SIS) 7-item": None,
    "Patient Health Questionnaire-9 Modified for Adolescents (PHQ-9-A)":
        "Patient Health Questionnaire for adolescents (PHQ-A)",
    "Paykel Suicidality Scale": "Paykel Suicide Scale (PSS)",
    "Paykel inventory": "Paykel Suicide Scale (PSS)",
    "Quick Inventory of Depressive Symptomatology-Self Report (QIDS-SR)":
        "QIDS-SR16 (QIDS‑SR16 = a 16‑item self‑report depression severity "
        "scale covering the 9 DSM symptom domains.)",
    "Repeated Episodes of Self-Harm score": None,
    "ResourceBot": None,
    "SAD PERSONS self-harm screening tool": "SAD PERSONS Scale (SPS)",
    "SBQ-ASC": "Suicidal Behaviours Questionnaire (SBQ-ASC)",
    "Scale for Suicidal Ideation (Beck et al., 1979)": "Beck Scale for Suicide Ideation (BSS)",
    "Suicide Crisis Inventory-2 Short Form": "Suicide Crisis Inventory-2",
    "Youth Risk Behavior Survey (YRBS)": "Youth Risk Behavior Surveillance System (YRBSS)",
    "item 3 of the Hamilton Depression Rating Scale (HAMD-17)":
        "Hamilton Depression Rating Scale (HDRS)",
    "nomogram": None,
}

# Two Excel rows sometimes describe the exact same instrument under two spellings/citation
# styles that the automated clustering (and the human review) never merged -- confirmed by
# inspecting each pair's raw variants directly. Consolidated here rather than left as
# phantom near-duplicate rows in the final 216/274-tool list.
DUPLICATE_TARGET_MERGES = {
    "Beck Scale for Suicidal Ideation": "Beck Scale for Suicide Ideation (BSS)",
    "Suicide Behaviors Questionnaire–Autism Spectrum Conditions (SBQ-ASC)":
        "Suicidal Behaviours Questionnaire (SBQ-ASC)",
    "Revised Suicide Crisis Inventory (SCI-2)": "Suicide Crisis Inventory-2",
    "Hamilton Depression Rating Scale (Hamilton, 1960)": "Hamilton Depression Rating Scale (HDRS)",
    "Substance Abuse and Mental Health Services Administration (SAMHSA) Suicide Assessment "
    "Five-step Evaluation and Triage (SAFE-T)":
        "Suicide Assessment 5-step Evaluation and Triage (SAFE-T)",
}


def load_targets():
    wb = openpyxl.load_workbook(EXCEL_PATH, data_only=True)
    ws = wb["Sheet1"]
    targets = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        _id, name, freq, suicide = row[0], row[1], row[2], row[3]
        if not name:
            continue
        targets.append({"name": str(name).strip(), "frequency_raw": freq, "scope": suicide})
    return targets


def parse_excel_frequency(val):
    """Handle the known 'A+B'-style text cell (e.g. C-SSRS's '150+6') alongside plain ints."""
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip()
    if "+" in s:
        return sum(int(part) for part in s.split("+"))
    return int(s)


def build_target_index(targets):
    bare_full = {}
    abbr_index = {}
    for t in targets:
        name = t["name"]
        abbr, full = extract_abbr_and_full(name)
        bare_full[name] = full or name
        if abbr:
            abbr_index.setdefault(abbr, []).append(name)
    return bare_full, abbr_index


def best_fuzzy(query, candidates, bare_full):
    best_name, best_score = None, 0.0
    for cand in candidates:
        if has_distinguishing_modifier(query, bare_full[cand]):
            continue
        score = difflib.SequenceMatcher(None, query.lower(), bare_full[cand].lower()).ratio()
        if score > best_score:
            best_name, best_score = cand, score
    return best_name, best_score


def match_raw_to_targets(raw_names, targets):
    target_names = [t["name"] for t in targets]
    target_name_set = set(target_names)
    bare_full, abbr_index = build_target_index(targets)
    target_freq = {t["name"]: parse_excel_frequency(t["frequency_raw"]) for t in targets}

    raw_to_canonical = {}
    unmatched = []

    for raw in raw_names:
        if raw in MANUAL_OVERRIDES:
            target = MANUAL_OVERRIDES[raw]
            if target is not None:
                raw_to_canonical[raw] = target
            continue

        if raw in target_name_set:
            raw_to_canonical[raw] = raw
            continue

        abbr_r, full_r = extract_abbr_and_full(raw)
        query = full_r or raw

        candidates = abbr_index.get(abbr_r, []) if abbr_r else []
        if candidates:
            if len(candidates) == 1:
                raw_to_canonical[raw] = candidates[0]
                continue
            if full_r is None:
                # Raw is itself a bare abbreviation shared by several distinct target
                # instruments -- comparing a short bare token against each candidate's
                # long full name structurally fails (length mismatch tanks the ratio), so
                # route to whichever candidate is the more commonly cited referent instead.
                raw_to_canonical[raw] = max(candidates, key=lambda c: target_freq.get(c, 0))
                continue
            match, score = best_fuzzy(query, candidates, bare_full)
            if match:
                raw_to_canonical[raw] = match
                continue
            # All candidates blocked by the modifier guard (e.g. a version qualifier
            # nobody in the target list carries) -- fall back to best score regardless
            # of the guard, but only if it's still a strong match.
            best_name, best_score = None, 0.0
            for cand in candidates:
                s = difflib.SequenceMatcher(None, query.lower(), bare_full[cand].lower()).ratio()
                if s > best_score:
                    best_name, best_score = cand, s
            if best_score >= FUZZY_THRESHOLD:
                raw_to_canonical[raw] = best_name
                continue

        # No abbreviation-based candidates (or none cleared) -- fuzzy match against the
        # full target list.
        match, score = best_fuzzy(query, target_names, bare_full)
        if score >= FUZZY_THRESHOLD:
            raw_to_canonical[raw] = match
        else:
            unmatched.append((raw, match, score))

    for raw, canon in list(raw_to_canonical.items()):
        raw_to_canonical[raw] = DUPLICATE_TARGET_MERGES.get(canon, canon)

    return raw_to_canonical, unmatched


def main():
    raw_names_with_src = load_raw_tool_names()
    unique_raw_names = sorted(set(n for n, _ in raw_names_with_src))
    raw_occurrence_counts = pd.Series([n for n, _ in raw_names_with_src]).value_counts()
    targets = load_targets()

    raw_to_canonical, unmatched = match_raw_to_targets(unique_raw_names, targets)

    with open(OUT_MAP, "w", encoding="utf-8") as f:
        json.dump(raw_to_canonical, f, indent=2, ensure_ascii=False, sort_keys=True)

    excluded_rows = []
    for raw, target in MANUAL_OVERRIDES.items():
        if target is None and raw in raw_occurrence_counts.index:
            excluded_rows.append({
                "raw_name": raw, "occurrences": int(raw_occurrence_counts[raw]),
                "reason": "manually reviewed, not one of the 274 reviewed canonical tools "
                          "or an ambiguous multi-tool field",
            })
    for raw, closest, score in unmatched:
        excluded_rows.append({
            "raw_name": raw, "occurrences": int(raw_occurrence_counts.get(raw, 0)),
            "reason": f"no confident match found (closest: {closest!r}, score={score:.2f})",
        })
    with open(OUT_EXCLUDED, "w", encoding="utf-8", newline="") as f:
        pd.DataFrame(excluded_rows).to_csv(f, index=False)

    with open(OUT_UNMATCHED, "w", encoding="utf-8", newline="") as f:
        pd.DataFrame(unmatched, columns=["raw_name", "closest_target", "score"]).to_csv(
            f, index=False
        )

    # Validation: recompute per-canonical frequency from the 700 raw occurrences and
    # compare against the Excel's own stated frequency for that canonical name. Targets
    # consolidated away by DUPLICATE_TARGET_MERGES are skipped here (they no longer appear
    # as a value in raw_to_canonical) and their Excel-stated frequency is folded into the
    # surviving target's expected count instead.
    computed = pd.Series(
        [raw_to_canonical[n] for n, _ in raw_names_with_src if n in raw_to_canonical]
    ).value_counts()
    expected_freq = {}
    for t in targets:
        name = DUPLICATE_TARGET_MERGES.get(t["name"], t["name"])
        expected_freq[name] = expected_freq.get(name, 0) + parse_excel_frequency(t["frequency_raw"])

    rows = []
    mismatches = 0
    for name, expected in sorted(expected_freq.items()):
        actual = int(computed.get(name, 0))
        if expected != actual:
            mismatches += 1
        rows.append({
            "canonical_tool_name": name, "excel_frequency": expected,
            "computed_frequency": actual, "match": expected == actual,
        })
    with open(OUT_FREQ_CHECK, "w", encoding="utf-8", newline="") as f:
        pd.DataFrame(rows).to_csv(f, index=False)

    final_canonical_count = len(expected_freq)
    print(f"Unique raw strings: {len(unique_raw_names)}")
    print(f"Matched to a target canonical: {len(raw_to_canonical)}")
    print(f"Excluded (manual review or no confident match): {len(excluded_rows)}")
    print(f"Final canonical tool count after duplicate-target consolidation: {final_canonical_count}")
    print(f"Canonical tools with frequency mismatch vs. Excel: {mismatches} / {final_canonical_count}")
    print(f"Wrote {OUT_MAP}, {OUT_UNMATCHED}, {OUT_EXCLUDED}, {OUT_FREQ_CHECK}")


if __name__ == "__main__":
    main()
