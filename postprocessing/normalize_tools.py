"""
Tool-name normalization for the full 770-article / 1012-CSV corpus.

The pipeline's tool_name field is deliberately free-text (see answer_normalizer.py's
_canon_tool_name: no alias lock-in), so the same instrument is recorded under many raw
spellings across articles: "Columbia-Suicide Severity Rating Scale (C-SSRS)" vs "Columbia
Suicide Severity Rating Scale (C-SSRS)" (hyphen), bare abbreviations ("ASQ", "C-SSRS"),
reversed patterns ("BSSI (Beck Scale for Suicide Ideation)"), and case variants. Naively
counting distinct strings overstates the number of unique tools substantially (416 raw
strings vs the ~239 previously reported) and was flagged by a reviewer as needing an
explicit, describable methodology.

Procedure (each stage only processes what the previous stage left unresolved):
  1. Abbreviation extraction: pull a short (2-12 char) alphanumeric abbreviation from
     "Full Name (ABBR)" or "ABBR (Full Name)" patterns, or treat a short all-caps/hyphenated
     token with no parentheses as a bare abbreviation.
  2. Abbreviation clustering: group all raw names sharing the same normalized abbreviation
     (uppercase, spaces/hyphens stripped for comparison only). The canonical display name for
     each cluster is the most frequent full-name variant containing that abbreviation.
  3. Fuzzy clustering for the remainder: raw names with no extractable abbreviation are
     matched against existing canonical names first (difflib ratio >= FUZZY_THRESHOLD), then
     against each other, to catch spelling/capitalization variants that never carried an
     abbreviation in any occurrence.
  4. Manual review queue: every merge decision in stage 3, plus any abbreviation cluster
     containing more than one distinct full-name spelling, is written to
     tool_normalization_review.csv for spot-checking before the counts are treated as final.
"""
import difflib
import glob
import re
from collections import Counter, defaultdict

import pandas as pd

FUZZY_THRESHOLD = 0.87

# Cross-abbreviation fuzzy matches (stage 3.5) are NOT auto-merged when either name carries a
# version/edition marker, a population qualifier, or an administration-mode qualifier — these
# routinely denote a genuinely distinct instrument or variant in the psychometrics literature
# (BDI vs BDI-II; MMPI-2 vs MMPI-3 vs MMPI-A "Adolescent"; PHQ-9 vs PHQ-2 vs PHQ-A; QIDS vs
# QIDS-SR "Self-Report"; SBQ vs SBQ-ASC "Autism Spectrum Conditions"), so high string similarity
# alone is not sufficient evidence they're the same tool. Such pairs are written to the review
# queue as "distinct version/variant — not merged" rather than merged automatically.
MODIFIER_PATTERN = re.compile(
    r"(-II\b|-2\b|-3\b|-9\b|-A\b|-SR\b|-ASC\b|-JR\b|-S\b|\bII\b|\bAdolescent\b|\bAdult\b|"
    r"\bYouth\b|\bChild\b|\bKiddie\b|\bJunior\b|\bSelf[- ]?Report\b|\bSelf[- ]?Rated\b|\bScreen\b|"
    r"\bBrief\b|\bElectronic\b|\be-|\bTelephone\b|\bRemote\b|\bComputerized\b|\bModified\b|"
    r"\bAdapted\b|\bTranslated\b|\bRevised\b)",
    re.IGNORECASE,
)

# Pairs a fuzzy score alone can't distinguish (a content word substitution, not an added
# qualifier) but that name genuinely different constructs on inspection, or are otherwise
# uncertain enough to need a domain expert's read rather than an automated call. Written to
# the review queue as "NOT merged: manually excluded" instead of silently combined.
MANUAL_EXCLUDE_PAIRS = {
    frozenset({"suicidal ideation attributes scale", "suicidal intrusions attributes scale"}),
    # Language/cultural adaptation — treated as a distinct validated variant, consistent with
    # how population- and mode-qualified variants are handled above.
    frozenset({"literacy of suicide scale", "bangla literacy of suicide scale"}),
}


def _strip_abbr_suffix(name: str) -> str:
    m = ABBR_TRAILING.match(name)
    return m.group(1).strip().lower() if m else name.strip().lower()


def has_distinguishing_modifier(name_a: str, name_b: str) -> bool:
    """True if exactly one of the two names carries a version/population/mode qualifier the
    other lacks — the asymmetry is what signals a real variant rather than a naming quirk."""
    if frozenset({_strip_abbr_suffix(name_a), _strip_abbr_suffix(name_b)}) in MANUAL_EXCLUDE_PAIRS:
        return True
    mods_a = set(m.upper() for m in MODIFIER_PATTERN.findall(name_a))
    mods_b = set(m.upper() for m in MODIFIER_PATTERN.findall(name_b))
    return mods_a != mods_b

ABBR_TRAILING = re.compile(r"^(.*\S)\s*\(([A-Za-z0-9\-]{2,12})\)\s*$")
ABBR_LEADING = re.compile(r"^([A-Za-z0-9\-]{2,12})\s*\((.+)\)\s*$")
BARE_ABBR = re.compile(r"^[A-Z0-9][A-Z0-9\-]{1,11}$")


def load_raw_tool_names():
    all_files = glob.glob("test_results/Done/Batch-1/*.csv") + glob.glob("test_results/Done/Batch-2/*.csv")
    tool_files = [f for f in all_files if "no_tool_results" not in f]
    names = []
    for f in tool_files:
        df = pd.read_csv(f)
        row = df[df["question"] == "tool_name"]
        if not row.empty:
            name = str(row.iloc[0]["answer"]).strip()
            if name and name.lower() != "unspecified_tool" and name.lower() != "nan":
                names.append((name, f))
    return names


def norm_abbr(a: str) -> str:
    return a.upper().replace(" ", "").replace("-", "").replace(".", "")


def extract_abbr_and_full(name: str):
    """Return (normalized_abbr_or_None, full_name_candidate_or_None) for one raw name."""
    m = ABBR_TRAILING.match(name)
    if m:
        return norm_abbr(m.group(2)), m.group(1).strip()
    m = ABBR_LEADING.match(name)
    if m:
        return norm_abbr(m.group(1)), m.group(2).strip()
    if BARE_ABBR.match(name) and len(name) <= 12:
        return norm_abbr(name), None
    return None, None


def build_clusters(raw_names_with_src):
    names_only = [n for n, _ in raw_names_with_src]
    counts = Counter(names_only)

    raw_abbr_clusters = defaultdict(list)   # norm_abbr -> list of raw names sharing that token
    unresolved = []

    for name in set(names_only):
        abbr, _full = extract_abbr_and_full(name)
        if abbr:
            raw_abbr_clusters[abbr].append(name)
        else:
            unresolved.append(name)

    # Two different real instruments can coincidentally share an abbreviation (SPS = both "SAD
    # PERSONS Scale" and "Suicide Probability Scale"; BHS = both "Beck Hopelessness Scale" and
    # "Behavioral Health Screen"; SIS = both "Suicide Intent Scale" and the conceptually
    # different "Suicidal Ideation Scale"). Sharing an abbreviation token is necessary but not
    # sufficient evidence of being the same tool, so within each abbreviation group the
    # full-name-bearing variants are sub-clustered by fuzzy similarity to each other (same
    # modifier guard as the cross-abbreviation pass); only genuinely similar full names merge
    # under one abbreviation-derived key. A collision produces multiple keys, e.g. "SPS#0" and
    # "SPS#1", each carrying the shared abbreviation but distinct canonical names.
    abbr_clusters = {}
    review_rows = []
    for abbr, members in raw_abbr_clusters.items():
        full_variants = sorted(set(m for m in members if extract_abbr_and_full(m)[1]))
        bare_only = [m for m in members if not extract_abbr_and_full(m)[1]]

        if len(full_variants) <= 1:
            key = abbr
            abbr_clusters[key] = list(members)
            continue

        # Greedy fuzzy sub-clustering of the distinct full-name variants. Compares the
        # extracted full-name portion (parenthetical stripped), not the raw string — otherwise
        # a reversed "ABBR (Full Name)" entry never matches its own "Full Name (ABBR)" twin,
        # since raw-string similarity scores low on reversed word order.
        bare_full = {m: (extract_abbr_and_full(m)[1] or m) for m in full_variants}
        remaining = list(full_variants)
        sub_clusters = []
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            rest = []
            for other in remaining:
                if has_distinguishing_modifier(bare_full[seed], bare_full[other]):
                    rest.append(other)
                    continue
                score = difflib.SequenceMatcher(None, bare_full[seed].lower(), bare_full[other].lower()).ratio()
                if score >= FUZZY_THRESHOLD:
                    group.append(other)
                else:
                    rest.append(other)
            remaining = rest
            sub_clusters.append(group)

        if len(sub_clusters) == 1:
            key = abbr
            abbr_clusters[key] = full_variants + bare_only
        else:
            # Genuine collision: split into distinct keys, each still tagged with the shared
            # abbreviation for traceability; bare-abbreviation-only mentions can't be
            # disambiguated and go to the largest (most frequent) sub-cluster.
            sub_clusters.sort(key=lambda g: -sum(counts[m] for m in g))
            for i, group in enumerate(sub_clusters):
                key = f"{abbr}#{i}"
                abbr_clusters[key] = list(group) + (bare_only if i == 0 else [])
            review_rows.append({
                "cluster_key": f"abbr_collision:{abbr}",
                "canonical_chosen": None,
                "raw_variants": " || ".join(" / ".join(g) for g in sub_clusters),
                "reason": f"SPLIT: abbreviation '{abbr}' used for {len(sub_clusters)} unrelated instruments",
            })

    # Canonical display name per (possibly split) abbreviation cluster: prefer the
    # highest-count raw string that itself contains a full name; fall back to the most common
    # raw string overall.
    canonical_for_abbr = {}
    for key, members in abbr_clusters.items():
        full_variants = [m for m in members if extract_abbr_and_full(m)[1]]
        pool = full_variants if full_variants else members
        canonical = max(pool, key=lambda m: counts[m])
        canonical_for_abbr[key] = canonical
        distinct_full_spellings = sorted(set(full_variants))
        if len(distinct_full_spellings) > 1:
            review_rows.append({
                "cluster_key": f"abbr:{key}",
                "canonical_chosen": canonical,
                "raw_variants": " | ".join(sorted(set(members))),
                "reason": "multiple full-name spellings under one abbreviation (already fuzzy-verified similar)",
            })

    # Cross-abbreviation merge: different source articles sometimes abbreviate the same
    # underlying instrument differently (e.g. "Beck Scale for Suicidal Ideation (SSI)" vs
    # "Beck Scale for Suicide Ideation (BSS)" vs "...(BSI)" vs "...(BSSI)") — same tool, four
    # distinct abbreviation tokens, so stage 2's exact-abbreviation clustering can't catch it.
    # Fuzzy-match each cluster's canonical full name against every other cluster's; merge
    # clusters whose full names are near-identical (abbreviation differences aside) under
    # whichever canonical has the higher total occurrence count.
    abbr_keys = list(abbr_clusters.keys())
    parent = {k: k for k in abbr_keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    cluster_total = {k: sum(counts[m] for m in abbr_clusters[k]) for k in abbr_keys}
    for i, k1 in enumerate(abbr_keys):
        for k2 in abbr_keys[i + 1:]:
            n1, n2 = canonical_for_abbr[k1], canonical_for_abbr[k2]
            score = difflib.SequenceMatcher(None, n1.lower(), n2.lower()).ratio()
            if score < FUZZY_THRESHOLD:
                continue
            if has_distinguishing_modifier(n1, n2) or has_distinguishing_modifier(k1, k2):
                review_rows.append({
                    "cluster_key": f"crossabbr_BLOCKED:{k1}<->{k2}",
                    "canonical_chosen": None,
                    "raw_variants": f"{n1}  ||  {n2}",
                    "reason": f"NOT merged: version/population/mode qualifier differs, score={score:.2f}",
                })
                continue
            union(k1, k2)
            review_rows.append({
                "cluster_key": f"crossabbr:{k1}<->{k2}",
                "canonical_chosen": None,  # resolved below once union-find settles
                "raw_variants": f"{n1}  ||  {n2}",
                "reason": f"different abbreviations, same-looking instrument, score={score:.2f}",
            })

    # Re-derive canonical name per merged group: highest total occurrence count wins.
    groups = defaultdict(list)
    for k in abbr_keys:
        groups[find(k)].append(k)
    for root, members in groups.items():
        if len(members) > 1:
            winner = max(members, key=lambda k: cluster_total[k])
            winning_name = canonical_for_abbr[winner]
            for k in members:
                canonical_for_abbr[k] = winning_name

    # Fuzzy-match unresolved (no-abbreviation) names against canonical names first, then
    # against each other.
    canonical_names = sorted(set(canonical_for_abbr.values()))
    fuzzy_assignment = {}  # raw_name -> canonical_name
    still_unresolved = []

    for name in unresolved:
        best_match, best_score = None, 0.0
        for cand in canonical_names:
            if has_distinguishing_modifier(name, cand):
                continue
            score = difflib.SequenceMatcher(None, name.lower(), cand.lower()).ratio()
            if score > best_score:
                best_match, best_score = cand, score
        if best_score >= FUZZY_THRESHOLD:
            fuzzy_assignment[name] = best_match
            review_rows.append({
                "cluster_key": f"fuzzy->{best_match}",
                "canonical_chosen": best_match,
                "raw_variants": name,
                "reason": f"fuzzy match to existing canonical, score={best_score:.2f}",
            })
        else:
            still_unresolved.append(name)

    # Fuzzy-cluster whatever's left against each other (greedy).
    remaining = list(still_unresolved)
    self_clusters = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        rest = []
        for other in remaining:
            if has_distinguishing_modifier(seed, other):
                rest.append(other)
                continue
            score = difflib.SequenceMatcher(None, seed.lower(), other.lower()).ratio()
            if score >= FUZZY_THRESHOLD:
                cluster.append(other)
            else:
                rest.append(other)
        remaining = rest
        self_clusters.append(cluster)
        if len(cluster) > 1:
            canon = max(cluster, key=lambda m: counts[m])
            review_rows.append({
                "cluster_key": f"selffuzzy:{canon}",
                "canonical_chosen": canon,
                "raw_variants": " | ".join(sorted(cluster)),
                "reason": "fuzzy-matched to each other, no abbreviation available",
            })

    # Assemble final raw_name -> canonical_name map
    raw_to_canonical = {}
    for abbr, members in abbr_clusters.items():
        for m in members:
            raw_to_canonical[m] = canonical_for_abbr[abbr]
    for name, canon in fuzzy_assignment.items():
        raw_to_canonical[name] = canon
    for cluster in self_clusters:
        canon = max(cluster, key=lambda m: counts[m])
        for m in cluster:
            raw_to_canonical[m] = canon

    return raw_to_canonical, review_rows, counts


def main():
    raw_names_with_src = load_raw_tool_names()
    raw_to_canonical, review_rows, counts = build_clusters(raw_names_with_src)

    canonical_counts = Counter()
    for name, _src in raw_names_with_src:
        canonical_counts[raw_to_canonical[name]] += 1

    unique_canonical = len(canonical_counts)

    with open("postprocessing/tool_normalization_review.csv", "w", encoding="utf-8", newline="") as f:
        review_df = pd.DataFrame(review_rows)
        review_df.to_csv(f, index=False)

    freq_df = pd.DataFrame(
        sorted(canonical_counts.items(), key=lambda kv: -kv[1]),
        columns=["canonical_tool_name", "frequency"],
    )
    freq_df.to_csv("postprocessing/tool_frequency_normalized.csv", index=False, encoding="utf-8")

    print(f"Raw tool_name occurrences: {len(raw_names_with_src)}")
    print(f"Raw unique strings: {len(set(n for n, _ in raw_names_with_src))}")
    print(f"Canonical unique tools after normalization: {unique_canonical}")
    print(f"Merge decisions written for review: {len(review_rows)}")
    print("Wrote postprocessing/tool_frequency_normalized.csv and tool_normalization_review.csv")


if __name__ == "__main__":
    main()
