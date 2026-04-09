import re
import json
from pathlib import Path
from difflib import SequenceMatcher

INPUT_MAP = "data/tool_name_map.raw.json"
OUTPUT_MAP = "data/tool_name_map.canonical.json"

# Normalization utilities
STOPWORDS = {
    "suicide", "suicidal", "questionnaire", "scale", "severity",
    "assessment", "screening", "ideation", "risk", "index",
    "inventory", "measure", "rating", "symptom", "mental",
    "health", "behavior", "behaviour", "behaviors", "behaviours"
}

ACRONYM_RE = re.compile(r"\(([A-Za-z0-9\-]+)\)")

def normalize_text(s: str) -> str:
    s = s.lower()
    s = s.replace("â€“", "-").replace("â€”", "-").replace("â€‘", "-")
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_acronym(name: str):
    """
    Extracts acronyms in two cases:
    1. Inside parentheses: (ASQ), (C-SSRS), (PHQ-9)
    2. The entire tool_name is itself an acronym: "ASQ", "C-SSRS", "PHQ9"
    """

    # Case A: acronym inside parentheses
    m = ACRONYM_RE.search(name)
    if m:
        raw = m.group(1).strip()
        ac = raw.replace("-", "").replace(" ", "").upper()
        if 1 <= len(ac) <= 10 and ac.isalnum():
            return ac

    # Case B: entire tool_name is an acronym
    raw = name.strip()

    # Normalize hyphens and uppercase
    ac = raw.replace("-", "").replace(" ", "").upper()

    # Must be short, uppercase, alphanumeric
    if 1 <= len(ac) <= 10 and ac.isalnum():
        # Reject if original contains lowercase letters
        if raw.upper() == raw:
            return ac

    return None

def remove_stopwords(text: str):
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def lexical_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# Stage 1 — Merge tool names with the same acronym
def merge_acronym_groups(tool_names):
    groups = {}
    for name in tool_names:
        ac = extract_acronym(name)
        if ac:
            groups.setdefault(ac, []).append(name)

    acronym_map = {}
    for ac, names in groups.items():
        if len(names) < 2:
            continue  # unique acronym → leave alone

        # canonical = longest + most complete name
        canonical = max(names, key=len)
        for n in names:
            acronym_map[n] = canonical

    return acronym_map


# Stage 3 — Lexical merging for names without acronyms
def lexical_merge(tool_names, acronym_map, threshold=0.90):
    no_acronym = [n for n in tool_names if n not in acronym_map]

    canonical_map = {}
    processed = set()
    for name in no_acronym:
        if name in processed:
            continue

        n_norm = remove_stopwords(normalize_text(name))
        group = [name]

        for other in no_acronym:
            if other == name or other in processed:
                continue

            o_norm = remove_stopwords(normalize_text(other))
            sim = lexical_similarity(n_norm, o_norm)
            if sim >= threshold:
                group.append(other)
                processed.add(other)

        canonical = max(group, key=len)
        for g in group:
            canonical_map[g] = canonical
        processed.add(name)

    return canonical_map


# Combine all maps
def build_final_map(tool_names):
    acronym_map = merge_acronym_groups(tool_names)
    lexical_map = lexical_merge(tool_names, acronym_map, threshold=0.90)

    final_map = {}
    for name in tool_names:
        if name in acronym_map:
            final_map[name] = acronym_map[name]
        elif name in lexical_map:
            final_map[name] = lexical_map[name]
        else:
            final_map[name] = name

    return final_map


def main():
    raw = json.loads(Path(INPUT_MAP).read_text(encoding="utf-8"))
    tool_names = list(raw.keys())

    canonical_map = build_final_map(tool_names)

    Path(OUTPUT_MAP).write_text(
        json.dumps(canonical_map, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Canonical tool map written to {OUTPUT_MAP} with {len(canonical_map)} entries")

if __name__ == "__main__":
    main()