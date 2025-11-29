import json
import os

# Richer query-to-field mapping
QUERY_FIELDS = {
    "Does the article study any suicide screening/assessment tools?": ["studies_tool"],
    "Which suicide screening/assessment tool is studied?": ["tool_name"],
    "Classify if the tool is screening or assessment.": ["tool_type"],
    "Discuss the study outcome.": ["outcome_summary"],
    "Discuss clinical settings where the tool is used.": ["clinical_setting"],
    "Discuss demographics of participants.": ["demographics_summary", "population_size", "population_text"],
    "Where was the study conducted?": ["location"],
    "Discuss majority medical conditions.": ["medical_conditions"],
    "Discuss study duration and population size.": ["duration_value", "duration_text", "population_size", "population_text"],
}


class QueryScopedNormalizer:
    """
    Converts a free-form answer into JSON containing only fields relevant to the query.
    Uses QUERY_FIELDS mapping to build prompts dynamically.
    """
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def normalize_query(self, query: str, answer: str) -> dict:
        fields = QUERY_FIELDS.get(query, [])
        schema_lines = "\n".join([
            f"- {f}: " + (
                "string/integer/null" if f not in {"duration_value", "population_size"} else "integer/null"
            )
            for f in fields
        ])

        # Build dynamic prompt
        prompt = f"""
You are a strict information extractor. Convert the following answer into JSON with ONLY these fields:

{schema_lines}

Rules:
- Do NOT guess. Use the provided answer text only.
- For studies_tool: output "yes" or "no" (or null if truly unknown).
- For tool_name: return free-text (string or list of strings if multiple tools). Do not alias or templatize.
- For tool_type: "screening" or "assessment" if explicitly stated; else null.
- For outcome_summary: 3–4 sentences summarizing findings (effectiveness, limitations, key metrics).
- For demographics_summary: 3–4 sentences summarizing participant characteristics (age, gender, group).
- For clinical_setting: concise free-text (e.g., "pediatric emergency department", "primary care clinics").
- For location: concise free-text (e.g., "USA", "urban hospitals in India").
- For medical_conditions: concise free-text (e.g., "major depressive disorder", "suicidal ideation").
- For duration_value: integer months if possible (convert years/weeks to months).
- For duration_text: concise narrative (e.g., "12 months (Jan–Dec 2022)").
- For population_size: integer N if available.
- For population_text: concise narrative (e.g., "N=452 pediatric ED patients ages 12–17").

Return ONLY valid JSON. No prose.

Answer:
{answer}
"""

        response = self.llm_client.chat_completion([
            {"role": "system", "content": "You output ONLY valid JSON with the requested fields."},
            {"role": "user", "content": prompt}
        ], temperature=0, max_tokens=600)

        try:
            data = json.loads(response)
        except Exception:
            data = {}

        # Filter only allowed fields
        return {k: v for k, v in data.items() if k in fields}


class AnswerAccumulator:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.partial_answers = {}            # base_field -> normalized value
        self.field_provenance = {}           # base_field -> list of queries that contributed
        self.query_history = {}              # query -> partial dict (for audit/debug)

    def update(self, query: str, partial: dict) -> list:
        """
        Update accumulator with normalized partial answers produced by `query`.
        Returns the list of base_fields updated.
        """
        if partial is None:
            partial = {}

        updated_fields = []
        # store query -> partial for audit/debug
        self.query_history[query] = dict(partial)

        for field, value in partial.items():
            self.partial_answers[field] = value
            updated_fields.append(field)

            # Track which queries contributed to each field (provenance)
            self.field_provenance.setdefault(field, []).append(query)

        return updated_fields

    def get_partial_answers(self) -> dict:
        return dict(self.partial_answers)

    def get_field_provenance(self) -> dict:
        return {f: list(set(qs)) for f, qs in self.field_provenance.items()}  # dedup queries
    

class FinalRecordAssembler:
    """
    Merges normalized base_fields into standardized records.
    Handles:
      - No tool studied (short-circuit)
      - Multiple tools studied (flatten into multiple rows)
      - Rich fields for outcome, demographics, duration, population
    """

    BASE_FIELDS = [
        "studies_tool",          # "yes"/"no"/null
        "tool_name",             # str or [str,...], free-text, no alias lock-in
        "tool_type",             # "screening"/"assessment"/null
        "outcome_summary",       # 3–4 sentence narrative
        "clinical_setting",      # concise free-text
        "demographics_summary",  # 3–4 sentence narrative
        "location",              # concise free-text
        "duration_value",        # integer months if possible
        "duration_text",         # narrative
        "population_size",       # integer N if available
        "population_text",       # narrative
        "medical_conditions",    # concise free-text
    ]

    def assemble(self, doc_id: str, accumulator: "AnswerAccumulator") -> list[dict]:
        # Initialize base record with all fields set to None
        base_record = {f: None for f in self.BASE_FIELDS}
        base_record["doc_id"] = doc_id

        # Merge normalized answers from accumulator.partial_answers
        for k, v in accumulator.partial_answers.items():
            if v not in (None, "", []):
                base_record[k] = v

        # Case 1: No tool studied → return single record with nulls
        if self._canon_bool_str(base_record.get("studies_tool")) == "no":
            base_record["studies_tool"] = "no"
            return [base_record]

        # Case 2: Multiple tools studied → flatten into multiple rows
        tool_names = base_record.get("tool_name")
        if isinstance(tool_names, list) and tool_names:
            records = []
            for tool in tool_names:
                rec = dict(base_record)
                rec["studies_tool"] = "yes"
                rec["tool_name"] = self._canon_tool_name(tool)
                rec["tool_type"] = self._canon_tool_type(rec["tool_type"])
                rec["population_size"] = self._canon_int(rec["population_size"])
                rec["duration_value"] = self._canon_int(rec["duration_value"])
                records.append(rec)
            return records

        # Case 3: Single tool studied → return one record
        base_record["studies_tool"] = self._canon_bool_str(base_record.get("studies_tool"))
        base_record["tool_name"] = self._canon_tool_name(base_record.get("tool_name"))
        base_record["tool_type"] = self._canon_tool_type(base_record.get("tool_type"))
        base_record["population_size"] = self._canon_int(base_record.get("population_size"))
        base_record["duration_value"] = self._canon_int(base_record.get("duration_value"))

        return [base_record]

    # --- Canonicalization helpers ---
    def _canon_bool_str(self, x):
        if not x: return None
        s = str(x).strip().lower()
        if s in {"yes", "true", "y"}: return "yes"
        if s in {"no", "false", "n"}: return "no"
        return None

    def _canon_tool_name(self, name):
        if not name: return None
        return str(name).strip()  # ✅ free-text only, no alias lock-in

    def _canon_tool_type(self, t):
        if not t: return None
        t = str(t).strip().lower()
        if "screen" in t: return "screening"
        if "assess" in t: return "assessment"
        return None

    def _canon_int(self, x):
        if x is None: return None
        try:
            return int(str(x).replace(",", "").strip())
        except Exception:
            return None


class AuditLogger:
    """
    Writes raw answers and normalized partials to a JSONL file for auditability.
    Each entry is valid JSON on its own line (for parsing),
    followed by extra blank lines for readability.
    """
    def __init__(self, log_folder="audit_logs"):
        os.makedirs(log_folder, exist_ok=True)
        self.log_file = os.path.join(log_folder, "audit.jsonl")

    def log(self, doc_id: str, query: str, answer: str, normalized: dict):
        entry = {
            "doc_id": doc_id,
            "query": query,
            "raw_answer": answer.strip(),
            "normalized": normalized
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            # Write JSON entry (machine-readable)
            f.write(json.dumps(entry) + "\n")
            # Add extra blank lines (human-friendly spacing)
            f.write("\n\n")