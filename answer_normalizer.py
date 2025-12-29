import json
import os
import re
from datetime import datetime
import calendar


QUERY_FIELDS = {
    "Does the article study any suicide screening/assessment tools?": ["studies_tool"],
    "Which suicide screening/assessment tool is studied?": ["tool_name"],
    "Classify if the tool is screening or assessment.": ["tool_type"],
    "Discuss the study outcome with the tool analyzed.": ["outcome_summary"],
    "Discuss clinical settings where the tool is used.": ["clinical_setting"],
    "Discuss demographics of participants for whom the tool is used.": ["demographics_summary", "population_size", "population_text"],
    "The geographic locations or countries where the study was conducted.": ["location"],
    "Discuss intended medical conditions of the patients in the study.": ["medical_conditions"],
    "Discuss the study duration and population size.": ["duration_value", "duration_text", "population_size", "population_text"],
}


class QueryScopedNormalizer:
    def __init__(self, llm_client, debug: bool = False):
        self.llm_client = llm_client
        self.debug = debug

    def _extract_duration_days(self, text: str) -> int | None:
        """
        Extract explicit durations (e.g., '30 days', '12 months') or compute from date ranges.
        Returns total duration in days if possible, else None.
        """
        total_days = 0

        # Step 1: Explicit durations
        duration_pattern = r"(\d+)\s*(days?|weeks?|months?|years?)"
        matches = re.findall(duration_pattern, text, flags=re.IGNORECASE)
        if self.debug:
            print(f"[DEBUG] Explicit duration matches: {matches}")

        for num, unit in matches:
            num = int(num)
            unit = unit.lower()
            if "day" in unit:
                total_days += num
            elif "week" in unit:
                total_days += num * 7
            elif "month" in unit:
                total_days += num * 30  # approximate
            elif "year" in unit:
                total_days += num * 365  # approximate

        # Step 2: Date ranges (support short and long month names)
        date_pattern = r"([A-Za-z]{3,9} \d{1,2}, \d{4})"
        date_matches = re.findall(date_pattern, text)
        if self.debug:
            print(f"[DEBUG] Date matches: {date_matches}")

        if len(date_matches) >= 2:
            for i in range(0, len(date_matches) - 1, 2):
                parsed = False
                for fmt in ("%B %d, %Y", "%b %d, %Y"):  # long and short month names
                    try:
                        start = datetime.strptime(date_matches[i], fmt)
                        end = datetime.strptime(date_matches[i + 1], fmt)
                        days = (end - start).days + 1
                        total_days += days
                        parsed = True
                        if self.debug:
                            print(f"[DEBUG] Parsed range {date_matches[i]} – {date_matches[i+1]} = {days} days")
                        break
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Failed parsing {date_matches[i]} – {date_matches[i+1]} with {fmt}: {e}")
                if not parsed and self.debug:
                    print(f"[DEBUG] Could not parse date range: {date_matches[i]} – {date_matches[i+1]}")

        # Step 3: Month–Month ranges with a single year (e.g., "March–April 2021")
        multi_month_pattern = r"([A-Za-z]{3,9})(?:\s*(?:-|to|through|and|,)\s*([A-Za-z]{3,9}))+?\s+(\d{4})"
        mm_matches = re.findall(multi_month_pattern, text, flags=re.IGNORECASE)

        if self.debug:
            print(f"[DEBUG] Multi-month matches: {mm_matches}")

        for m1, m_last, year in mm_matches:
            try:
                # Parse first and last month
                start_month = datetime.strptime(f"{m1} {year}", "%B %Y").month \
                    if len(m1) > 3 else datetime.strptime(f"{m1} {year}", "%b %Y").month

                end_month = datetime.strptime(f"{m_last} {year}", "%B %Y").month \
                    if len(m_last) > 3 else datetime.strptime(f"{m_last} {year}", "%b %Y").month

                # Compute full-month durations
                days = 0
                for month in range(start_month, end_month + 1):
                    days += calendar.monthrange(int(year), month)[1]

                total_days += days

                if self.debug:
                    print(f"[DEBUG] Full multi-month span {m1}-{m_last} {year} = {days} days")

            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Could not parse multi-month span {m1}-{m_last} {year}: {e}")

        # Step 4: Month–Year ranges (e.g., "December 2019 to January 2020")
        month_year_pattern = r"([A-Za-z]{3,9}) (\d{4})"
        my_matches = re.findall(month_year_pattern, text)

        if self.debug:
            print(f"[DEBUG] Month-year matches: {my_matches}")

        # We need pairs: (Dec 2019) → (Jan 2020)
        if len(my_matches) >= 2:
            for i in range(0, len(my_matches) - 1, 2):
                start_str = f"{my_matches[i][0]} 1, {my_matches[i][1]}"
                end_str = f"{my_matches[i+1][0]} 1, {my_matches[i+1][1]}"

                parsed = False
                for fmt in ("%B %d, %Y", "%b %d, %Y"):
                    try:
                        start = datetime.strptime(start_str, fmt)
                        end = datetime.strptime(end_str, fmt)
                        days = (end - start).days + 1
                        total_days += days
                        parsed = True
                        if self.debug:
                            print(f"[DEBUG] Parsed month-year range {start_str} – {end_str} = {days} days")
                        break
                    except:
                        continue

                if not parsed and self.debug:
                    print(f"[DEBUG] Could not parse month-year range: {start_str} – {end_str}")

        if self.debug:
            print(f"[DEBUG] Total computed days: {total_days}")

        return total_days if total_days > 0 else None
  
    def _normalize_text(self, text: str) -> str:
        """Fix encoding issues and replace en-dash with hyphen."""
        if not text:
            return text
        try:
            text = text.encode("latin1").decode("utf-8")
        except Exception:
            pass
        return text.replace("–", "-")

    def normalize_query(self, query: str, answer: str) -> dict:
        fields = QUERY_FIELDS.get(query, [])

        schema_lines = "\n".join([
            f"- {f}: " + (
                "string/integer/null"
                if f not in {"population_size"}
                else "integer/null"
            )
            for f in fields
        ])

        prompt = f"""
You are a strict information extractor. Convert the following answer into JSON with ONLY these fields:

{schema_lines}

Rules:
- Do NOT invent information. Use explicit mentions or clearly implied categories from the answer text.
- For studies_tool: output "yes" or "no" (or null if truly unknown).
- For tool_name: extract any explicitly named tool (e.g., "Comprehensive Suicide Risk Evaluation (CSRE)"), even if the text later says no specific tool is studied. If multiple tools are mentioned, return a list.
- For tool_type: "screening" or "assessment" if explicitly stated; else null.
- For outcome_summary: 3–4 sentences summarizing findings (effectiveness, limitations, key metrics).
- For demographics_summary: 3–4 sentences summarizing participant characteristics (age, gender, group).
- For clinical_setting: concise free-text (e.g., "pediatric emergency department", "primary care clinics").
- For location: concise free-text (e.g., "USA", "urban hospitals in India").
- For medical_conditions: extract all explicitly or implicitly mentioned conditions (e.g., "cancer-related conditions, general medical and surgical issues").
- For duration_value: if date ranges are provided, calculate the integer number of days, months, or years (e.g., June 5–21, 2020 → 17 days). If multiple phases, sum them. Extract ONLY the duration of study itself. 
- For duration_text: concise narrative (e.g., "Phase I: June 5–21, 2020; Phase II: Dec 6–13, 2020"). Normalize to use plain hyphens (-).
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

        if "duration_value" in QUERY_FIELDS.get(query, []):
            # 1. Always use the ORIGINAL GenAI answer (never rewritten text)
            raw_text = answer
            if self.debug:
                print("[DEBUG] Duration extraction: using raw GenAI answer")

            # 2. Compute duration safely
            try:
                computed_days = self._extract_duration_days(raw_text)
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Duration extraction error: {e}")
                computed_days = None

            # 3. Assign duration_value explicitly
            if computed_days is not None:
                data["duration_value"] = computed_days
                if self.debug:
                    print(f"[DEBUG] Duration extracted (days): {computed_days}")
            else:
                data["duration_value"] = None
                if self.debug:
                    print("[DEBUG] No duration extracted → setting duration_value=None")

        # Normalize duration_text (if present)
        if "duration_text" in fields:
            # Ensure the field exists even if the LLM omitted it
            duration_text = data.get("duration_text", None)
            if duration_text:
                data["duration_text"] = self._normalize_text(duration_text)
                if self.debug:
                    print(f"[DEBUG] Normalized duration_text: {data['duration_text']}")
            else:
                data["duration_text"] = None
                if self.debug:
                    print("[DEBUG] duration_text missing → setting duration_text=None")

        # Return only the fields required by the schema
        return {k: data.get(k, None) for k in fields}


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

        # Enforce: if studies_tool == "yes" but tool_name is missing → assign fallback
        if self._canon_bool_str(base_record.get("studies_tool")) == "yes" and not base_record.get("tool_name"):
            base_record["tool_name"] = "unspecified_tool"

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