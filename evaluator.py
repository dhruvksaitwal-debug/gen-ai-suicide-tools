import json
import os
from typing import Any, Dict, List
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def load_gold_examples(path: str = "gold_data/gold_examples.json") -> Dict[str, Dict[str, str]]:
    """Load gold examples from a JSON file."""
    if not os.path.exists(path):
        print(f"Gold examples file not found at {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sanitize_str(value: Any) -> str:
    return "" if value is None else str(value)


def _sanitize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


class Evaluator:
    def __init__(self, gold_examples: Dict[str, Dict[str, str]]):
        self.gold_examples = gold_examples

        # Load API key from .env
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        # Configure GPT‑4o‑mini via LangChain
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

    def build_dataset(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert flattened records into RAGAS dataset format.
        Each entry: {question, answer, contexts, gold, doc_id}
        """
        dataset: List[Dict[str, Any]] = []
        for record in records:
            doc_id = _sanitize_str(record.get("doc_id", "UnknownDoc"))
            question = _sanitize_str(record.get("question"))
            answer = _sanitize_str(record.get("answer"))
            contexts = _sanitize_list(record.get("contexts", []))

            # Lookup gold answer if available
            gold_record = self.gold_examples.get(doc_id, {})
            gold_answer = _sanitize_str(gold_record.get(question, ""))

            dataset.append({
                "doc_id": doc_id,
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "reference": gold_answer,
            })
        return dataset

    def evaluate(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        dataset = self.build_dataset(records)
        hf_dataset = Dataset.from_list(dataset)

        results = evaluate(
            dataset=hf_dataset,
            # metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness],
            metrics=[faithfulness, answer_relevancy, context_recall],
            llm=self.llm   # pass GPT‑4o‑mini
        )
        return results.to_pandas()