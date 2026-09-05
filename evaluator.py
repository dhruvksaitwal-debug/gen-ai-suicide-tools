import json
import logging
import os
import threading
from typing import Any, Dict, List
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI

from config import load_openai_config
from constants import DEFAULT_MODEL_NAME, RAGAS_MAX_WORKERS

logger = logging.getLogger(__name__)

# ragas.evaluate() patches the asyncio event loop (nest_asyncio) by default, which is not
# designed for concurrent calls from multiple threads. Serialize evaluation calls so the
# rest of the pipeline (extraction/retrieval/answering) can still run concurrently across PDFs.
_ragas_lock = threading.Lock()


def load_gold_examples(path: str = "gold_data/gold_examples.json") -> Dict[str, Dict[str, str]]:
    """Load gold examples from a JSON file."""
    if not os.path.exists(path):
        logger.warning("Gold examples file not found at %s", path)
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
    def __init__(self, gold_examples: Dict[str, Dict[str, str]], model_name: str = DEFAULT_MODEL_NAME):
        self.gold_examples = gold_examples

        config = load_openai_config()
        self.llm = ChatOpenAI(model=model_name, api_key=config.api_key, base_url=config.base_url)

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

        with _ragas_lock:
            results = evaluate(
                dataset=hf_dataset,
                metrics=[faithfulness, answer_relevancy, context_recall],
                llm=self.llm,   # pass GPT‑4o‑mini
                run_config=RunConfig(max_workers=RAGAS_MAX_WORKERS),
            )
        return results.to_pandas()