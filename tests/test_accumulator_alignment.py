import unittest
from suicide_safe_care.answer_normalizer import AnswerAccumulator

class TestAnswerAccumulatorAlignment(unittest.TestCase):
    def test_update_and_provenance(self):
        acc = AnswerAccumulator(doc_id="Gold1")

        # Simulate normalization of two queries
        q1 = "Which suicide screening tool is studied?"
        partial1 = {"tool": "PHQ-9"}
        fields1 = acc.update(q1, partial1)

        q2 = "Discuss the study outcome."
        partial2 = {"outcome": "Validated"}
        fields2 = acc.update(q2, partial2)

        # Check updated fields
        self.assertEqual(fields1, ["tool"])
        self.assertEqual(fields2, ["outcome"])

        # Check partial answers
        self.assertEqual(acc.get_partial_answers()["tool"], "PHQ-9")
        self.assertEqual(acc.get_partial_answers()["outcome"], "Validated")

        # Check provenance mapping
        provenance = acc.get_field_provenance()
        self.assertIn("tool", provenance)
        self.assertIn(q1, provenance["tool"])
        self.assertIn("outcome", provenance)
        self.assertIn(q2, provenance["outcome"])

    def test_context_alignment(self):
        acc = AnswerAccumulator(doc_id="Gold1")

        # Simulate normalization
        q1 = "Which suicide screening tool is studied?"
        acc.update(q1, {"tool": "PHQ-9"})
        q2 = "Discuss the study outcome."
        acc.update(q2, {"outcome": "Validated"})

        # Simulate contexts per query
        per_query_contexts = {
            q1: ["The study validated the PHQ-9 questionnaire."],
            q2: ["Results showed PHQ-9 had high sensitivity."]
        }

        # Flatten with alignment
        records = [{"doc_id": "Gold1", "tool": "PHQ-9", "outcome": "Validated"}]
        flattened = []
        for record in records:
            for field, answer in record.items():
                if field == "doc_id":
                    continue
                contexts = []
                for q in acc.get_field_provenance().get(field, []):
                    contexts.extend(per_query_contexts.get(q, []))
                flattened.append({
                    "doc_id": record["doc_id"],
                    "question": field,
                    "answer": answer,
                    "contexts": contexts
                })

        # Assertions
        tool_row = next(r for r in flattened if r["question"] == "tool")
        self.assertIn("PHQ-9 questionnaire", tool_row["contexts"][0])

        outcome_row = next(r for r in flattened if r["question"] == "outcome")
        self.assertIn("sensitivity", outcome_row["contexts"][0])

if __name__ == "__main__":
    unittest.main()