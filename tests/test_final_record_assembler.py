import unittest

from answer_normalizer import AnswerAccumulator, FinalRecordAssembler


def _accumulator_with(doc_id: str, **fields) -> AnswerAccumulator:
    acc = AnswerAccumulator(doc_id=doc_id)
    acc.update("q", fields)
    return acc


class TestFinalRecordAssemblerNoTool(unittest.TestCase):
    def test_no_tool_short_circuits_to_single_null_record(self):
        acc = _accumulator_with("doc1", studies_tool="no")
        records = FinalRecordAssembler().assemble("doc1", acc)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["studies_tool"], "no")
        self.assertIsNone(records[0]["tool_name"])
        self.assertEqual(records[0]["doc_id"], "doc1")


class TestFinalRecordAssemblerSingleTool(unittest.TestCase):
    def test_single_tool_canonicalizes_fields(self):
        acc = _accumulator_with(
            "doc1",
            studies_tool="yes",
            tool_name="  PHQ-9  ",
            tool_type="Screening Tool",
            population_size="1,234",
            duration_value="45",
        )
        records = FinalRecordAssembler().assemble("doc1", acc)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["studies_tool"], "yes")
        self.assertEqual(record["tool_name"], "PHQ-9")
        self.assertEqual(record["tool_type"], "screening")
        self.assertEqual(record["population_size"], 1234)
        self.assertEqual(record["duration_value"], 45)

    def test_missing_tool_name_falls_back_to_unspecified(self):
        acc = _accumulator_with("doc1", studies_tool="yes")
        records = FinalRecordAssembler().assemble("doc1", acc)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["tool_name"], "unspecified_tool")


class TestFinalRecordAssemblerMultiTool(unittest.TestCase):
    def test_multiple_tools_flatten_into_one_record_each(self):
        acc = _accumulator_with(
            "doc1",
            studies_tool="yes",
            tool_name=["PHQ-9", "C-SSRS"],
            tool_type="assessment",
        )
        records = FinalRecordAssembler().assemble("doc1", acc)

        self.assertEqual(len(records), 2)
        names = {r["tool_name"] for r in records}
        self.assertEqual(names, {"PHQ-9", "C-SSRS"})
        for record in records:
            self.assertEqual(record["studies_tool"], "yes")
            self.assertEqual(record["tool_type"], "assessment")


class TestCanonicalizationHelpers(unittest.TestCase):
    def setUp(self):
        self.assembler = FinalRecordAssembler()

    def test_canon_bool_str(self):
        self.assertEqual(self.assembler._canon_bool_str("TRUE"), "yes")
        self.assertEqual(self.assembler._canon_bool_str("Y"), "yes")
        self.assertEqual(self.assembler._canon_bool_str("no"), "no")
        self.assertEqual(self.assembler._canon_bool_str("N"), "no")
        self.assertIsNone(self.assembler._canon_bool_str("maybe"))
        self.assertIsNone(self.assembler._canon_bool_str(None))

    def test_canon_tool_type(self):
        self.assertEqual(self.assembler._canon_tool_type("Screening"), "screening")
        self.assertEqual(self.assembler._canon_tool_type("Assessment Tool"), "assessment")
        self.assertIsNone(self.assembler._canon_tool_type("other"))
        self.assertIsNone(self.assembler._canon_tool_type(None))

    def test_canon_int(self):
        self.assertEqual(self.assembler._canon_int("1,234"), 1234)
        self.assertEqual(self.assembler._canon_int(42), 42)
        self.assertIsNone(self.assembler._canon_int("abc"))
        self.assertIsNone(self.assembler._canon_int(None))


if __name__ == "__main__":
    unittest.main()
