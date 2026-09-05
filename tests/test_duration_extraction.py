import unittest

from answer_normalizer import QueryScopedNormalizer


class TestDurationExtraction(unittest.TestCase):
    """
    QueryScopedNormalizer._extract_duration_days is pure (no LLM call) and tries several
    parsing strategies in sequence: explicit "N days/weeks/months/years", full date-to-date
    ranges, "Month-Month Year" spans, and "Month Year to Month Year" spans. It's the most
    intricate piece of logic in the codebase, so it's worth pinning its behavior directly.
    """

    def setUp(self):
        self.normalizer = QueryScopedNormalizer(llm_client=None)

    def _days(self, text: str):
        return self.normalizer._extract_duration_days(text)

    def test_explicit_days(self):
        self.assertEqual(self._days("The study lasted 30 days."), 30)

    def test_explicit_months_uses_30_day_approximation(self):
        self.assertEqual(self._days("Data collection took 12 months."), 360)

    def test_explicit_years_uses_365_day_approximation(self):
        self.assertEqual(self._days("The trial ran for 2 years."), 730)

    def test_explicit_hyphenated_units(self):
        # "45-day trial" (adjective form, hyphen-joined) must match same as "45 days".
        self.assertEqual(self._days("The 45-day trial began in March 2020."), 45)
        self.assertEqual(self._days("A 12-month follow-up was conducted."), 360)

    def test_full_date_range(self):
        self.assertEqual(
            self._days("Data was collected from June 5, 2020 to June 21, 2020."), 17
        )

    def test_multi_month_range_single_year(self):
        self.assertEqual(self._days("The study spanned March-April 2021."), 61)  # 31 + 30
        self.assertEqual(self._days("The study spanned March to April 2021."), 61)

    def test_month_year_to_month_year_range(self):
        self.assertEqual(
            self._days("Enrollment ran from December 2019 to January 2020."), 32
        )

    def test_no_duration_information_returns_none(self):
        self.assertIsNone(self._days("No duration information is mentioned in this text."))

    def test_malformed_date_does_not_raise(self):
        # Not a real calendar date; must fail closed (None), not raise.
        result = self._days("The study ran from Blursday 45, 2020 to Blursday 46, 2020.")
        self.assertIsNone(result)

    def test_duration_restated_two_ways_is_not_double_counted(self):
        # A single duration is often restated in more than one way in the same sentence.
        # Regression test: this used to sum every matching strategy (explicit "6 months"
        # PLUS the "January 2020 to June 2020" range), landing at ~333 days instead of
        # the correct ~152-182 day span.
        result = self._days(
            "The study enrolled participants over 6 months, from January 2020 to June 2020."
        )
        self.assertLess(result, 200, "explicit-units and month-year-range matches must not both count")

    def test_explicit_units_take_priority_over_weaker_signals(self):
        # When an explicit duration is present, it should win even if a weaker pattern
        # (a bare "Month Year" mention elsewhere in the text) could also match.
        self.assertEqual(
            self._days("The trial lasted 45 days and began in March 2020."), 45
        )


if __name__ == "__main__":
    unittest.main()
