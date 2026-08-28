"""Unit tests for retrieval precision and recall metrics."""

import unittest

from src.evaluation import evaluate_query, precision_at_k, recall_at_k


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.categories = ["flowers", "flowers", "food", "flowers", "beach"]

    def test_precision_basic(self) -> None:
        self.assertAlmostEqual(precision_at_k("flowers", self.categories, 5), 0.6)

    def test_recall_basic(self) -> None:
        self.assertAlmostEqual(
            recall_at_k("flowers", self.categories, 5, 99), 3 / 99
        )

    def test_all_relevant(self) -> None:
        categories = ["flowers"] * 5
        self.assertEqual(precision_at_k("flowers", categories, 5), 1.0)

    def test_no_relevant(self) -> None:
        categories = ["food", "beach", "buses", "africa", "horses"]
        self.assertEqual(precision_at_k("flowers", categories, 5), 0.0)
        self.assertEqual(recall_at_k("flowers", categories, 5, 99), 0.0)

    def test_invalid_k(self) -> None:
        for k in (0, -1):
            with self.subTest(k=k), self.assertRaises(ValueError):
                precision_at_k("flowers", self.categories, k)

    def test_results_shorter_than_k(self) -> None:
        with self.assertRaises(ValueError):
            precision_at_k("flowers", self.categories, 10)
        with self.assertRaises(ValueError):
            recall_at_k("flowers", self.categories, 10, 99)

    def test_invalid_total_relevant(self) -> None:
        for total in (0, -1):
            with self.subTest(total=total), self.assertRaises(ValueError):
                recall_at_k("flowers", self.categories, 5, total)

    def test_evaluate_query_required_metrics(self) -> None:
        categories = (
            ["flowers", "flowers", "food", "flowers", "beach"]
            + ["food"] * 15
        )
        results = [{"category": category} for category in categories]
        metrics = evaluate_query("flowers", results)
        expected = {
            "precision_at_5",
            "precision_at_10",
            "precision_at_20",
            "recall_at_5",
            "recall_at_10",
            "recall_at_20",
        }
        self.assertEqual(set(metrics), expected)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
