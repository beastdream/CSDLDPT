"""Unit tests for retrieval precision, recall, F1, AP and mAP metrics."""

import unittest

from src.evaluation import (
    average_precision,
    evaluate_query,
    f1_score,
    mean_average_precision,
    precision_at_k,
    recall_at_k,
)


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
            "f1_at_5",
            "f1_at_10",
            "f1_at_20",
        }
        self.assertEqual(set(metrics), expected)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in metrics.values()))

    def test_evaluate_query_includes_ap_on_request(self) -> None:
        results = [{"category": c} for c in ["flowers", "food", "flowers", "food", "food"]]
        metrics = evaluate_query("flowers", results, ks=(5,), total_relevant=2, include_ap=True)
        self.assertAlmostEqual(metrics["average_precision"], (1 / 1 + 2 / 3) / 2)
        self.assertAlmostEqual(metrics["f1_at_5"], f1_score(0.4, 1.0))

    def test_f1_score(self) -> None:
        self.assertAlmostEqual(f1_score(0.5, 0.5), 0.5)
        self.assertAlmostEqual(f1_score(1.0, 0.5), 2 / 3)
        self.assertEqual(f1_score(0.0, 0.0), 0.0)

    def test_average_precision_perfect_ranking(self) -> None:
        ranking = ["flowers"] * 3 + ["food"] * 5
        self.assertAlmostEqual(average_precision("flowers", ranking, 3), 1.0)

    def test_average_precision_known_value(self) -> None:
        ranking = ["food", "flowers", "food", "flowers"]
        self.assertAlmostEqual(
            average_precision("flowers", ranking, 2), (1 / 2 + 2 / 4) / 2
        )

    def test_average_precision_missing_relevant_counts_as_zero(self) -> None:
        self.assertAlmostEqual(average_precision("flowers", ["flowers", "food"], 4), 0.25)

    def test_average_precision_invalid_total(self) -> None:
        with self.assertRaises(ValueError):
            average_precision("flowers", ["flowers", "flowers"], 1)
        with self.assertRaises(ValueError):
            average_precision("flowers", ["flowers"], 0)

    def test_mean_average_precision(self) -> None:
        self.assertAlmostEqual(mean_average_precision([1.0, 0.5, 0.0]), 0.5)
        with self.assertRaises(ValueError):
            mean_average_precision([])


if __name__ == "__main__":
    unittest.main()
