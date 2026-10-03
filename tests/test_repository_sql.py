"""Unit tests for repository SQL and CSV loading without a live MySQL server."""

from contextlib import contextmanager
import unittest
from unittest import mock

import numpy as np

import src.repository as repository


class FakeCursor:
    def __init__(self, rows=None, categories=None):
        self.rows = rows or []
        self.categories = categories or []
        self.executed: list[tuple[str, tuple]] = []
        self.lastrowid = 0

    def execute(self, query, params=()):
        self.executed.append((query, params))
        if query.lstrip().startswith("INSERT INTO experiment_runs"):
            self.lastrowid += 1

    def fetchall(self):
        last_query = self.executed[-1][0]
        return self.categories if "FROM categories" in last_query else self.rows

    def close(self):
        pass


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.in_transaction = False
        self.committed = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed += 1
        self.in_transaction = False

    def start_transaction(self):
        self.in_transaction = True

    def rollback(self):
        self.in_transaction = False


def fake_scope(connection):
    @contextmanager
    def scope():
        yield connection
    return scope


class RepositoryTests(unittest.TestCase):
    def test_get_global_features_filters_by_color_space(self) -> None:
        row = (7, "7.jpg", "data/7.jpg", "beach", *range(9))
        cursor = FakeCursor(rows=[row])
        with mock.patch.object(repository, "connection_scope", fake_scope(FakeConnection(cursor))):
            records = repository.get_global_features("hsv")

        query, params = cursor.executed[0]
        self.assertEqual(params, ("HSV", "GLOBAL"))
        self.assertIn("f.c1_mean", query)
        self.assertEqual(records[0]["image_id"], 7)
        np.testing.assert_allclose(records[0]["features"], np.arange(9))

    def test_get_global_features_rejects_nan(self) -> None:
        row = (7, "7.jpg", "data/7.jpg", "beach", float("nan"), *range(8))
        cursor = FakeCursor(rows=[row])
        with mock.patch.object(repository, "connection_scope", fake_scope(FakeConnection(cursor))):
            with self.assertRaises(ValueError):
                repository.get_global_features("RGB")

    def test_save_experiment_results(self) -> None:
        metrics = {
            **{f"{name}_at_{k}": 0.5 for name in ("precision", "recall", "f1") for k in (5, 10, 20)},
            "map": 0.4,
        }
        config = {"method": "RGB+HSV", "distance_metric": "cosine", "normalization": "zscore"}
        overall = [{**config, **metrics, "feature_dim": 18, "num_queries": 1000}]
        categories = [{**config, **metrics, "category": "beach", "num_queries": 100}]
        cursor = FakeCursor(categories=[("beach", 2)])
        connection = FakeConnection(cursor)
        with mock.patch.object(repository, "connection_scope", fake_scope(connection)):
            saved = repository.save_experiment_results("test", overall, categories)

        self.assertEqual(saved, 1)
        self.assertEqual(connection.committed, 2)
        run_params = cursor.executed[1][1]
        self.assertEqual(run_params[:6], ("test", "RGB+HSV", "cosine", "zscore", 18, 1000))
        self.assertEqual(run_params[-1], 0.4)
        category_params = cursor.executed[2][1]
        self.assertEqual(category_params[:3], (1, 2, 100))

    def test_load_features_from_csv_rgb(self) -> None:
        if not repository.feature_csv_path("RGB").is_file():
            self.skipTest("RGB feature CSV has not been extracted.")
        records = repository.load_features_from_csv("RGB")
        self.assertEqual(len(records), 1000)
        self.assertEqual(records[0]["features"].shape, (9,))

    def test_load_features_unknown_source(self) -> None:
        with self.assertRaises(ValueError):
            repository.load_features("RGB", source="sqlite")


if __name__ == "__main__":
    unittest.main()
