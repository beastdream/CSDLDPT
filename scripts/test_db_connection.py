"""Test the configured connection to the WANG MySQL database."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import connection_scope


def main() -> int:
    """Connect and print the selected database and category count."""
    try:
        with connection_scope() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute("SELECT DATABASE()")
                database = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM categories")
                category_count = cursor.fetchone()[0]
            finally:
                cursor.close()
    except Exception as exc:
        print(f"ERROR: MySQL connection test failed: {exc}")
        return 1

    print("Connected to MySQL successfully.")
    print(f"Database: {database}")
    print(f"Categories: {category_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
