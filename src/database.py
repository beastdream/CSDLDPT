"""MySQL connection management using project environment variables."""

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, Iterator

import mysql.connector
from mysql.connector import Error as MySQLError
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


class DatabaseConfigError(ValueError):
    """Raised when the database environment configuration is invalid."""


def _database_config() -> dict[str, Any]:
    """Load and validate MySQL settings from the project .env file."""
    load_dotenv(ENV_PATH)
    env_names = {
        "host": "DB_HOST",
        "port": "DB_PORT",
        "user": "DB_USER",
        "password": "DB_PASSWORD",
        "database": "DB_NAME",
    }
    missing = [name for name in env_names.values() if not os.getenv(name)]
    if missing:
        raise DatabaseConfigError(
            f"Missing database settings in {ENV_PATH.name}: {', '.join(missing)}"
        )

    password = os.environ["DB_PASSWORD"]
    if password == "your_password":
        raise DatabaseConfigError(
            "DB_PASSWORD still uses the placeholder value. Update .env with your "
            "local MySQL password."
        )

    try:
        port = int(os.environ["DB_PORT"])
    except ValueError as exc:
        raise DatabaseConfigError("DB_PORT must be an integer.") from exc

    return {
        "host": os.environ["DB_HOST"],
        "port": port,
        "user": os.environ["DB_USER"],
        "password": password,
        "database": os.environ["DB_NAME"],
    }


def get_connection():
    """Create and verify a MySQL connection from the project .env settings."""
    try:
        connection = mysql.connector.connect(**_database_config())
    except DatabaseConfigError:
        raise
    except MySQLError as exc:
        raise ConnectionError(
            "Could not connect to MySQL. Check that the server is running and "
            f"the credentials in {ENV_PATH.name} are correct. MySQL error: {exc}"
        ) from exc

    if not connection.is_connected():
        connection.close()
        raise ConnectionError("MySQL connection was created but is not active.")
    return connection


@contextmanager
def connection_scope() -> Iterator[Any]:
    """Yield an active connection and always close it on context exit."""
    connection = get_connection()
    try:
        yield connection
    finally:
        if connection.is_connected():
            connection.close()
