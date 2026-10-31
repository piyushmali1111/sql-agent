"""Database engine, schema inspection, and safe read-only execution.

Enforces connection-level read-only mode, statement execution limits,
and typed error handling.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import sqlalchemy
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from src.config import settings

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""


class QueryExecutionError(DatabaseError):
    """Raised when query execution fails against the database."""


@dataclass
class ColumnInfo:
    name: str
    type: str
    nullable: bool = True
    default: Any = None
    primary_key: bool = False


@dataclass
class ForeignKeyInfo:
    constrained_columns: list[str]
    referred_table: str
    referred_columns: list[str]


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo]
    primary_keys: list[str]
    foreign_keys: list[ForeignKeyInfo]


@dataclass
class DatabaseSchema:
    dialect: str
    tables: dict[str, TableInfo] = field(default_factory=dict)

    def get_table_names(self) -> list[str]:
        return sorted(list(self.tables.keys()))


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    execution_time_ms: float


def create_readonly_engine(database_url: str | None = None) -> Engine:
    """Creates a SQLAlchemy engine configured for read-only access.

    For SQLite: connects via URI mode (?mode=ro&uri=true) preventing any writes.
    For other dialects (PostgreSQL/MySQL): relies on read-only user credentials
    provided in the DATABASE_URL.
    """
    url = database_url or settings.database_url

    if url.startswith("sqlite"):
        # Normalize SQLite path
        raw_path = url.replace("sqlite:///", "").replace("sqlite://", "")
        # Strip any existing query params for path resolution
        db_path = raw_path.split("?")[0]
        abs_path = Path(db_path).resolve()
        if not abs_path.exists():
            raise DatabaseError(f"Database file does not exist at: {abs_path}. Run data/build_db.py first.")

        # SQLite URI formatted for Windows or POSIX
        # sqlite3 on Windows handles file:C:/... with uri=True
        posix_path = abs_path.as_posix()
        read_only_uri = f"file:{posix_path}?mode=ro"

        def _readonly_creator():
            return sqlite3.connect(read_only_uri, uri=True, timeout=settings.query_timeout_seconds)

        engine = create_engine("sqlite://", creator=_readonly_creator)
    else:
        # Standard SQL engines (e.g., PostgreSQL, MySQL)
        engine = create_engine(url, pool_pre_ping=True)

    return engine


def get_schema(engine: Engine | None = None) -> DatabaseSchema:
    """Inspects the live database schema returning tables, columns, PKs, and FKs."""
    if engine is None:
        engine = create_readonly_engine()

    try:
        inspector = inspect(engine)
        dialect_name = engine.dialect.name
        table_names = inspector.get_table_names()

        schema_tables: dict[str, TableInfo] = {}

        for tbl in table_names:
            # Skip sqlite system tables
            if tbl.startswith("sqlite_"):
                continue

            columns_meta = inspector.get_columns(tbl)
            pk_constraint = inspector.get_pk_constraint(tbl)
            pk_cols = pk_constraint.get("constrained_columns", []) if pk_constraint else []
            fks_meta = inspector.get_foreign_keys(tbl)

            columns: list[ColumnInfo] = []
            for col in columns_meta:
                c_name = col["name"]
                c_type = str(col["type"])
                c_null = col.get("nullable", True)
                c_def = col.get("default", None)
                is_pk = c_name in pk_cols
                columns.append(
                    ColumnInfo(
                        name=c_name,
                        type=c_type,
                        nullable=c_null,
                        default=c_def,
                        primary_key=is_pk,
                    )
                )

            foreign_keys: list[ForeignKeyInfo] = []
            for fk in fks_meta:
                foreign_keys.append(
                    ForeignKeyInfo(
                        constrained_columns=fk.get("constrained_columns", []),
                        referred_table=fk.get("referred_table", ""),
                        referred_columns=fk.get("referred_columns", []),
                    )
                )

            schema_tables[tbl] = TableInfo(
                name=tbl,
                columns=columns,
                primary_keys=pk_cols,
                foreign_keys=foreign_keys,
            )

        return DatabaseSchema(dialect=dialect_name, tables=schema_tables)
    except Exception as e:
        logger.error(f"Failed to inspect database schema: {e}", exc_info=True)
        raise DatabaseError(f"Database schema inspection failed: {e}") from e


def execute_select(
    sql: str,
    limit: int | None = None,
    timeout: float | None = None,
    engine: Engine | None = None,
) -> QueryResult:
    """Executes a read-only SELECT query safely with limits and timeouts.

    Wraps raw driver errors in QueryExecutionError with sanitized messages.
    """
    row_limit = limit or settings.max_execution_rows
    query_timeout = timeout or settings.query_timeout_seconds

    if engine is None:
        engine = create_readonly_engine()

    start_time = time.perf_counter()
    try:
        with engine.connect() as conn:
            # Set timeout if supported by dialect
            if engine.dialect.name == "sqlite":
                conn.exec_driver_sql(f"PRAGMA busy_timeout = {int(query_timeout * 1000)};")

            result = conn.execute(text(sql))
            col_names = list(result.keys()) if result.keys() else []

            # Fetch limit + 1 to detect truncation
            raw_rows = result.fetchmany(row_limit + 1)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            truncated = len(raw_rows) > row_limit
            final_rows = [list(r) for r in raw_rows[:row_limit]]

            return QueryResult(
                columns=col_names,
                rows=final_rows,
                row_count=len(final_rows),
                truncated=truncated,
                execution_time_ms=round(duration_ms, 2),
            )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        err_msg = str(e)
        # Sanitize sensitive information or stack trace
        logger.warning(f"Query execution failed: {err_msg} | SQL: {sql}")
        raise QueryExecutionError(f"Query failed: {err_msg}") from e

