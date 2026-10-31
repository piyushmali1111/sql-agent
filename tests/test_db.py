"""Tests for database creation, schema inspection, and read-only connection enforcement."""

import os
from pathlib import Path
import pytest
from sqlalchemy import text

from data.build_db import build_database
from src.db import (
    DatabaseError,
    QueryExecutionError,
    create_readonly_engine,
    execute_select,
    get_schema,
)


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("test_db")
    path = tmp_dir / "test_business.db"
    build_database(db_path=path, seed=42)
    return path


def test_database_builds_deterministically(tmp_path):
    p1 = tmp_path / "db1.db"
    p2 = tmp_path / "db2.db"
    build_database(db_path=p1, seed=123)
    build_database(db_path=p2, seed=123)

    assert p1.exists() and p2.exists()
    assert p1.stat().st_size == p2.stat().st_size


def test_schema_inspection(db_path):
    engine = create_readonly_engine(f"sqlite:///{db_path}")
    schema = get_schema(engine)

    expected_tables = {
        "categories",
        "customers",
        "employees",
        "order_items",
        "orders",
        "payments",
        "products",
        "suppliers",
    }
    assert set(schema.get_table_names()) == expected_tables

    # Check customers table columns
    cust_table = schema.tables["customers"]
    col_names = {c.name for c in cust_table.columns}
    assert "email" in col_names
    assert "signup_date" in col_names
    assert "segment" in col_names
    assert cust_table.primary_keys == ["id"]

    # Check orders foreign keys
    orders_table = schema.tables["orders"]
    fk_targets = {fk.referred_table for fk in orders_table.foreign_keys}
    assert "customers" in fk_targets
    assert "employees" in fk_targets


def test_readonly_connection_rejects_writes(db_path):
    engine = create_readonly_engine(f"sqlite:///{db_path}")

    # Attempt INSERT
    with pytest.raises(Exception) as exc_info:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO categories (name) VALUES ('Hacked Category');"))
            conn.commit()
    assert "readonly" in str(exc_info.value).lower()

    # Attempt DROP
    with pytest.raises(Exception) as exc_info:
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE customers;"))
            conn.commit()
    assert "readonly" in str(exc_info.value).lower()


def test_execute_select_safe(db_path):
    engine = create_readonly_engine(f"sqlite:///{db_path}")
    res = execute_select("SELECT id, name FROM categories ORDER BY id ASC", engine=engine)

    assert res.row_count == 8
    assert res.columns == ["id", "name"]
    assert not res.truncated


def test_execute_select_limit_truncation(db_path):
    engine = create_readonly_engine(f"sqlite:///{db_path}")
    res = execute_select("SELECT id, name FROM customers", limit=10, engine=engine)

    assert res.row_count == 10
    assert res.truncated is True


def test_execute_select_driver_error_wrapping(db_path):
    engine = create_readonly_engine(f"sqlite:///{db_path}")
    with pytest.raises(QueryExecutionError) as exc_info:
        execute_select("SELECT non_existent_column FROM non_existent_table", engine=engine)
    assert "failed" in str(exc_info.value).lower()

