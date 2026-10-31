"""Tests for SQL syntax validation, safety checks, and schema grounding."""

import pytest
from src.validator import SQLValidator


@pytest.fixture(scope="module")
def validator():
    return SQLValidator()


def test_valid_simple_select(validator):
    sql = "SELECT id, name, email FROM customers WHERE segment = 'Corporate'"
    res = validator.validate(sql)
    assert res.is_valid
    assert len(res.violations) == 0
    assert "LIMIT" in res.transformed_sql.upper()


def test_valid_joins_and_ctes(validator):
    sql = """
    WITH high_qty_orders AS (
        SELECT order_id, SUM(quantity) AS total_items
        FROM order_items
        GROUP BY order_id
        HAVING SUM(quantity) > 10
    )
    SELECT o.id, c.name, h.total_items
    FROM orders o
    JOIN customers c ON o.customer_id = c.id
    JOIN high_qty_orders h ON o.id = h.order_id
    WHERE o.status = 'Completed'
    """
    res = validator.validate(sql)
    assert res.is_valid
    assert len(res.violations) == 0


def test_reject_multiple_statements(validator):
    sql = "SELECT * FROM customers; DROP TABLE orders;"
    res = validator.validate(sql)
    assert not res.is_valid
    assert any("Multiple SQL statements" in v for v in res.violations)


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO categories (name) VALUES ('Test')",
        "UPDATE customers SET name = 'Hacked' WHERE id = 1",
        "DELETE FROM orders WHERE id = 1",
        "DROP TABLE payments",
        "ALTER TABLE products ADD COLUMN secret TEXT",
        "PRAGMA table_info(customers)",
    ],
)
def test_reject_forbidden_dml_ddl(validator, sql):
    res = validator.validate(sql)
    assert not res.is_valid
    assert any("Forbidden SQL operation" in v or "Only read-only SELECT" in v for v in res.violations)


def test_reject_unknown_table(validator):
    sql = "SELECT user_id, password_hash FROM user_logins"
    res = validator.validate(sql)
    assert not res.is_valid
    assert any("Unknown table 'user_logins'" in v for v in res.violations)


def test_reject_unknown_column_qualified(validator):
    sql = "SELECT c.id, c.credit_card_number FROM customers c"
    res = validator.validate(sql)
    assert not res.is_valid
    assert any("Column 'credit_card_number' does not exist in table 'customers'" in v for v in res.violations)


def test_reject_unknown_column_unqualified(validator):
    sql = "SELECT non_existent_column_xyz FROM customers"
    res = validator.validate(sql)
    assert not res.is_valid
    assert any("not found in referenced tables" in v for v in res.violations)


def test_preserves_aggregate_without_injecting_unnecessary_limit(validator):
    # Pure aggregate query without GROUP BY returns exactly 1 row
    sql = "SELECT COUNT(*) AS total_count FROM customers"
    res = validator.validate(sql)
    assert res.is_valid
    assert "LIMIT" not in res.transformed_sql.upper()

