"""SQL safety, syntax parsing, and schema-grounding validator.

Uses sqlglot to enforce single-statement read-only execution, schema grounding
against live database tables and columns, and automatic LIMIT enforcement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.config import settings
from src.db import DatabaseSchema, create_readonly_engine, get_schema

logger = logging.getLogger(__name__)

FORBIDDEN_EXPRESSIONS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Command,
    exp.Pragma,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
)


@dataclass
class ValidationResult:
    is_valid: bool
    violations: list[str] = field(default_factory=list)
    transformed_sql: str | None = None


class SQLValidator:
    """Validates SQL statements against dialect syntax, read-only safety, and schema grounding."""

    def __init__(self, schema: DatabaseSchema | None = None, engine: Engine | None = None):
        self.engine = engine or create_readonly_engine()
        self.schema = schema or get_schema(self.engine)
        self.dialect = "sqlite" if self.schema.dialect == "sqlite" else self.schema.dialect
        self.known_tables = {k.lower(): set(c.name.lower() for c in v.columns) for k, v in self.schema.tables.items()}

    def validate(
        self,
        sql: str,
        dry_run: bool = True,
        enforce_limit: bool = True,
    ) -> ValidationResult:
        violations: list[str] = []

        if not sql or not sql.strip():
            return ValidationResult(is_valid=False, violations=["SQL query is empty."])

        clean_sql = sql.strip().rstrip(";")

        # 1. Parse with sqlglot for dialect
        try:
            statements = sqlglot.parse(clean_sql, read=self.dialect)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                violations=[f"SQL syntax error (failed to parse with dialect '{self.dialect}'): {e}"],
            )

        # 2. Exactly one statement
        if len(statements) == 0:
            return ValidationResult(is_valid=False, violations=["No valid SQL statements found."])
        if len(statements) > 1:
            violations.append(f"Multiple SQL statements ({len(statements)}) detected. Only a single statement is allowed.")

        stmt = statements[0]

        # 3. Read-only SELECT enforcement
        # Must be a Select, Union, or Subquery (with optional CTEs)
        for forbidden in FORBIDDEN_EXPRESSIONS:
            if list(stmt.find_all(forbidden)):
                violations.append(
                    f"Forbidden SQL operation detected: {forbidden.__name__}. Only read-only SELECT queries are permitted."
                )

        if not isinstance(stmt, (exp.Select, exp.Union)):
            # Check if root is CTE containing Select
            has_select = any(isinstance(n, exp.Select) for n in stmt.walk())
            if not has_select:
                violations.append(
                    f"Invalid statement type: '{type(stmt).__name__}'. Only read-only SELECT queries are permitted."
                )

        # 4. Schema Grounding Check
        # Identify CTE names defined in query
        cte_names = {cte.alias.lower() for cte in stmt.find_all(exp.CTE) if cte.alias}

        # Build alias map: table alias -> real table name (or CTE)
        alias_to_table: dict[str, str] = {}
        referenced_tables: set[str] = set()

        for tbl_exp in stmt.find_all(exp.Table):
            t_name = tbl_exp.name.lower()
            if not t_name:
                continue

            # Check if table exists in live schema or CTEs
            if t_name not in self.known_tables and t_name not in cte_names:
                violations.append(
                    f"Schema grounding violation: Unknown table '{t_name}' referenced. "
                    f"Valid tables are: {sorted(list(self.known_tables.keys()))}."
                )
            else:
                referenced_tables.add(t_name)

            alias = tbl_exp.alias.lower() if tbl_exp.alias else t_name
            alias_to_table[alias] = t_name
            alias_to_table[t_name] = t_name

        # Verify referenced columns
        # Collect aliases defined in SELECT expressions (e.g. SELECT col AS alias)
        select_aliases = {
            s.alias.lower()
            for s in stmt.find_all(exp.Alias)
            if s.alias
        }

        for col_exp in stmt.find_all(exp.Column):
            c_name = col_exp.name.lower()
            # Skip wildcard expressions like *
            if not c_name or c_name == "*":
                continue

            # If column has table qualification (e.g., c.name)
            t_alias = col_exp.table.lower() if col_exp.table else ""
            if t_alias:
                resolved_table = alias_to_table.get(t_alias)
                if resolved_table and resolved_table in self.known_tables:
                    valid_cols = self.known_tables[resolved_table]
                    if c_name not in valid_cols:
                        violations.append(
                            f"Schema grounding violation: Column '{col_exp.name}' does not exist in table '{resolved_table}'. "
                            f"Available columns: {sorted(list(valid_cols))}."
                        )
            else:
                # Unqualified column: must exist in at least one referenced real table, CTE, or select alias
                exists_in_tables = any(
                    c_name in self.known_tables[t]
                    for t in referenced_tables
                    if t in self.known_tables
                )
                if not exists_in_tables and c_name not in select_aliases and not cte_names:
                    violations.append(
                        f"Schema grounding violation: Unqualified column '{col_exp.name}' not found in referenced tables "
                        f"({sorted(list(referenced_tables))})."
                    )

        # 5. LIMIT enforcement on non-aggregate queries
        transformed_sql = clean_sql
        if enforce_limit and not violations:
            has_limit = stmt.args.get("limit") is not None
            has_group = bool(list(stmt.find_all(exp.Group)))
            has_agg = bool(list(stmt.find_all(exp.AggFunc)))

            # Pure aggregate query without GROUP BY returns 1 row; everything else gets LIMIT
            is_single_row_agg = has_agg and not has_group

            if not has_limit and not is_single_row_agg:
                # Add limit to statement
                stmt = stmt.limit(settings.max_execution_rows)
                transformed_sql = stmt.sql(dialect=self.dialect)

        # 6. Optional Dry Run (EXPLAIN QUERY PLAN)
        if dry_run and not violations:
            try:
                explain_query = f"EXPLAIN QUERY PLAN {transformed_sql}"
                with self.engine.connect() as conn:
                    conn.execute(text(explain_query))
            except Exception as e:
                violations.append(f"Database query planning (EXPLAIN) failed: {e}")

        is_valid = len(violations) == 0
        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            transformed_sql=transformed_sql if is_valid else None,
        )


def validate_sql(sql: str, schema: DatabaseSchema | None = None, dry_run: bool = True) -> ValidationResult:
    validator = SQLValidator(schema=schema)
    return validator.validate(sql, dry_run=dry_run)

