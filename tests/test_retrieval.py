"""Tests for semantic schema retrieval, table closure, and example matching."""

import pytest
from src.retrieval import SchemaRetriever, retrieve_schema_context


@pytest.fixture(scope="module")
def retriever():
    return SchemaRetriever()


def test_retrieval_returns_schema_and_examples(retriever):
    ctx = retriever.retrieve("Which customers have placed the most orders?")
    assert len(ctx.tables) > 0
    assert "customers" in ctx.tables
    assert "orders" in ctx.tables
    assert len(ctx.examples) > 0
    assert "### RELEVANT DATABASE TABLES" in ctx.formatted_context


def test_table_closure_for_column_references(retriever):
    # Query referencing discounts and categories
    ctx = retriever.retrieve("What is the average discount for electronics products?")
    # Check that required tables are included
    assert "products" in ctx.tables or "categories" in ctx.tables
    # Ensure every table card present is non-empty
    assert len(ctx.table_cards) >= len(ctx.tables)


def test_relationship_join_hints_included(retriever):
    ctx = retriever.retrieve("Show customer orders with employee details")
    # Should include orders, customers, and employees join hints
    rel_hints = "\n".join(ctx.relationship_cards)
    assert "orders" in rel_hints.lower()
    assert "customers" in rel_hints.lower() or "employees" in rel_hints.lower()

