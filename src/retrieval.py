"""Semantic retrieval over schema context and curated examples.

Provides table closure, relationship join-path discovery, and formatted
prompt context for grounded SQL generation.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.db import DatabaseSchema, create_readonly_engine, get_schema
from src.schema_index import load_schema_index

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    question: str
    tables: list[str]
    table_cards: list[str] = field(default_factory=list)
    column_cards: list[str] = field(default_factory=list)
    relationship_cards: list[str] = field(default_factory=list)
    examples: list[dict[str, str]] = field(default_factory=list)
    formatted_context: str = ""


class SchemaRetriever:
    """Retrieves relevant schema elements and examples with table closure."""

    def __init__(
        self,
        vectorstore: Chroma | None = None,
        schema: DatabaseSchema | None = None,
    ):
        self.vectorstore = vectorstore or load_schema_index()
        self.schema = schema or get_schema()
        self._all_table_names = set(self.schema.get_table_names())
        self._table_cards_cache: dict[str, str] = {}
        self._load_all_table_cards()

    def _load_all_table_cards(self) -> None:
        """Preloads table card text for all tables in the live schema."""
        try:
            results = self.vectorstore.get(where={"type": "table"})
            metadatas = results.get("metadatas", [])
            documents = results.get("documents", [])
            for meta, doc in zip(metadatas, documents):
                tbl = meta.get("table", "")
                if tbl:
                    self._table_cards_cache[tbl] = doc
        except Exception as e:
            logger.warning(f"Failed to preload table cards: {e}")

    def retrieve(
        self,
        question: str,
        table_k: int | None = None,
        column_k: int | None = None,
        example_k: int | None = None,
    ) -> RetrievedContext:
        """Retrieves schema cards and examples for a question, enforcing table closure."""
        t_k = table_k or settings.table_retrieval_k
        c_k = column_k or settings.column_retrieval_k
        ex_k = example_k or settings.few_shot_k

        # Retrieve relevant schema items (tables, columns, relationships)
        schema_results = self.vectorstore.similarity_search(
            question,
            k=t_k + c_k + 6,
            filter={"type": {"$in": ["table", "column", "relationship"]}},
        )

        # Retrieve relevant few-shot examples
        example_results = self.vectorstore.similarity_search(
            question,
            k=ex_k,
            filter={"type": "example"},
        )

        selected_tables: set[str] = set()
        table_cards_dict: dict[str, str] = {}
        column_cards: list[str] = []
        rel_cards_dict: dict[str, str] = {}

        # 1. Inspect retrieved schema docs
        for doc in schema_results:
            doc_type = doc.metadata.get("type")
            tbl = doc.metadata.get("table", "")

            if doc_type == "table" and tbl in self._all_table_names:
                selected_tables.add(tbl)
                table_cards_dict[tbl] = doc.page_content
            elif doc_type == "column":
                if tbl in self._all_table_names:
                    selected_tables.add(tbl)
                column_cards.append(doc.page_content)
            elif doc_type == "relationship":
                if tbl in self._all_table_names:
                    selected_tables.add(tbl)
                ref_tbl = doc.metadata.get("referred_table", "")
                if ref_tbl in self._all_table_names:
                    selected_tables.add(ref_tbl)
                rel_cards_dict[doc.page_content] = doc.page_content

        # 2. Enforce Table Closure:
        # Every table required by retrieved columns/relationships MUST have its full table card included
        for tbl in list(selected_tables):
            if tbl not in table_cards_dict:
                cached_card = self._table_cards_cache.get(tbl)
                if cached_card:
                    table_cards_dict[tbl] = cached_card

        # 3. Join Path Inclusion:
        # For all pairs of selected tables, include relationship cards that join them
        for tbl in selected_tables:
            tbl_info = self.schema.tables.get(tbl)
            if not tbl_info:
                continue
            for fk in tbl_info.foreign_keys:
                if fk.referred_table in selected_tables:
                    # Relationship joins two selected tables
                    c_col = fk.constrained_columns[0] if fk.constrained_columns else ""
                    r_col = fk.referred_columns[0] if fk.referred_columns else "id"
                    join_str = f"Relationship: {tbl}.{c_col} -> {fk.referred_table}.{r_col} (Join: {tbl} JOIN {fk.referred_table} ON {tbl}.{c_col} = {fk.referred_table}.{r_col})"
                    rel_cards_dict[join_str] = join_str

        # Format examples
        examples_list: list[dict[str, str]] = []
        for doc in example_results:
            examples_list.append(
                {
                    "content": doc.page_content,
                    "question": doc.metadata.get("question", ""),
                }
            )

        # Build formatted schema context string
        context_parts = []
        context_parts.append("### RELEVANT DATABASE TABLES & COLUMNS:")
        for tbl_name in sorted(list(selected_tables)):
            if tbl_name in table_cards_dict:
                context_parts.append(table_cards_dict[tbl_name])
                context_parts.append("-" * 40)

        if rel_cards_dict:
            context_parts.append("\n### JOIN RELATIONSHIPS & HINTS:")
            for rel in rel_cards_dict.values():
                context_parts.append(f"- {rel}")

        if column_cards:
            context_parts.append("\n### RELEVANT COLUMN DETAILS & SAMPLE VALUES:")
            for col_card in column_cards[:6]:
                context_parts.append(col_card)
                context_parts.append("")

        formatted_context = "\n".join(context_parts)

        return RetrievedContext(
            question=question,
            tables=sorted(list(selected_tables)),
            table_cards=list(table_cards_dict.values()),
            column_cards=column_cards,
            relationship_cards=list(rel_cards_dict.values()),
            examples=examples_list,
            formatted_context=formatted_context,
        )


def retrieve_schema_context(question: str) -> RetrievedContext:
    retriever = SchemaRetriever()
    return retriever.retrieve(question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test schema context retrieval for a question.")
    parser.add_argument("question", type="str", help="Natural language question to test.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    ctx = retrieve_schema_context(args.question)

    print("=" * 60)
    print(f"QUESTION: {ctx.question}")
    print(f"RETRIEVED TABLES ({len(ctx.tables)}): {ctx.tables}")
    print(f"EXAMPLES RETRIEVED: {len(ctx.examples)}")
    print("=" * 60)
    print("FORMATTED SCHEMA CONTEXT:")
    print(ctx.formatted_context)
    print("=" * 60)
    print("FEW-SHOT EXAMPLES:")
    for i, ex in enumerate(ctx.examples, 1):
        print(f"\n--- Example {i} ---")
        print(ex["content"])

