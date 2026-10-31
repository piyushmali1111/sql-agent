"""Schema knowledge base indexing using Chroma.

Builds a searchable semantic index of tables, columns, relationships,
and few-shot examples grounded in the live database schema.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.config import settings
from src.db import DatabaseSchema, create_readonly_engine, get_schema
from src.llm import get_embeddings

logger = logging.getLogger(__name__)


def load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.warning(f"YAML file not found at: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_column_sample_values(engine: Engine, table: str, column: str, limit: int = 5) -> list[str]:
    """Safely retrieves a small sample of distinct non-null values for column context."""
    try:
        # Tables and columns originate from get_schema() inspection (trusted identifiers)
        query = text(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT {limit}")
        with engine.connect() as conn:
            rows = conn.execute(query).fetchall()
            return [str(r[0]) for r in rows if r[0] is not None]
    except Exception as e:
        logger.debug(f"Could not sample values for {table}.{column}: {e}")
        return []


def build_schema_documents(
    schema: DatabaseSchema,
    descriptions: dict[str, Any],
    examples_data: dict[str, Any],
    engine: Engine,
) -> list[Document]:
    """Constructs Table, Column, Relationship, and Example cards for Chroma indexing."""
    docs: list[Document] = []
    tables_meta = descriptions.get("tables", {})

    # 1. Table Cards
    for tbl_name, tbl_info in schema.tables.items():
        tbl_desc_data = tables_meta.get(tbl_name, {})
        tbl_desc = tbl_desc_data.get("description", f"Database table '{tbl_name}'.")

        cols_summary = []
        for col in tbl_info.columns:
            c_desc = tbl_desc_data.get("columns", {}).get(col.name, {}).get("description", "")
            desc_part = f" - {c_desc}" if c_desc else ""
            pk_part = " [PK]" if col.primary_key else ""
            cols_summary.append(f"  * {col.name} ({col.type}){pk_part}{desc_part}")

        fk_summary = []
        for fk in tbl_info.foreign_keys:
            c_cols = ", ".join(fk.constrained_columns)
            r_cols = ", ".join(fk.referred_columns)
            fk_summary.append(f"  * {c_cols} -> {fk.referred_table}({r_cols})")

        content = (
            f"Table: {tbl_name}\n"
            f"Business Description: {tbl_desc}\n"
            f"Columns:\n" + "\n".join(cols_summary) + "\n"
            f"Primary Key: {', '.join(tbl_info.primary_keys) or 'None'}\n"
            f"Foreign Keys:\n" + ("\n".join(fk_summary) if fk_summary else "  * None")
        )

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "type": "table",
                    "table": tbl_name,
                    "column": "",
                },
            )
        )

        # 2. Column Cards
        for col in tbl_info.columns:
            col_desc_info = tbl_desc_data.get("columns", {}).get(col.name, {})
            col_desc = col_desc_info.get("description", f"Column '{col.name}' in table '{tbl_name}'.")
            enums = col_desc_info.get("enums", [])
            unit = col_desc_info.get("unit", "")

            samples = get_column_sample_values(engine, tbl_name, col.name, limit=5)
            sample_str = ", ".join(samples) if samples else "None available"
            enum_str = f"Allowed Values: {', '.join(enums)}" if enums else ""
            unit_str = f"Unit: {unit}" if unit else ""

            col_content = (
                f"Column: {col.name}\n"
                f"Table: {tbl_name}\n"
                f"Type: {col.type}\n"
                f"Primary Key: {'Yes' if col.primary_key else 'No'}\n"
                f"Description: {col_desc}\n"
            )
            if unit_str:
                col_content += f"{unit_str}\n"
            if enum_str:
                col_content += f"{enum_str}\n"
            col_content += f"Sample Distinct Values: [{sample_str}]"

            docs.append(
                Document(
                    page_content=col_content,
                    metadata={
                        "type": "column",
                        "table": tbl_name,
                        "column": col.name,
                    },
                )
            )

        # 3. Relationship Cards
        for fk in tbl_info.foreign_keys:
            c_col = fk.constrained_columns[0] if fk.constrained_columns else ""
            r_col = fk.referred_columns[0] if fk.referred_columns else "id"
            join_hint = f"{tbl_name} JOIN {fk.referred_table} ON {tbl_name}.{c_col} = {fk.referred_table}.{r_col}"

            rel_content = (
                f"Relationship: {tbl_name}.{c_col} -> {fk.referred_table}.{r_col}\n"
                f"Plain Language: Each {tbl_name} row references {fk.referred_table}.\n"
                f"Join Hint: {join_hint}\n"
                f"From Table: {tbl_name}\n"
                f"To Table: {fk.referred_table}"
            )

            docs.append(
                Document(
                    page_content=rel_content,
                    metadata={
                        "type": "relationship",
                        "table": tbl_name,
                        "column": c_col,
                        "referred_table": fk.referred_table,
                    },
                )
            )

    # 4. Example Cards
    examples = examples_data.get("examples", [])
    for ex in examples:
        q = ex.get("question", "")
        sql = ex.get("sql", "").strip()
        tables = ex.get("tables", [])
        technique = ex.get("technique", "")

        ex_content = (
            f"Example Question: {q}\n"
            f"Technique: {technique}\n"
            f"Tables Used: {', '.join(tables)}\n"
            f"SQL:\n{sql}"
        )

        docs.append(
            Document(
                page_content=ex_content,
                metadata={
                    "type": "example",
                    "table": ",".join(tables),
                    "column": "",
                    "question": q,
                },
            )
        )

    return docs


def build_schema_index(
    reset: bool = False,
    persist_dir: Path | None = None,
    embedding_model_name: str | None = None,
) -> Chroma:
    """Builds or refreshes the persistent Chroma schema index."""
    target_dir = persist_dir or settings.chroma_dir
    model_name = embedding_model_name or settings.embedding_model
    meta_path = target_dir / "index_meta.json"

    if reset and target_dir.exists():
        logger.info(f"Resetting existing Chroma index at {target_dir}")
        shutil.rmtree(target_dir, ignore_errors=True)

    target_dir.mkdir(parents=True, exist_ok=True)

    engine = create_readonly_engine()
    schema = get_schema(engine)
    descriptions = load_yaml_file(settings.schema_descriptions_path)
    examples = load_yaml_file(settings.examples_path)

    embeddings = get_embeddings(model_name)

    docs = build_schema_documents(schema, descriptions, examples, engine)
    logger.info(f"Generated {len(docs)} schema documents to index.")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="schema_knowledge_base",
        persist_directory=str(target_dir),
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model": model_name,
                "document_count": len(docs),
                "tables": schema.get_table_names(),
            },
            f,
            indent=2,
        )

    logger.info(f"Schema index built successfully with {len(docs)} documents at {target_dir}")
    return vectorstore


def load_schema_index(
    persist_dir: Path | None = None,
    embedding_model_name: str | None = None,
) -> Chroma:
    """Loads an existing Chroma schema index and validates embedding model consistency."""
    target_dir = persist_dir or settings.chroma_dir
    model_name = embedding_model_name or settings.embedding_model
    meta_path = target_dir / "index_meta.json"

    if not target_dir.exists() or not meta_path.exists():
        logger.info("Chroma index not found. Building initial index...")
        return build_schema_index(reset=False, persist_dir=target_dir, embedding_model_name=model_name)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    indexed_model = meta.get("embedding_model")
    if indexed_model and indexed_model != model_name:
        logger.warning(
            f"Embedding model mismatch! Index was built with '{indexed_model}', "
            f"but current configuration is '{model_name}'. "
            f"Please rebuild index using: python -m src.schema_index --reset"
        )

    embeddings = get_embeddings(model_name)
    return Chroma(
        collection_name="schema_knowledge_base",
        embedding_function=embeddings,
        persist_directory=str(target_dir),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Chroma schema index for NL2SQL agent.")
    parser.add_argument("--reset", action="store_true", help="Clear and rebuild index from scratch.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    vs = build_schema_index(reset=args.reset)
    print("Index build completed successfully.")

