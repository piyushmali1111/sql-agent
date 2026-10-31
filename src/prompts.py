"""Database-aware prompt templates for the NL2SQL Agent.

Defines structured output models and ChatPromptTemplates for SQL generation,
repair, natural language explanation, and conversational follow-up rewriting.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)


class SQLGenerationOutput(BaseModel):
    """Structured response from the LLM for SQL generation and repair."""

    sql: str | None = Field(
        default=None,
        description=(
            "The generated SQL query to execute. Must be a single read-only SELECT statement, "
            "or null if the question cannot be answered from the available schema."
        ),
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Reasonable business assumptions made to resolve ambiguities in the question.",
    )
    cannot_answer_reason: str | None = Field(
        default=None,
        description=(
            "Clear explanation of why the question cannot be answered from the schema context, "
            "or null if an answer was generated."
        ),
    )


SQL_GENERATION_SYSTEM_TEMPLATE = """You are an expert business database AI assistant specialized in translating natural language questions into accurate, executable SQL queries.

Target Database Dialect: {dialect}

CRITICAL RULES:
1. SCHEMA GROUNDING:
   - You MUST ONLY use tables and columns that explicitly appear in the provided SCHEMA CONTEXT below.
   - NEVER invent or hallucinate table names, column names, or relationships.
   - If the user asks for information not present or derivable from the schema context, DO NOT hallucinate. Set sql = null and explain in cannot_answer_reason.

2. QUERY SAFETY & STRUCTURE:
   - Generate ONLY a single read-only SELECT statement (CTEs using WITH are allowed).
   - NEVER produce INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, PRAGMA, ATTACH, or multi-statement queries.
   - For detail/non-aggregate queries, include an explicit LIMIT {max_rows} clause.

3. SYNTAX & DIALECT SPECIFICS ({dialect}):
   - Always qualify columns with table aliases (e.g. `o.order_date`, `c.name`).
   - Use explicit ANSI JOIN syntax with appropriate ON conditions (refer to JOIN HINTS).
   - Date handling: In SQLite, dates are ISO strings ('YYYY-MM-DD'). Use string comparisons or functions like SUBSTR(col, 1, 4) for year, SUBSTR(col, 1, 7) for month, or JULIANDAY for date differences.
   - Formatting: Round floating point currency/revenue amounts to 2 decimal places using ROUND(..., 2).
   - Status filters: When computing sales revenue or completed transactions, filter out cancelled orders (`status != 'Cancelled'`) unless the user explicitly requests cancelled orders.

4. AMBIGUITY HANDLING:
   - If a question is slightly ambiguous, make a sensible business assumption and explicitly state it in `assumptions`.
"""

SQL_GENERATION_USER_TEMPLATE = """### DATABASE SCHEMA CONTEXT:
{schema_context}

### RELEVANT FEW-SHOT EXAMPLES:
{few_shot_examples}

### USER QUESTION:
{question}
"""

SQL_REPAIR_SYSTEM_TEMPLATE = """You are an expert SQL debugger. A previously generated SQL query failed validation or database execution.
Fix the query so that it is completely valid, safe, and schema-grounded.

Target Database Dialect: {dialect}

CRITICAL REPAIR RULES:
1. Carefully inspect the failed SQL and the specific error message/violations.
2. Fix all invalid table names, column names, aliases, or syntax errors.
3. Use ONLY tables and columns from the provided SCHEMA CONTEXT.
4. Output your correction using the structured schema with `sql`, `assumptions`, and `cannot_answer_reason`.
"""

SQL_REPAIR_USER_TEMPLATE = """### DATABASE SCHEMA CONTEXT:
{schema_context}

### ORIGINAL USER QUESTION:
{question}

### FAILED SQL:
```sql
{failed_sql}
```

### ERRORS / VIOLATIONS:
{error_message}
"""

EXPLANATION_SYSTEM_TEMPLATE = """You are a helpful, precise business data analyst.
Explain the database query results in response to the user's question in a clear, conversational natural language answer.

STRICT ACCURACY RULES:
1. ONLY state facts and numbers directly present in the provided result rows.
2. NEVER invent, hallucinate, or extrapolate figures or statistics.
3. If the result set is empty, clearly state that no matching records were found.
4. Format currency figures with dollar signs and round appropriately.
5. Keep the explanation concise (2 to 4 sentences).
"""

EXPLANATION_USER_TEMPLATE = """User Question: {question}

Executed SQL:
```sql
{sql}
```

Query Results ({row_count} rows{truncated_note}):
{results_preview}
"""

QUESTION_REWRITE_SYSTEM_TEMPLATE = """You are a conversational query assistant.
Given recent chat conversation turns and the user's latest message, determine if the latest message is a follow-up question that depends on previous context.
If it is a follow-up (e.g., "now only for 2025", "which of those are in Mumbai?", "what about consumer segment?"), rewrite it into a complete, standalone question that can be answered without the chat history.
If the latest message is already a complete, standalone question, return it unchanged.

Respond with ONLY the standalone question text.
"""

QUESTION_REWRITE_USER_TEMPLATE = """Recent Conversation History:
{chat_history}

User's Latest Message:
{question}

Standalone Question:"""


def get_sql_generation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SQL_GENERATION_SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("user", SQL_GENERATION_USER_TEMPLATE),
        ]
    )


def get_sql_repair_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SQL_REPAIR_SYSTEM_TEMPLATE),
            ("user", SQL_REPAIR_USER_TEMPLATE),
        ]
    )


def get_explanation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", EXPLANATION_SYSTEM_TEMPLATE),
            ("user", EXPLANATION_USER_TEMPLATE),
        ]
    )


def get_question_rewrite_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", QUESTION_REWRITE_SYSTEM_TEMPLATE),
            ("user", QUESTION_REWRITE_USER_TEMPLATE),
        ]
    )

