"""NL2SQL Agent pipeline: generate, validate, execute, and repair.

Coordinates semantic schema retrieval, structured SQL generation with Gemini,
sqlglot AST validation, read-only DB execution, and natural language explanations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.config import settings
from src.conversation import ConversationManager
from src.db import DatabaseError, QueryExecutionError, QueryResult, create_readonly_engine, execute_select, get_schema
from src.llm import (
    LLMCallMetrics,
    get_llm,
    get_llm_metrics,
    invoke_with_retry,
)
from src.prompts import (
    SQLGenerationOutput,
    get_explanation_prompt,
    get_sql_generation_prompt,
    get_sql_repair_prompt,
)
from src.retrieval import RetrievedContext, SchemaRetriever, retrieve_schema_context
from src.utils import format_results_preview
from src.validator import SQLValidator, ValidationResult, validate_sql

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    question: str
    standalone_question: str
    sql: str | None = None
    assumptions: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    answer_text: str = ""
    retrieved_tables: list[str] = field(default_factory=list)
    repair_attempts: int = 0
    violations_history: list[list[str]] = field(default_factory=list)
    llm_calls: int = 0
    latency_ms: float = 0.0
    status: str = "ok"  # "ok", "cannot_answer", "failed"
    error: str | None = None


class NL2SQLAgent:
    """The complete Natural Language to SQL generation and execution agent."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        retriever: SchemaRetriever | None = None,
        validator: SQLValidator | None = None,
    ):
        self._llm = llm
        self.retriever = retriever or SchemaRetriever()
        self.validator = validator or SQLValidator()
        self.engine = self.validator.engine
        self.schema = self.validator.schema

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = get_llm(temperature=0.0)
        return self._llm

    def _generate_structured_sql(
        self,
        question: str,
        retrieved: RetrievedContext,
        chat_history_messages: list[BaseMessage] | None = None,
        metrics: LLMCallMetrics | None = None,
    ) -> SQLGenerationOutput:
        """Invokes Gemini with structured output to generate grounded SQL."""
        prompt = get_sql_generation_prompt()
        structured_llm = self.llm.with_structured_output(SQLGenerationOutput)
        chain = prompt | structured_llm

        examples_text = "\n\n".join(ex["content"] for ex in retrieved.examples)
        if not examples_text:
            examples_text = "No few-shot examples provided."

        inputs = {
            "dialect": self.schema.dialect.upper(),
            "max_rows": settings.max_execution_rows,
            "schema_context": retrieved.formatted_context,
            "few_shot_examples": examples_text,
            "question": question,
            "chat_history": chat_history_messages or [],
        }

        try:
            output = invoke_with_retry(
                chain.invoke,
                inputs,
                step_name="sql_generation",
                metrics=metrics,
            )
            if isinstance(output, SQLGenerationOutput):
                return output
            if isinstance(output, dict):
                return SQLGenerationOutput(**output)
            return SQLGenerationOutput(sql=str(output))
        except Exception as e:
            logger.warning(f"Structured output failed on generation: {e}. Retrying once with raw prompt...")
            # Fallback retry
            raw_chain = prompt | self.llm
            raw_resp = invoke_with_retry(
                raw_chain.invoke,
                inputs,
                step_name="sql_generation_fallback",
                metrics=metrics,
            )
            content = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)
            # Extract SQL between ```sql ... ```
            clean_sql = content
            if "```sql" in content:
                clean_sql = content.split("```sql")[1].split("```")[0].strip()
            elif "```" in content:
                clean_sql = content.split("```")[1].split("```")[0].strip()
            return SQLGenerationOutput(sql=clean_sql, assumptions=["Parsed from raw output fallback"])

    def _repair_sql(
        self,
        question: str,
        failed_sql: str,
        errors: list[str],
        retrieved: RetrievedContext,
        metrics: LLMCallMetrics | None = None,
    ) -> SQLGenerationOutput:
        """Attempts to fix failed SQL using the repair prompt."""
        prompt = get_sql_repair_prompt()
        structured_llm = self.llm.with_structured_output(SQLGenerationOutput)
        chain = prompt | structured_llm

        inputs = {
            "dialect": self.schema.dialect.upper(),
            "schema_context": retrieved.formatted_context,
            "question": question,
            "failed_sql": failed_sql,
            "error_message": "\n".join(f"- {err}" for err in errors),
        }

        try:
            output = invoke_with_retry(
                chain.invoke,
                inputs,
                step_name="sql_repair",
                metrics=metrics,
            )
            if isinstance(output, SQLGenerationOutput):
                return output
            if isinstance(output, dict):
                return SQLGenerationOutput(**output)
            return SQLGenerationOutput(sql=str(output))
        except Exception as e:
            logger.warning(f"Repair invocation failed: {e}")
            return SQLGenerationOutput(cannot_answer_reason=f"Failed to repair query: {e}")

    def _generate_explanation(
        self,
        question: str,
        sql: str,
        result: QueryResult,
        metrics: LLMCallMetrics | None = None,
    ) -> str:
        """Produces a short, strictly factual natural language explanation of the results."""
        prompt = get_explanation_prompt()
        chain = prompt | self.llm

        preview = format_results_preview(result.columns, result.rows, max_rows=8)
        trunc_note = " (truncated to preview limit)" if result.truncated else ""

        inputs = {
            "question": question,
            "sql": sql,
            "row_count": result.row_count,
            "truncated_note": trunc_note,
            "results_preview": preview,
        }

        try:
            resp = invoke_with_retry(
                chain.invoke,
                inputs,
                step_name="nl_explanation",
                metrics=metrics,
            )
            return resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
        except Exception as e:
            logger.warning(f"Failed to generate NL explanation: {e}")
            return f"Query returned {result.row_count} row(s) successfully."

    def answer_question(
        self,
        question: str,
        conversation_manager: ConversationManager | None = None,
    ) -> AgentResult:
        """Executes the full NL2SQL pipeline from natural language to validated SQL and answer."""
        start_time = time.perf_counter()
        session_metrics = LLMCallMetrics()

        # Step 1: Follow-up question rewrite if chat history exists
        standalone_question = question
        chat_messages = []
        if conversation_manager and conversation_manager.turns:
            try:
                standalone_question = conversation_manager.get_standalone_question(question, llm=self.llm)
                chat_messages = conversation_manager.to_langchain_messages()
            except Exception as e:
                logger.warning(f"Follow-up resolution failed, using original: {e}")
                standalone_question = question

        # Step 2: Retrieve schema context and few-shot examples
        retrieved = self.retriever.retrieve(standalone_question)

        # Step 3: Generate SQL
        gen_output = self._generate_structured_sql(
            standalone_question,
            retrieved,
            chat_history_messages=chat_messages,
            metrics=session_metrics,
        )

        if gen_output.cannot_answer_reason and not gen_output.sql:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return AgentResult(
                question=question,
                standalone_question=standalone_question,
                assumptions=gen_output.assumptions,
                answer_text=gen_output.cannot_answer_reason,
                retrieved_tables=retrieved.tables,
                llm_calls=session_metrics.total_calls,
                latency_ms=round(duration_ms, 2),
                status="cannot_answer",
            )

        candidate_sql = gen_output.sql or ""
        assumptions = list(gen_output.assumptions)
        repair_attempts = 0
        violations_history: list[list[str]] = []

        # Step 4: Validate and Repair Loop
        validation = self.validator.validate(candidate_sql, dry_run=True, enforce_limit=True)
        if not validation.is_valid:
            violations_history.append(validation.violations)

        while not validation.is_valid and repair_attempts < settings.max_repair_attempts:
            repair_attempts += 1
            logger.info(
                f"Validation violations encountered (attempt {repair_attempts}/{settings.max_repair_attempts}): "
                f"{validation.violations}"
            )
            repair_output = self._repair_sql(
                standalone_question,
                candidate_sql,
                validation.violations,
                retrieved,
                metrics=session_metrics,
            )

            if repair_output.cannot_answer_reason and not repair_output.sql:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                return AgentResult(
                    question=question,
                    standalone_question=standalone_question,
                    sql=candidate_sql,
                    assumptions=assumptions,
                    answer_text=repair_output.cannot_answer_reason,
                    retrieved_tables=retrieved.tables,
                    repair_attempts=repair_attempts,
                    violations_history=violations_history,
                    llm_calls=session_metrics.total_calls,
                    latency_ms=round(duration_ms, 2),
                    status="cannot_answer",
                )

            candidate_sql = repair_output.sql or candidate_sql
            if repair_output.assumptions:
                assumptions.extend(repair_output.assumptions)

            validation = self.validator.validate(candidate_sql, dry_run=True, enforce_limit=True)
            if not validation.is_valid:
                violations_history.append(validation.violations)

        if not validation.is_valid:
            # Reached max repair attempts without passing validation. Never execute invalid SQL!
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            error_msg = f"Generated query failed validation after {repair_attempts} repair attempts: {'; '.join(validation.violations)}"
            return AgentResult(
                question=question,
                standalone_question=standalone_question,
                sql=candidate_sql,
                assumptions=assumptions,
                answer_text=error_msg,
                retrieved_tables=retrieved.tables,
                repair_attempts=repair_attempts,
                violations_history=violations_history,
                llm_calls=session_metrics.total_calls,
                latency_ms=round(duration_ms, 2),
                status="failed",
                error=error_msg,
            )

        # Step 5: Safe Execution
        executable_sql = validation.transformed_sql or candidate_sql
        try:
            exec_result = execute_select(executable_sql, engine=self.engine)
        except (QueryExecutionError, DatabaseError) as db_err:
            logger.warning(f"Database execution error: {db_err}")
            # Database execution error: attempt one repair with DB error if budget remains
            if repair_attempts < settings.max_repair_attempts:
                repair_attempts += 1
                violations_history.append([str(db_err)])
                repair_output = self._repair_sql(
                    standalone_question,
                    executable_sql,
                    [str(db_err)],
                    retrieved,
                    metrics=session_metrics,
                )
                if repair_output.sql:
                    candidate_sql = repair_output.sql
                    validation = self.validator.validate(candidate_sql, dry_run=True, enforce_limit=True)
                    if validation.is_valid:
                        executable_sql = validation.transformed_sql or candidate_sql
                        try:
                            exec_result = execute_select(executable_sql, engine=self.engine)
                        except Exception as e:
                            duration_ms = (time.perf_counter() - start_time) * 1000.0
                            return AgentResult(
                                question=question,
                                standalone_question=standalone_question,
                                sql=executable_sql,
                                assumptions=assumptions,
                                answer_text=f"Database execution failed: {e}",
                                retrieved_tables=retrieved.tables,
                                repair_attempts=repair_attempts,
                                violations_history=violations_history,
                                llm_calls=session_metrics.total_calls,
                                latency_ms=round(duration_ms, 2),
                                status="failed",
                                error=str(e),
                            )
                    else:
                        duration_ms = (time.perf_counter() - start_time) * 1000.0
                        return AgentResult(
                            question=question,
                            standalone_question=standalone_question,
                            sql=candidate_sql,
                            assumptions=assumptions,
                            answer_text=f"Repaired SQL failed validation: {'; '.join(validation.violations)}",
                            retrieved_tables=retrieved.tables,
                            repair_attempts=repair_attempts,
                            violations_history=violations_history,
                            llm_calls=session_metrics.total_calls,
                            latency_ms=round(duration_ms, 2),
                            status="failed",
                            error=str(validation.violations),
                        )
                else:
                    duration_ms = (time.perf_counter() - start_time) * 1000.0
                    return AgentResult(
                        question=question,
                        standalone_question=standalone_question,
                        sql=executable_sql,
                        assumptions=assumptions,
                        answer_text=f"Database execution failed: {db_err}",
                        retrieved_tables=retrieved.tables,
                        repair_attempts=repair_attempts,
                        violations_history=violations_history,
                        llm_calls=session_metrics.total_calls,
                        latency_ms=round(duration_ms, 2),
                        status="failed",
                        error=str(db_err),
                    )
            else:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                return AgentResult(
                    question=question,
                    standalone_question=standalone_question,
                    sql=executable_sql,
                    assumptions=assumptions,
                    answer_text=f"Database execution failed: {db_err}",
                    retrieved_tables=retrieved.tables,
                    repair_attempts=repair_attempts,
                    violations_history=violations_history,
                    llm_calls=session_metrics.total_calls,
                    latency_ms=round(duration_ms, 2),
                    status="failed",
                    error=str(db_err),
                )

        # Step 6: Generate Natural Language Explanation (if enabled)
        if settings.enable_nl_explanation:
            answer_text = self._generate_explanation(
                standalone_question,
                executable_sql,
                exec_result,
                metrics=session_metrics,
            )
        else:
            answer_text = f"Query executed successfully, returning {exec_result.row_count} row(s)."

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 7: Record turn if conversation manager provided
        if conversation_manager:
            conversation_manager.add_user_message(question)
            preview = format_results_preview(exec_result.columns, exec_result.rows, max_rows=3)
            conversation_manager.add_assistant_message(
                answer_text,
                sql=executable_sql,
                data_preview=preview,
            )

        return AgentResult(
            question=question,
            standalone_question=standalone_question,
            sql=executable_sql,
            assumptions=assumptions,
            columns=exec_result.columns,
            rows=exec_result.rows,
            row_count=exec_result.row_count,
            truncated=exec_result.truncated,
            answer_text=answer_text,
            retrieved_tables=retrieved.tables,
            repair_attempts=repair_attempts,
            violations_history=violations_history,
            llm_calls=session_metrics.total_calls,
            latency_ms=round(duration_ms, 2),
            status="ok",
        )

