"""Unit tests for the NL2SQL Agent pipeline with mocked LLM."""

from unittest.mock import MagicMock
import pytest
from langchain_core.runnables import RunnableLambda

from src.agent import NL2SQLAgent
from src.conversation import ConversationManager
from src.prompts import SQLGenerationOutput


def test_agent_successful_generation():
    mock_llm = MagicMock()
    output_obj = SQLGenerationOutput(
        sql="SELECT id, name FROM customers WHERE city = 'New York'",
        assumptions=["Filtering for active customers in New York"],
    )
    mock_llm.with_structured_output.return_value = RunnableLambda(lambda x: output_obj)
    mock_llm.__or__ = MagicMock(return_value=RunnableLambda(lambda x: MagicMock(content="Here are the customers in New York.")))

    agent = NL2SQLAgent(llm=mock_llm)
    result = agent.answer_question("Show customers in New York")

    assert result.status == "ok"
    assert result.sql is not None
    assert "customers" in result.sql.lower()
    assert result.row_count > 0
    assert result.repair_attempts == 0


def test_agent_validation_triggers_repair():
    mock_llm = MagicMock()
    # First invocation returns invalid SQL (unknown table 'clients')
    # Second invocation (repair) returns valid SQL (table 'customers')
    outputs = [
        SQLGenerationOutput(
            sql="SELECT id, name FROM clients",  # Invalid table
            assumptions=[],
        ),
        SQLGenerationOutput(
            sql="SELECT id, name FROM customers",  # Valid table
            assumptions=["Corrected table to customers"],
        ),
    ]
    call_idx = 0

    def mock_invoke(x):
        nonlocal call_idx
        idx = min(call_idx, len(outputs) - 1)
        call_idx += 1
        return outputs[idx]

    mock_llm.with_structured_output.return_value = RunnableLambda(mock_invoke)
    mock_llm.__or__ = MagicMock(return_value=RunnableLambda(lambda x: MagicMock(content="Returned customers successfully.")))

    agent = NL2SQLAgent(llm=mock_llm)
    result = agent.answer_question("List customers")

    assert result.status == "ok"
    assert result.repair_attempts == 1
    assert "customers" in result.sql.lower()
    assert len(result.violations_history) >= 1


def test_agent_repair_stops_at_max_limit_never_executing_invalid():
    mock_llm = MagicMock()
    # Always returns invalid SQL with forbidden DROP
    mock_llm.with_structured_output.return_value = RunnableLambda(
        lambda x: SQLGenerationOutput(
            sql="DROP TABLE customers;",
            assumptions=[],
        )
    )

    agent = NL2SQLAgent(llm=mock_llm)
    result = agent.answer_question("Drop the customer table")

    assert result.status == "failed"
    assert result.repair_attempts == 2
    assert "failed validation" in result.error.lower()
    assert result.row_count == 0


def test_agent_cannot_answer_structured_output():
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = RunnableLambda(
        lambda x: SQLGenerationOutput(
            sql=None,
            cannot_answer_reason="The database does not contain satellite GPS data.",
            assumptions=[],
        )
    )

    agent = NL2SQLAgent(llm=mock_llm)
    result = agent.answer_question("What is the GPS coordinate of each supplier warehouse?")

    assert result.status == "cannot_answer"
    assert "satellite GPS" in result.answer_text
    assert result.sql is None
    assert result.row_count == 0


def test_conversation_followup_detection():
    cm = ConversationManager()
    assert not cm.is_likely_followup("Show all products")

    cm.add_user_message("Show products in Audio category")
    cm.add_assistant_message("Here are the 10 audio products.")

    assert cm.is_likely_followup("now only for 2025")
    assert cm.is_likely_followup("which of those are under $50?")
    assert cm.is_likely_followup("in Mumbai?")
