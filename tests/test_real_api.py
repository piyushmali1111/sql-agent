"""Integration tests against the live Gemini API.

Runs only when GOOGLE_API_KEY environment variable is provided.
"""

import os
import pytest

from src.agent import NL2SQLAgent
from src.config import settings

skip_no_key = pytest.mark.skipif(
    not bool(settings.google_api_key),
    reason="Live Gemini API key (GOOGLE_API_KEY) not set in environment or .env",
)


@skip_no_key
def test_real_gemini_api_end_to_end():
    agent = NL2SQLAgent()
    result = agent.answer_question("How many total customers are in the database?")

    assert result.status == "ok"
    assert result.sql is not None
    assert "count" in result.sql.lower()
    assert result.row_count == 1
    assert result.rows[0][0] == 300
    assert len(result.answer_text) > 0
    assert result.llm_calls >= 1

