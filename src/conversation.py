"""Conversation history management and conversational follow-up rewriting.

Maintains recent conversation turns and rewrites follow-up questions into
grounded, standalone queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.config import settings
from src.llm import get_llm, invoke_with_retry
from src.prompts import get_question_rewrite_prompt

logger = logging.getLogger(__name__)


@dataclass
class ChatTurn:
    role: str  # "user" or "assistant"
    content: str
    sql: str | None = None
    data_preview: str | None = None


class ConversationManager:
    """Manages session conversation history and follow-up question resolution."""

    def __init__(self, max_turns: int | None = None):
        self.max_turns = max_turns or settings.conversation_history_turns
        self.turns: list[ChatTurn] = []
        self._rewrite_cache: dict[str, str] = {}

    def add_user_message(self, content: str) -> None:
        self.turns.append(ChatTurn(role="user", content=content))
        self._trim()

    def add_assistant_message(
        self,
        content: str,
        sql: str | None = None,
        data_preview: str | None = None,
    ) -> None:
        self.turns.append(
            ChatTurn(
                role="assistant",
                content=content,
                sql=sql,
                data_preview=data_preview,
            )
        )
        self._trim()

    def clear(self) -> None:
        self.turns.clear()
        self._rewrite_cache.clear()

    def _trim(self) -> None:
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-(self.max_turns * 2) :]

    def format_history_text(self) -> str:
        """Formats recent chat turns as plain text for the rewrite prompt."""
        if not self.turns:
            return "No previous conversation history."
        lines = []
        for turn in self.turns[- (self.max_turns * 2) :]:
            role_label = "User" if turn.role == "user" else "Assistant"
            line = f"{role_label}: {turn.content}"
            if turn.sql:
                line += f" [SQL executed: {turn.sql}]"
            lines.append(line)
        return "\n".join(lines)

    def to_langchain_messages(self) -> list[BaseMessage]:
        """Converts stored turns to LangChain BaseMessage objects."""
        messages: list[BaseMessage] = []
        for turn in self.turns[- (self.max_turns * 2) :]:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            else:
                messages.append(AIMessage(content=turn.content))
        return messages

    def is_likely_followup(self, question: str) -> bool:
        """Lightweight heuristic to detect follow-up phrasing before calling LLM."""
        if not self.turns:
            return False

        q_lower = question.lower().strip()

        # Follow-up trigger keywords and pronouns
        followup_starts = (
            "now ",
            "also ",
            "what about",
            "how about",
            "which of those",
            "which of these",
            "filter by",
            "only for",
            "and in",
            "and for",
            "same for",
            "instead",
            "sort by",
            "order by",
            "limit to",
        )
        if any(q_lower.startswith(start) for start in followup_starts):
            return True

        pronouns = ("those", "these", "them", "their", "it", "that", "above")
        tokens = q_lower.split()
        if any(p in tokens for p in pronouns):
            return True

        # Very short queries (e.g. "in 2025?", "in Mumbai?")
        if len(tokens) <= 4:
            return True

        return False

    def get_standalone_question(self, question: str, llm: Any = None) -> str:
        """Resolves conversational references into a standalone question."""
        if not self.turns or not self.is_likely_followup(question):
            return question

        # Check rewrite cache
        cache_key = f"{self.format_history_text()}||{question}"
        if cache_key in self._rewrite_cache:
            return self._rewrite_cache[cache_key]

        if llm is None:
            try:
                llm = get_llm(temperature=0.0)
            except Exception as e:
                logger.warning(f"Could not load LLM for question rewrite: {e}")
                return question

        prompt = get_question_rewrite_prompt()
        history_text = self.format_history_text()

        try:
            chain = prompt | llm
            response = invoke_with_retry(
                chain.invoke,
                {"chat_history": history_text, "question": question},
                step_name="question_rewrite",
            )
            standalone = response.content.strip().strip('"') if hasattr(response, "content") else str(response).strip()
            self._rewrite_cache[cache_key] = standalone
            logger.info(f"Rewrote follow-up question '{question}' -> '{standalone}'")
            return standalone
        except Exception as e:
            logger.warning(f"Failed to rewrite follow-up question: {e}")
            return question

