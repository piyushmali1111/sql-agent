"""Streamlit Interactive Web Application for the Natural Language to SQL Agent.

Provides conversational SQL analytics, schema inspection, auto-charting,
execution history, and grounded explanations.
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import streamlit as st

from src.agent import AgentResult, NL2SQLAgent
from src.config import settings
from src.conversation import ConversationManager
from src.db import create_readonly_engine, get_schema
from src.schema_index import load_schema_index

# 1. Page Configuration
st.set_page_config(
    page_title="NL to SQL Agent | Conversational Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Cached Resources
@st.cache_resource(show_spinner="Initializing Database Connection...")
def get_cached_database():
    try:
        engine = create_readonly_engine()
        schema = get_schema(engine)
        return engine, schema, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource(show_spinner="Loading Schema Knowledge Base & Agent...")
def get_cached_agent(api_key: str = ""):
    try:
        if api_key:
            settings.google_api_key = api_key
            os.environ["GOOGLE_API_KEY"] = api_key
        agent = NL2SQLAgent()
        return agent, None
    except Exception as e:
        return None, str(e)


# 3. Session State Initialization
if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationManager()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {"calls": 0, "latency_ms": 0.0}

if "sample_question_to_ask" not in st.session_state:
    st.session_state.sample_question_to_ask = None


# Helper to render auto-chart
def render_auto_chart(df: pd.DataFrame) -> None:
    """Renders a bar or line chart if data contains 1 label/date and 1 numeric column."""
    if len(df) <= 1 or df.shape[1] < 2:
        return

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    if len(numeric_cols) >= 1 and len(non_numeric_cols) >= 1:
        x_col = non_numeric_cols[0]
        y_col = numeric_cols[0]

        with st.expander("📊 Data Visualization", expanded=True):
            is_date = any(term in x_col.lower() for term in ["date", "month", "year", "day"])
            chart_data = df.set_index(x_col)[[y_col]]

            if is_date:
                st.line_chart(chart_data)
            else:
                st.bar_chart(chart_data)


# 4. Sidebar Configuration
with st.sidebar:
    st.title("🔍 SQL Agent Settings")
    st.caption("Schema-Grounded Natural Language to SQL Assistant")

    # API Key Handling
    active_key = settings.google_api_key or os.getenv("GOOGLE_API_KEY", "")
    if not active_key:
        user_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google AI Studio API key.",
        )
        if user_key:
            active_key = user_key
            settings.google_api_key = user_key
            os.environ["GOOGLE_API_KEY"] = user_key
            st.rerun()
        else:
            st.warning("⚠️ Enter your Gemini API Key to enable queries.")

    st.markdown("---")
    st.subheader("Database & Schema Status")
    engine, schema, db_err = get_cached_database()

    if db_err:
        st.error(f"Database error: {db_err}")
        st.info("Run `python data/build_db.py` to create the business database.")
    else:
        st.success(f"Connected: `{schema.dialect.upper()}` (Read-Only)")
        with st.expander("View Tables & Columns"):
            for tbl, info in schema.tables.items():
                col_names = ", ".join(c.name for c in info.columns)
                st.markdown(f"**{tbl}** ({len(info.columns)} cols): `{col_names}`")

    st.markdown("---")
    st.subheader("Model & Embeddings")
    st.markdown(f"- **LLM:** `{settings.gemini_model}`")
    st.markdown(f"- **Embedding Model:** `{settings.embedding_model}`")

    st.markdown("---")
    st.subheader("Last Query Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("LLM Calls", st.session_state.last_metrics["calls"])
    with col2:
        st.metric("Latency", f"{st.session_state.last_metrics['latency_ms']:.0f} ms")

    st.markdown("---")
    st.subheader("Example Questions")
    sample_queries = [
        "What is total revenue from completed orders?",
        "Which products have less than 20 items in stock?",
        "Top 5 best selling products by revenue?",
        "Show orders delivered to London or Berlin.",
        "List all customers in California.",
        "Average discount percentage by employee region?",
        "Which products have never been ordered?",
    ]
    for sq in sample_queries:
        if st.button(sq, key=f"btn_{sq}", use_container_width=True):
            st.session_state.sample_question_to_ask = sq
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.conversation.clear()
        st.session_state.chat_messages.clear()
        st.session_state.last_metrics = {"calls": 0, "latency_ms": 0.0}
        st.rerun()

    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #888; margin-top: 20px;">
        🔒 <b>Read-Only Safety:</b> Queries run in strict read-only SQLite URI mode.
        All statements are AST-validated with sqlglot prior to execution.
        </div>
        """,
        unsafe_allow_html=True,
    )


# 5. Main Chat Area
st.title("Natural Language to SQL Assistant")
st.markdown(
    "Ask questions in plain English across sales, customers, inventory, and employees. "
    "Queries are automatically grounded in the database schema and executed safely."
)

# Display historical messages
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sql" in msg and msg["sql"]:
            st.markdown(f"```sql\n{msg['sql']}\n```")
        if "df" in msg and msg["df"] is not None:
            st.dataframe(msg["df"], use_container_width=True)
            if msg.get("truncated"):
                st.caption(f"⚠️ Showing first {len(msg['df'])} rows (truncated).")
            render_auto_chart(msg["df"])
        if "meta" in msg and msg["meta"]:
            with st.expander("🔍 Schema Context & Details", expanded=False):
                meta = msg["meta"]
                st.markdown(f"**Retrieved Tables:** {', '.join(meta.get('retrieved_tables', []))}")
                if meta.get("assumptions"):
                    st.markdown("**Assumptions Made:**")
                    for asm in meta["assumptions"]:
                        st.markdown(f"- {asm}")
                if meta.get("repair_attempts", 0) > 0:
                    st.warning(f"Query was repaired through {meta['repair_attempts']} self-healing attempt(s).")

# Check for input from sample question button or chat input
user_prompt = st.chat_input("Ask a question about the business data...")
if st.session_state.sample_question_to_ask:
    user_prompt = st.session_state.sample_question_to_ask
    st.session_state.sample_question_to_ask = None

if user_prompt:
    # 1. Render User Message
    with st.chat_message("user"):
        st.markdown(user_prompt)

    st.session_state.chat_messages.append({"role": "user", "content": user_prompt})

    # 2. Process Agent Execution
    with st.chat_message("assistant"):
        with st.spinner("Analyzing schema and generating SQL query..."):
            agent, agent_err = get_cached_agent(api_key=active_key)

            if agent_err or not agent:
                st.error(f"Agent initialization error: {agent_err}")
                st.info("Ensure `GOOGLE_API_KEY` is provided in the sidebar or `.env`.")
            else:
                try:
                    result: AgentResult = agent.answer_question(
                        user_prompt,
                        conversation_manager=st.session_state.conversation,
                    )

                    # Update sidebar metrics
                    st.session_state.last_metrics = {
                        "calls": result.llm_calls,
                        "latency_ms": result.latency_ms,
                    }

                    # Render text explanation
                    st.markdown(result.answer_text)

                    df_result = None
                    if result.status == "ok" and result.sql:
                        # Render SQL block
                        st.markdown(f"```sql\n{result.sql}\n```")

                        if result.columns and result.rows:
                            df_result = pd.DataFrame(result.rows, columns=result.columns)
                            st.dataframe(df_result, use_container_width=True)
                            if result.truncated:
                                st.caption(f"⚠️ Showing first {result.row_count} rows (results truncated).")
                            render_auto_chart(df_result)
                        elif result.status == "ok":
                            st.info("Query returned 0 rows matching the criteria.")

                    # Render Expander
                    meta_info = {
                        "retrieved_tables": result.retrieved_tables,
                        "assumptions": result.assumptions,
                        "repair_attempts": result.repair_attempts,
                    }

                    with st.expander("🔍 Schema Context & Details", expanded=False):
                        st.markdown(f"**Retrieved Tables:** {', '.join(result.retrieved_tables)}")
                        if result.assumptions:
                            st.markdown("**Assumptions Made:**")
                            for asm in result.assumptions:
                                st.markdown(f"- {asm}")
                        if result.repair_attempts > 0:
                            st.warning(f"Query was repaired through {result.repair_attempts} self-healing attempt(s).")
                        if result.violations_history:
                            st.markdown("**Violations Resolved:**")
                            for attempt_idx, viols in enumerate(result.violations_history, 1):
                                st.markdown(f"- *Attempt {attempt_idx}:* {'; '.join(viols)}")

                    # Store in chat history
                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content": result.answer_text,
                            "sql": result.sql,
                            "df": df_result,
                            "truncated": result.truncated,
                            "meta": meta_info,
                        }
                    )

                except Exception as e:
                    err_msg = f"An error occurred while answering your question: {e}"
                    st.error(err_msg)
                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content": err_msg,
                            "sql": None,
                            "df": None,
                            "meta": {},
                        }
                    )

