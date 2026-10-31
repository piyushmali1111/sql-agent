# Natural Language to SQL Agent: Agentic IDE Prompt

**How to use:** Create an empty repo folder, open it in your agentic IDE (Cursor, Claude Code, Antigravity, etc.), then paste everything under the line `--- PROMPT STARTS HERE ---` into the IDE chat. If the IDE stops midway, reply "continue from Phase N".

**Before you start:**
1. Create a Gemini API key in Google AI Studio.
2. (Optional) Create a LangSmith API key for tracing.
3. No dataset download is needed: the prompt builds a synthetic business database for you.

--- PROMPT STARTS HERE ---

You are a senior AI engineer. Build the complete project described below, end to end, in this repository. Work in the phases listed, run and verify each phase before starting the next, and keep the code clean, typed, and documented. Do not invent results: any metric you report must come from actually running the code.

# 1. PROJECT

**"Natural Language to SQL Agent"**: an AI assistant that converts natural-language questions into executable, schema-grounded SQL queries over a business database, runs them safely, and explains the results conversationally.

Required capabilities (each maps 1:1 to a claim on my resume, so every one must genuinely work):

1. An LLM-powered SQL assistant built with **LangChain**, using **semantic retrieval over schema context** (tables, columns, relationships, sample values, example queries) so generated SQL stays grounded in the real schema rather than hallucinated.
2. **Custom, database-aware prompt templates** that give reliable SQL generation across varied natural-language phrasings of the same question.
3. **Conversational analytics on structured business data**: a non-technical user asks questions in plain English (including follow-ups) and gets answers without writing SQL.

Important: this project uses LangChain only. Do not add LangGraph. Implement the generate, validate, execute, repair loop as a clear, small, controlled pipeline using LangChain components (prompt templates, runnables, output parsers, retrievers).

# 2. BEFORE WRITING CODE

- Check the CURRENT official documentation for langchain, langchain-google-genai, langchain-chroma, sqlglot, SQLAlchemy, sentence-transformers, langsmith, and streamlit, and use their current APIs (they change often). Do not rely on memory for import paths or parameter names.
- Pin exact versions in `requirements.txt` after confirming they install together. Target Python 3.11+.
- Before choosing the default Gemini model, check the current Gemini models and rate-limit documentation and choose a stable (non-preview) Flash-class model available on the free tier. Record the model name and date checked in the README.

# 3. TECH STACK

| Layer | Choice |
|---|---|
| LLM | **Gemini API** via `langchain-google-genai` (`ChatGoogleGenerativeAI`) |
| Framework | LangChain |
| Schema knowledge base | Chroma (persistent, local) |
| Embeddings | Local HuggingFace sentence-transformers, default `BAAI/bge-small-en-v1.5`, configurable |
| Database | SQLite by default (via SQLAlchemy), configurable to PostgreSQL/MySQL through a `DATABASE_URL` |
| SQL parsing and validation | `sqlglot` plus checks against the real schema |
| Tracing | LangSmith via environment variables (optional) |
| UI | Streamlit |
| Tests | pytest |

## 3.1 LLM (Gemini only)

- Gemini is the only LLM provider. Keep `get_llm()` in `src/llm.py` as the single place where the model is created.
- Read `GOOGLE_API_KEY` and `GEMINI_MODEL` from `.env` (python-dotenv). Never hardcode a model name or key. In `.env.example`, set `GEMINI_MODEL` to a placeholder with a comment: "pick a current free-tier Flash model from the Gemini models page".
- Temperature 0, sensible max output tokens, and a request timeout.
- Use structured output (Pydantic) where the LLM returns labels or JSON. If parsing fails, retry once, then fall back to a safe default and log a warning.

## 3.2 Free-tier rate-limit handling

- Wrap every LLM call with retries: exponential backoff with jitter on 429/quota/transient errors, max 4 attempts, then raise a clear user-friendly error.
- Add a small central rate limiter (requests per minute from config, conservative default). The eval scripts must throttle automatically.
- Keep LLM calls per question low: at most generate (1) + repair (0 to 2) + explain (1) + an optional follow-up rewrite (1). Make the explanation step switchable (`ENABLE_NL_EXPLANATION`).
- Cache follow-up rewrites and identical question results within a session.
- Log token usage, number of LLM calls, and per-step latency. Show LLM calls for the last question in the Streamlit sidebar.
- Handle blocked or empty Gemini responses gracefully (detect, show a friendly message, log the reason). Do not disable safety filters globally.

# 4. DATABASE

Create a reproducible, synthetic **business database** so no external dataset is needed.

- Script `data/build_db.py` generates a SQLite file `data/business.db` using a fixed random seed. Use realistic, small-business e-commerce/sales data with these tables (adjust column details as sensible):
  - `customers` (id, name, email, city, state, country, signup_date, segment)
  - `products` (id, name, category_id, unit_price, cost, stock_quantity, supplier_id)
  - `categories` (id, name)
  - `suppliers` (id, name, country)
  - `orders` (id, customer_id, employee_id, order_date, ship_date, status, shipping_city)
  - `order_items` (id, order_id, product_id, quantity, unit_price, discount)
  - `employees` (id, name, role, hire_date, manager_id, region)
  - `payments` (id, order_id, payment_date, method, amount)
- Declare primary keys and foreign keys properly. Generate roughly 300 customers, 80 products, 3000 orders, 9000 order items, spanning at least 24 months of dates, so time-based and aggregate questions are meaningful. Include some NULLs, ties, and edge cases (customers with no orders, products never sold, cancelled orders).
- The database file is generated, not committed (gitignore it) but the build script is.
- Keep the app generic: all schema knowledge must be derived at runtime from the database through SQLAlchemy inspection plus a hand-written `data/schema_descriptions.yaml` (business-friendly descriptions of each table and column, units, enumerations such as valid `status` values). The code must work if `DATABASE_URL` is pointed at a different database and a different descriptions file.

# 5. REPO STRUCTURE

```
nl2sql-agent/
├── data/
│   ├── build_db.py                 # builds data/business.db (seeded)
│   ├── schema_descriptions.yaml    # business descriptions, enums, units
│   └── examples.yaml               # curated question -> SQL few-shot pairs (NOT the eval set)
├── src/
│   ├── config.py                   # settings, flags, paths, limits
│   ├── llm.py                      # get_llm(), get_embeddings(), retry + rate limiter
│   ├── db.py                       # read-only engine, schema inspection, safe execution
│   ├── schema_index.py             # builds the Chroma schema knowledge base
│   ├── retrieval.py                # semantic retrieval of schema + examples
│   ├── prompts.py                  # all prompt templates (database-aware)
│   ├── validator.py                # SQL safety + schema-grounding validation
│   ├── agent.py                    # the generate -> validate -> execute -> repair pipeline
│   ├── conversation.py             # follow-up rewriting + chat history
│   └── utils.py
├── app.py                          # Streamlit UI
├── eval/
│   ├── questions.json              # question set with gold SQL (schema in Phase 5)
│   ├── run_eval.py                 # metrics -> eval/results.md
│   └── ablation.py                 # component ablations -> eval/ablation.md
├── tests/                          # pytest tests (LLM mocked)
├── docs/architecture.md            # Mermaid diagram + design notes
├── docs/img/                       # placeholder for screenshots
├── .env.example, .gitignore, requirements.txt, README.md
```

# 6. PHASES

Complete and verify each phase before moving on. After each phase, run the code or tests, fix failures, then summarize what you did and what I should verify manually.

## Phase 1: Database and read-only access

- Implement `data/build_db.py` and `schema_descriptions.yaml` as described in Section 4. Run it and print row counts per table.
- `src/db.py`:
  - Open the database with a **read-only** connection (for SQLite, use the read-only URI mode; for other databases, document that a read-only DB user must be used).
  - Provide `get_schema()` returning tables, columns, types, primary keys, foreign keys via SQLAlchemy inspection.
  - Provide `execute_select(sql, limit, timeout)` that enforces a maximum row limit and a timeout, returns columns and rows, and never raises raw driver errors to the UI (wrap them in a typed error with the message).
- Gate: schema inspection and a manual SELECT work, and a write attempt (INSERT/DROP) fails at the connection level.

## Phase 2: Schema knowledge base (semantic retrieval)

- `src/schema_index.py` builds a Chroma collection from the live schema plus the descriptions file. Create separate document types, each with metadata (`type`, `table`, `column`):
  1. **Table cards**: table name, business description, columns with types and descriptions, primary key, foreign keys and the joins they imply.
  2. **Column cards**: column name, type, description, a few sample distinct values or value ranges (queried from the DB, capped and sanitized), enumerations.
  3. **Relationship cards**: each foreign key as a plain-language join hint (for example "orders.customer_id references customers.id").
  4. **Example cards**: curated question to SQL pairs from `data/examples.yaml` (about 15 to 20 pairs, covering joins, aggregations, GROUP BY with HAVING, date filters, subqueries, ranking, NULL handling). These must be DIFFERENT from the evaluation questions.
- `src/retrieval.py`: given a question, retrieve the top relevant table/column/relationship cards and the top few example cards. Ensure that every table needed by retrieved column cards is also included (retrieve by table closure), and that join paths between retrieved tables are included. Return a structured object and a formatted schema context string.
- Embeddings: default local `BAAI/bge-small-en-v1.5` (free, offline, no Gemini quota use). Make `EMBEDDING_MODEL` configurable. Store the embedding model name with the index and warn clearly if the configured model differs from the one used to build the index (instruct to rebuild with `--reset`).
- CLI: `python -m src.schema_index [--reset]` builds the index; `python -m src.retrieval "question"` prints what would be retrieved so retrieval can be checked before any SQL generation exists.
- Gate: for 8 manual questions, the retrieved context contains all tables the correct SQL needs.

## Phase 3: Database-aware prompt templates

- In `src/prompts.py`, create LangChain `ChatPromptTemplate`s that are clearly database-aware:
  - **System prompt:** role, the exact SQL dialect (from the connected database), rules: use ONLY tables and columns present in the provided schema context; never invent names; produce a single read-only SELECT statement; use explicit JOINs with the provided join hints; qualify columns with table aliases; always LIMIT detail queries; handle NULLs and date functions according to the dialect; if the question cannot be answered from the schema, return a structured "cannot answer" with a reason instead of guessing; if the question is ambiguous, state the assumption made.
  - **Context block:** the retrieved schema context and the retrieved example pairs as few-shot examples.
  - **Conversation block:** recent chat history (last N turns) when relevant.
  - **Output:** structured output (Pydantic): `{sql: str | null, assumptions: list[str], cannot_answer_reason: str | null}`.
- A separate **repair prompt**: given the question, the failed SQL, the error message or validation violation, and the schema context, return a corrected query in the same structure.
- A separate **explanation prompt**: given the question, the executed SQL, and the (truncated) result rows, produce a short, accurate natural-language answer. It must only state facts present in the results, and must not invent numbers.
- Keep all templates in one file with clear names and docstrings. Make the number of few-shot examples and history turns configurable.

## Phase 4: Validation, execution, and repair loop

- `src/validator.py` runs these checks in order and returns a list of violations:
  1. Parse with `sqlglot` for the configured dialect. Reject on parse failure.
  2. Exactly one statement.
  3. Statement type must be SELECT (CTEs allowed). Reject INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, PRAGMA, ATTACH, and anything that is not a read query.
  4. **Schema grounding check:** every table referenced must exist in the live schema, and every column reference must exist in the referenced tables (resolve aliases and CTEs correctly). Report exact unknown names. This is how the project proves queries are grounded rather than hallucinated.
  5. Enforce or inject a LIMIT on non-aggregate queries according to config.
  6. Optional dry run: ask the database to plan the query (for example EXPLAIN) before executing.
- `src/agent.py` implements the pipeline `answer_question(question, chat_history) -> Result`:
  1. Rewrite the question into a standalone question if there is chat history and it appears to be a follow-up (skip the call when not needed).
  2. Retrieve schema context and examples.
  3. Generate SQL (structured output).
  4. Validate. If violations exist, run the repair prompt (max `MAX_REPAIR_ATTEMPTS`, default 2), then validate again.
  5. Execute safely. On a database error, repair once with the error message (counts toward the same repair limit).
  6. Optionally generate the natural-language explanation.
  7. Return a `Result` object: `question`, `standalone_question`, `sql`, `assumptions`, `columns`, `rows`, `row_count`, `truncated`, `answer_text`, `retrieved_tables`, `repair_attempts`, `violations_history`, `llm_calls`, `latency_ms`, `status` (`ok` | `cannot_answer` | `failed`), `error`.
- Never execute SQL that failed validation. If it still fails after the repair limit, return `status = failed` with a clear message and the SQL attempted.
- Gate: 10 manual questions of increasing difficulty run end to end.

## Phase 5: Tests and evaluation

**pytest** (LLM mocked so tests run without a key or network):
- DB builds deterministically; read-only connection rejects writes.
- Validator: accepts valid SELECTs; rejects multi-statement input, DML/DDL, unknown tables, unknown columns, bad aliases; handles CTEs and subqueries correctly.
- Retrieval returns the needed tables for sample questions.
- Agent loop: repair is triggered on a validation failure and on a DB error, and stops at the limit.
- Put real-API tests in a separate, clearly marked file that runs only when `GOOGLE_API_KEY` is set.

**`eval/questions.json` schema:**
`[{"id", "intent_id", "question", "gold_sql", "difficulty": "easy|medium|hard", "category": "select_filter|join|aggregate|group_having|date|subquery|ranking|null_handling|unanswerable|unsafe"}]`

- Build about 30 distinct intents. For each answerable intent, write **3 paraphrases** with the same `intent_id` (different wording, synonyms, casual vs formal phrasing) to test robustness across varied phrasings. Include about 3 `unanswerable` questions (asking for data not in the schema) and about 3 `unsafe` requests (for example asking to delete or modify data).
- Write gold SQL by hand, verify every gold query executes, and check each result is non-trivial. These questions must be different from the few-shot examples in `data/examples.yaml`.

**`eval/run_eval.py`** computes and writes `eval/results.md`:
- **Execution success rate:** generated SQL executes without error.
- **Execution accuracy:** the result set of the generated SQL matches the gold SQL result (compare as order-insensitive multisets of rows unless the gold query has ORDER BY; normalize floats and column order, and document the comparison rule).
- **Grounding violation rate:** percent of first-attempt generations that referenced unknown tables or columns (before repair), and after repair.
- **Repair rate and success-after-repair rate.**
- **Phrasing consistency:** for each intent, the fraction of its paraphrases that reach a correct result (and the share of intents where all paraphrases succeed).
- **Safety:** percent of `unsafe` questions correctly refused or blocked; percent of `unanswerable` questions correctly returned as `cannot_answer`.
- Breakdowns by difficulty and category, plus average latency and LLM calls per question.
- Optional `--judge` flag for an LLM-judged quality score of the natural-language explanation, clearly labelled as LLM-judged.
- Automatic throttling for the Gemini rate limit.

**`eval/ablation.py`** compares configurations on the same question set (use the exact same questions and report the same metrics), writing `eval/ablation.md`:
1. Full system (semantic schema retrieval + examples + validation/repair).
2. Full schema dumped into the prompt (no semantic retrieval).
3. No few-shot examples.
4. No validation/repair loop.
5. Generic prompt (no database-aware rules) instead of the custom templates.
- Also run a retrieval-only comparison of embedding models (`BAAI/bge-small-en-v1.5`, `BAAI/bge-base-en-v1.5`, `sentence-transformers/all-MiniLM-L6-v2`): for each, build a separate index and measure whether the retrieved context contains every table required by the gold SQL (table recall at k = 3, 5, 8), plus index build time. Pick the default embedding model from the real results (prefer the smaller/faster model if results are close) and explain the choice in the README.
- Never fabricate numbers. If a run fails or is skipped because of quota, report that explicitly.

## Phase 6: Streamlit UI (`app.py`)

- `st.chat_message` chat with `st.session_state` history; follow-up questions must work (for example "now only for 2025", "which of those are in Mumbai?").
- For each answer show: the natural-language answer, the **generated SQL** in a code block (with a copy option), a results table (`st.dataframe`) with row count and a "truncated" notice when applicable, an expander "Schema context used" (retrieved tables/columns/examples), the assumptions the model made, and repair attempts if any.
- Optional: a simple auto-chart (bar/line) when the result has one categorical or date column and one numeric column.
- Sidebar: connected database name and tables list, embedding model in use, LLM calls and latency for the last question, an "Example questions" list (clickable), a "Clear chat" button, a read-only/safety note.
- Cache the schema index and agent with `st.cache_resource`.
- Friendly errors for: missing API key, missing database, empty schema index (tell the user which command to run), rate limit reached, blocked response, and failed queries. Never log or store user data to disk.

## Phase 7: Documentation and polish

- **README.md:** overview, architecture diagram (Mermaid), features, setup (venv, install, `.env`, build database, build schema index, run app), project structure, the evaluation results and ablation tables copied from real runs, the embedding-model decision, Gemini free-tier notes (limits are per project; content on the free tier may be used by Google to improve products; do not enter sensitive data), how to point the app to another database (DATABASE_URL, descriptions file, read-only DB user), safety design (read-only connection, SELECT-only validation, row limit, schema grounding check), limitations (ambiguous questions, complex window functions, very large schemas), and troubleshooting (429 errors, blocked responses, model not found).
- **docs/architecture.md:** the pipeline, data flow, design decisions (why retrieval over full-schema prompting, why validation before execution, why repair loop is bounded, how few-shot examples are chosen).
- **.gitignore:** `.env`, `data/business.db`, `data/chroma*/`, `__pycache__/`, `.venv/`.
- Include a clear disclaimer that generated SQL should be reviewed before use on production data.

# 7. CODE QUALITY RULES

- Type hints, docstrings, small functions. No hard-coded paths or keys; everything configurable through `config.py` and `.env`.
- Prompts live in `prompts.py`. Use structured output (Pydantic) wherever an LLM returns a label or JSON.
- Use `logging` instead of `print` in library code. Handle API and database errors with retries and timeouts where appropriate.
- Do not use string concatenation to build SQL from user input. User text goes only to the LLM; only validated SELECT statements reach the database.
- If something is ambiguous, choose a sensible default, note it in the README, and continue. Ask me only if you are truly blocked (for example, no API key available).

# 8. DEFINITION OF DONE

1. `pip install -r requirements.txt` works in a clean virtual environment.
2. `python data/build_db.py` creates the database; `python -m src.schema_index --reset` builds the schema index.
3. `streamlit run app.py` gives a working chat with follow-ups, showing SQL, results, and the schema context used, using Gemini.
4. Writes and unsafe SQL are impossible: blocked at validation and at the read-only connection.
5. `pytest` passes (LLM mocked), and `python -m eval.run_eval` produces `eval/results.md` from real runs.
6. `python -m eval.ablation` produces `eval/ablation.md` from real runs, and the README's design claims (semantic retrieval, custom prompts, validation) are supported by those numbers.
7. README is complete and accurate, with no invented numbers.

implement complete project


--- PROMPT ENDS HERE ---
