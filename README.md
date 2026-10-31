# Natural Language to SQL Agent

An enterprise-ready AI assistant that translates natural-language business questions into executable, schema-grounded SQL queries over relational databases. Built with **LangChain**, **Google Gemini**, **Chroma**, and **sqlglot**, featuring semantic schema retrieval, AST validation, bounded self-repair, and conversational analytics.

---

## Architecture

```mermaid
flowchart TD
    User([User Question]) --> RewriteCheck{Has Chat History?}
    RewriteCheck -- Yes --> LLM_Rewrite[LLM Follow-up Rewriter]
    RewriteCheck -- No --> StandaloneQ[Standalone Question]
    LLM_Rewrite --> StandaloneQ

    StandaloneQ --> EmbedQ[Embed Query: BAAI/bge-small-en-v1.5]
    EmbedQ --> Chroma[(Chroma Knowledge Base)]
    Chroma --> RetrievedContext[Retrieved Cards: Tables, Columns, Relationships, Examples]
    RetrievedContext --> TableClosure[Table Closure & Join Path Expansion]

    TableClosure --> PromptGen[Database-Aware Prompt Template]
    PromptGen --> GeminiGen[Google Gemini: Structured Output]
    GeminiGen --> CandidateSQL[Candidate SQL Query]

    CandidateSQL --> SqlglotParser[sqlglot AST Parser & Security Guard]
    SqlglotParser --> ASTCheck{Passes AST & Schema Grounding?}

    ASTCheck -- Violations Found --> RepairAttempts{Attempts < MAX_REPAIR_ATTEMPTS?}
    RepairAttempts -- Yes --> GeminiRepair[Gemini SQL Self-Repair Prompt]
    GeminiRepair --> SqlglotParser
    RepairAttempts -- No --> FailHalt[Return Failure: Block Execution]

    ASTCheck -- Valid & Grounded --> SafeExec[Execute on Read-Only SQLite Engine]
    SafeExec --> DBErrorCheck{Execution Succeeded?}
    DBErrorCheck -- DB Runtime Error --> RepairAttempts

    DBErrorCheck -- Success --> Explainer[Gemini Factual Explainer]
    Explainer --> UI([Streamlit Chat UI: Answer + SQL + Table + Auto-Chart])
```

---

## Core Capabilities

1. **Semantic Schema Retrieval with Table Closure:**
   Retrieves relevant tables, columns, join relationships, and sample values using local Chroma vector storage. Enforces table closure so that any table referenced by a retrieved column or relationship is guaranteed to have its complete schema definition included in the context.
2. **Database-Aware Prompt Templates:**
   Translates questions into dialect-specific SQL (SQLite default, extensible to PostgreSQL/MySQL) with strict rules: qualified table aliases, ANSI joins, NULL handling, and row limits.
3. **Multi-Stage AST Security & Grounding:**
   Uses `sqlglot` to parse the query AST, ensuring single-statement execution, rejecting all non-read commands (INSERT, UPDATE, DELETE, DROP, PRAGMA), and verifying that every table and column reference physically exists in the database.
4. **Self-Healing SQL Repair Loop:**
   If a generated query violates schema grounding or fails database planning, the compiler errors are fed back into a targeted repair prompt. Bounded to a configurable maximum of 2 repair attempts to prevent infinite loops.
5. **Conversational Business Analytics & Auto-Charting:**
   Maintains chat history, resolves follow-up references (e.g. *"now only for 2025"*), generates concise factual explanations, and automatically visualizes results with interactive Streamlit charts.

---

## Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| **LLM** | **Google Gemini** (`gemini-2.0-flash`) via `langchain-google-genai` | High reasoning speed, large context, cost-effective on free tier. |
| **Framework** | **LangChain** (Runnables, Templates, Output Parsers) | Small, controlled, composable pipeline without unnecessary agent graph overhead. |
| **Schema Index** | **Chroma** (Persistent local storage) | Fast offline vector search for tables, columns, relationships, and examples. |
| **Embeddings** | **BAAI/bge-small-en-v1.5** | Local HuggingFace embeddings: 100% offline, zero Gemini quota consumption. |
| **SQL Parser** | **sqlglot** | Dialect-aware AST inspection, security verification, and schema grounding. |
| **Database** | **SQLite** (SQLAlchemy read-only URI mode) | Fully deterministic synthetic business database with 8 relational tables. |
| **UI** | **Streamlit** | Interactive chat interface with SQL inspection, auto-charts, and schema metrics. |
| **Tests** | **pytest** | Comprehensive test suite with mocked LLM for offline execution. |

---

## Repository Structure

```
nl2sql-agent/
├── data/
│   ├── build_db.py                 # Generates reproducible SQLite database (seeded)
│   ├── schema_descriptions.yaml    # Business descriptions, units, and enumerations
│   └── examples.yaml               # Curated few-shot question -> SQL pairs
├── src/
│   ├── config.py                   # Central settings, paths, limits, and environment config
│   ├── llm.py                      # get_llm(), get_embeddings(), retries, and rate limiter
│   ├── db.py                       # Read-only SQLAlchemy engine, inspection, safe execution
│   ├── schema_index.py             # Builds and manages Chroma schema knowledge base
│   ├── retrieval.py                # Semantic retrieval with table closure and join paths
│   ├── prompts.py                  # All database-aware ChatPromptTemplates & Pydantic models
│   ├── validator.py                # sqlglot AST parsing, safety checks, schema grounding
│   ├── agent.py                    # Complete generate -> validate -> execute -> repair pipeline
│   ├── conversation.py             # Chat history management and follow-up question rewriting
│   └── utils.py                    # Formatting helpers and result set comparison logic
├── app.py                          # Streamlit web application
├── eval/
│   ├── questions.json              # 78 evaluation questions across 30 intents (with gold SQL)
│   ├── build_eval_set.py           # Verification script for evaluation questions
│   ├── run_eval.py                 # End-to-end evaluation harness -> eval/results.md
│   └── ablation.py                 # Component and embedding model ablation study -> eval/ablation.md
├── tests/                          # 28 pytest tests (LLM mocked)
├── docs/architecture.md            # Comprehensive system architecture & Mermaid diagrams
├── .env.example                    # Environment variable template
├── requirements.txt                # Pinned dependencies
└── README.md                       # Documentation & run instructions
```

---

## Quickstart & Installation

### 1. Prerequisites
- Python 3.11+ (Python 3.12 recommended)
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### 2. Environment Setup
```bash
# Clone the repository and navigate into it
cd "SQL chatbot"

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and enter your Gemini API key:
```ini
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
DATABASE_URL=sqlite:///data/business.db
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
REQUESTS_PER_MINUTE=15
MAX_REPAIR_ATTEMPTS=2
ENABLE_NL_EXPLANATION=true
MAX_EXECUTION_ROWS=100
QUERY_TIMEOUT_SECONDS=10.0
```

### 4. Build the Synthetic Database
Generates `data/business.db` with 300 customers, 80 products, 3,000 orders, 8,898 order items, and 2,824 payments:
```bash
python data/build_db.py
```

### 5. Build the Schema Knowledge Base
Indexes tables, columns, join relationships, and few-shot examples into local Chroma:
```bash
python -m src.schema_index --reset
```

### 6. Run the Application
Launch the Streamlit conversational UI:
```bash
streamlit run app.py
```

---

## Empirical Benchmark & Ablation Results

All numbers below were generated by running the automated evaluation suite against the database.

### 1. Embedding Model Comparison (Table Recall Benchmark)

Table recall measures the percentage of questions for which the retrieved schema context contains 100% of the tables required by the gold SQL query across all test queries in `eval/questions.json`:

| Embedding Model | Dimension | Index Build Time | Table Recall @ 3 | Table Recall @ 5 | Table Recall @ 8 |
|---|---|---|---|---|---|
| **`BAAI/bge-small-en-v1.5`** (Selected Default) | **384** | **9.60s** | **100.0%** | **100.0%** | **100.0%** |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 8.19s | 94.4% | 100.0% | 100.0% |

**Decision:** `BAAI/bge-small-en-v1.5` achieves **100.0% table recall** at $k=3$ with negligible build time difference, completely offline, using zero API quota.

### 2. Architecture Component Ablation

| Configuration | Retrieval Strategy | Few-Shot Examples | AST Validation & Repair | Custom Prompt Rules | Performance Impact |
|---|---|---|---|---|---|
| **Full System (Proposed)** | Semantic (Chroma + Closure) | 3 Examples | Active (sqlglot + DB Repair) | Dialect-Aware | **Optimal (High Accuracy, Minimal Hallucination)** |
| **No Retrieval (Full Schema)** | Full Schema Injected | Yes | Active | Dialect-Aware | Increased token usage, higher latency |
| **No Few-Shot Examples** | Semantic | None | Active | Dialect-Aware | Decreased syntax consistency on complex queries |
| **No Validation & Repair** | Semantic | Yes | Disabled | Dialect-Aware | Uncaught runtime syntax errors crash queries |
| **Generic Prompt** | Semantic | Yes | Active | Generic | Ambiguous date handling & unqualified columns |

---

## Running Tests & Evaluation

### Run Unit Tests (LLM Mocked)
The unit tests verify database determinism, read-only enforcement, AST validator rules, schema grounding, and the repair loop without making network calls or using API quota:
```bash
pytest -v
```

### Run Full Benchmark Evaluation
Evaluates all 78 questions across 30 intents and writes the benchmark report to `eval/results.md`:
```bash
python -m eval.run_eval
```

### Run Ablation Studies
Executes the embedding model comparison and pipeline ablation:
```bash
python -m eval.ablation
```

---

## Pointing to Another Database

The agent is completely database-agnostic. To connect to an external production or staging database:
1. Update `DATABASE_URL` in `.env`:
   - PostgreSQL: `postgresql://readonly_user:password@hostname:5432/dbname`
   - MySQL: `mysql+pymysql://readonly_user:password@hostname:3306/dbname`
2. Update `data/schema_descriptions.yaml` with your table and column descriptions.
3. Rebuild the schema knowledge base:
   ```bash
   python -m src.schema_index --reset
   ```
4. Always configure a dedicated **read-only database user** with SELECT-only privileges for external databases.

---

## Safety & Security Design

- **Connection-Level Read-Only Mode:** SQLite is opened via `sqlite:///file:...?mode=ro`, preventing modifications at the driver level.
- **AST Whitelist Verification:** `sqlglot` parses every query AST and rejects any statement containing `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, or `PRAGMA`.
- **Schema Grounding Validation:** Every referenced table and column is checked against the live database catalog before execution. Queries referencing hallucinated names are intercepted and self-repaired.
- **Row Limits & Timeouts:** Non-aggregate queries have automatic LIMIT injection, and execution times are capped by configurable timeouts.
- **Zero Disk Logging of User Data:** Conversations are kept in memory for the active session and never persisted to disk.

---

## Gemini Free-Tier Notes & Best Practices

- **Rate Limits:** The Gemini free tier provides 15 requests per minute (RPM). The built-in rate limiter throttles calls automatically.
- **Data Privacy Notice:** Per Google AI Studio terms, prompts and responses on the free tier may be reviewed by human reviewers to train models. Do not submit sensitive personal or confidential information.
- **Model Recommendation:** Checked as of October 2026: `gemini-2.0-flash` (or `gemini-1.5-flash`) offers the optimal combination of structured output accuracy, speed, and free-tier allowance.

---

## Disclaimer

> **Production Notice:** This software is designed as an agentic analytical assistant. While protected by multi-layered AST validation and read-only database connections, generated SQL queries should always be reviewed before applying them to mission-critical production systems.

