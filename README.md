# Vanna AI Agent

A production-ready AI-powered SQL agent that converts natural language questions into SQL queries using Google Gemini LLM. Query your databases conversationally without writing any SQL.

## Features

- **Natural Language Querying**: Ask questions in plain English, get SQL results automatically
- **Google Gemini Integration**: Powered by Gemini 2.5 Flash for fast and accurate SQL generation
- **DuckDB Database**: In-memory analytical engine for efficient data processing
- **Agent Memory**: ChromaDB vector database that learns from past interactions
- **Role-Based Access Control**: Cookie-based authentication with admin & user roles
- **Web UI & REST API**: Built-in web interface + RESTful API endpoints
- **Auto CSV Ingestion**: Automatically loads and processes CSV files on startup
- **Production Ready**: Deployable with Docker, Gunicorn, or standard WSGI servers

## System Architecture

```
┌─────────────────────────────────────────────────┐
│         User Interface (Web Browser)            │
│        http://127.0.0.1:5000                    │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│         Flask Web Server + REST API             │
│                                                 │
│  ┌─────────┐  ┌─────────┐  ┌────────────────┐   │
│  │  Vanna  │  │ Gemini  │  │ ChromaDB       │   │
│  │  Agent  │──│ LLM     │  │ (Memory Store) │   │
│  │         │  │ Service │  │                │   │
│  └────┬────┘  └────┬────┘  └────────────────┘   │
└───────┼────────────┼──────────────────────────
        │            │
        └──────┬─────┘
               │
               ▼
        ┌──────────────────┐
        │  DuckDB Engine   │
        │  (SQL Executor)  │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  CSV Data File   │
        │  (layoffs table) │
        └──────────────────┘
```

### Data Flow

```
1. Question Input
   └→ "Which company had the most layoffs?"

2. Gemini Processing
   └→ Analyzes question + retrieves similar past queries from ChromaDB
   └→ Generates SQL: SELECT Company, SUM("Number of Workers") ...

3. Query Execution
   └→ DuckDB executes the generated SQL against layoffs table
   └→ Returns data rows

4. Response Formatting
   └→ Formats results for UI display
   └→ Saves question-SQL pair to ChromaDB memory

5. User Display
   └→ Shows results in table format
   └→ Displays generated SQL for transparency
   └→ Updates agent memory for future queries
```

## Requirements

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|------------|
| Python | 3.8+ | 3.10+ |
| RAM | 4 GB | 8 GB |
| Disk Space | 500 MB | 2 GB |
| Operating System | Windows/Linux/macOS | Ubuntu 20.04+ |
| Network | Internet (API calls) | Stable connection |

### API Requirements

| Service | Key | Cost | Purpose |
|---------|-----|------|---------|
| **Google Gemini** | Required | Free (60 req/min) | LLM for SQL generation |
| **Vanna Cloud** | Optional | Optional | Advanced features |

### Python Dependencies

```
vanna==2.0.0rc1              # Core AI framework
google-genai                 # Google Gemini API client
duckdb==0.9.0+              # SQL database engine
chromadb==0.4.0+            # Vector embeddings store
flask==3.0+                 # Web framework
python-dotenv               # Environment configuration
pandas                       # Data manipulation
sqlalchemy                   # SQL toolkit
requests                     # HTTP client
rich                         # Terminal formatting
uvicorn                      # ASGI server (optional)
```

See `requirements.txt` for specific versions.

Quick Start

Prerequisites

- Python 3.8+ installed
- Google Gemini API key (free from https://makersuite.google.com/app/apikey)
- 4GB RAM minimum

Clone & Setup (2 minutes)

```bash
# Clone repository
git clone <repo-url>
cd "Vanna Agent"

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# OR
source .venv/bin/activate             # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

Configure (1 minute)

Create `.env` file:

```env
GEMINI_API_KEY=paste_your_key_here
DUCKDB_PATH=./data/your_database.duckdb
CHROMA_PERSIST_DIR=./data/chroma_db
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
```

Run (30 seconds)

```bash
python src/app.py
```

**Expected output:**
```
INFO:vanna.core.agent.agent:Loaded vanna.core.agent.agent module
INFO:__main__:Gemini LLM Service initialized with model: gemini-2.5-flash
INFO:__main__:Ingesting CSV before starting server...
Your app is running at: http://localhost:5000
```

Use (Open browser)

Visit: **http://127.0.0.1:5000**
