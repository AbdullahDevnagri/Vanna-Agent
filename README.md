Vanna AI Agent
<img width="1260" height="826" alt="second" src="https://github.com/user-attachments/assets/b44895a0-e99f-4d08-84ab-45728dc8738c" />
<img width="1860" height="657" alt="sql queries" src="https://github.com/user-attachments/assets/0b313231-9628-455c-a6a9-fd7f742354a6" />

A production-ready AI-powered application that converts natural language questions into SQL queries using Google Gemini LLM. The system enables users to query company layoff data through conversational interfaces without writing SQL.

Features
- Natural Language Processing: Ask questions in plain English and get SQL results
- Google Gemini Integration: Powered by Gemini 2.5 Flash for fast and accurate query generation
- DuckDB Database: In-memory analytical database for efficient CSV data processing
- Agent Memory: ChromaDB-based vector memory that learns from past interactions
- Role-Based Access Control: Cookie-based authentication with admin and user roles
- Flask REST API: RESTful API server for easy integration
- Automatic CSV Ingestion: Automatically loads CSV data into DuckDB on startup

Data Flow

```
CSV File (data.csv)
    │
    ▼
[CSV Ingestion Function]
    │
    ▼
DuckDB Table (layoffs)
    │
    ▼
[SQL Query Execution]
    │
    ▼
Results → User
```

Requirements
System Requirements

- Python: 3.8 or higher
- Operating System: Windows, Linux, or macOS
- Memory: Minimum 4GB RAM (8GB recommended)
- Storage: ~500MB for dependencies and data

Python Dependencies
- vanna (2.0.0rc1): Core Vanna AI framework
- google-genai: Google Gemini API client
- duckdb: In-memory analytical database
- chromadb: Vector database for agent memory
- flask: Web framework for REST API
- python-dotenv: Environment variable management
- pandas: Data manipulation (used by Vanna)
- sqlalchemy: SQL toolkit (used by Vanna)
- uvicorn: ASGI server (optional, for production)
- requests: HTTP library

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd "Vanna Agent"
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Google Gemini API Key (Required)
GEMINI_API_KEY=your_gemini_api_key_here

# Vanna API Key (Optional - for cloud features)
VANNA_API_KEY=your_vanna_api_key_here

# Database Configuration
DUCKDB_PATH=./src/data/your_database.duckdb
CHROMA_PERSIST_DIR=./src/data/chroma_db

# Flask Server Configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
```
