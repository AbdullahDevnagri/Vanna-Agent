import os
import logging
import textwrap
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from flask import Flask, request, jsonify

from vanna.integrations.google import GeminiLlmService

from vanna import Agent
from vanna.core.registry import ToolRegistry
from vanna.tools import RunSqlTool

from vanna.integrations.duckdb import DuckDBRunner

from vanna.tools.agent_memory import SaveQuestionToolArgsTool, SearchSavedCorrectToolUsesTool, SaveTextMemoryTool
from vanna.integrations.chromadb import ChromaAgentMemory

from vanna.core.user import UserResolver, User, RequestContext

from vanna.servers.flask import VannaFlaskServer
from vanna.core.system_prompt import DefaultSystemPromptBuilder
from vanna.core.lifecycle import LifecycleHook
from config import settings

import asyncio
import duckdb

SCHEMA_REFERENCE = textwrap.dedent(
    f"""
    DATA CONTEXT:
    - DuckDB database lives at {settings.DUCKDB_PATH}.
    - Primary table: nissan_service_dataset (one row per scheduled service interval per model/fuel type).

    TABLE nissan_service_dataset COLUMNS:
    - modelname (TEXT): model name such as Kicks, Micra, Terrano, Magnite, etc.
    - fueltype (TEXT): Petrol or Diesel.
    - mileage_in_km (INTEGER): odometer reading of the scheduled service (e.g., 80000).
    - duration_in_month (INTEGER): months since purchase that align with the mileage.
    - partsprice (DECIMAL): parts cost for that service.
    - laborprice (DECIMAL): labor cost for that service.
    - totalprice (DECIMAL): total service cost (parts + labor).
    - service_replacements_check_reference (TEXT): comma-separated codes describing what gets replaced (AF, OF, ENG 10W30, etc.).

    QUERY TIPS:
    - Filter on mileage_in_km to answer questions like "service replacements at 80,000 km".
    - Break out comma-separated replacements with string functions (e.g., SPLIT or LIKE) if needed.
    - Aggregate by modelname and fueltype for comparisons.
    - Use totalprice = partsprice + laborprice (already stored) when summing costs.
    """
).strip()


class NissanSystemPromptBuilder(DefaultSystemPromptBuilder):
    def __init__(self, dataset_doc: str):
        super().__init__()
        self.dataset_doc = dataset_doc

    async def build_system_prompt(self, user, tools):
        base_prompt = await super().build_system_prompt(user, tools) or ""
        return f"{base_prompt}\n\n{self.dataset_doc}"


system_prompt_builder = NissanSystemPromptBuilder(SCHEMA_REFERENCE)


class LatencyLoggingHook(LifecycleHook):
    """Logs end-to-end latency for each user question."""

    def __init__(self):
        self._start_times: dict[int, float] = {}

    def _task_key(self) -> int | None:
        task = asyncio.current_task()
        return id(task) if task else None

    async def before_message(self, user, message: str):
        key = self._task_key()
        if key is not None:
            self._start_times[key] = time.perf_counter()
        return None  

    async def after_message(self, result):
        key = self._task_key()
        if key is None:
            return
        start = self._start_times.pop(key, None)
        if start is None:
            return
        duration = time.perf_counter() - start
        logger.info("Agent message latency: %.2fs", duration)


latency_hook = LatencyLoggingHook()


llm = GeminiLlmService(
    model="gemini-2.5-flash",
    api_key=settings.GEMINI_API_KEY
)

logger.info(f"Gemini LLM Service initialized with model")

# Connect to DuckDB and ingest CSV
def ingest_csv():
    """Ingest CSV file into DuckDB"""
    csv_path = os.path.join(os.path.dirname(__file__), "data", "data.csv")
    table_name = "nissan_service_dataset"
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        return False
    
    try:
        # Connect to DuckDB
        con = duckdb.connect(settings.DUCKDB_PATH)
        
        # Drop table if exists
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Create table from CSV
        con.execute(f"""
            CREATE TABLE {table_name} AS
            SELECT *
            FROM read_csv_auto(
                '{csv_path}',
                header=True,
                normalize_names=True
            )
        """)
        
        # Verify
        result = con.execute(f"SELECT COUNT(*) as cnt FROM {table_name}").fetchall()
        logger.info(f"✓ Table '{table_name}' created with {result[0][0]} rows")
        
        con.close()
        return True
        
    except Exception as e:
        logger.error(f"✗ Error ingesting CSV: {e}")
        return False

# Build DuckDB tool
db_runner = DuckDBRunner(database_path=settings.DUCKDB_PATH)
db_tool = RunSqlTool(sql_runner=db_runner)

# Agent memory using ChromaDB (local vector database) - Agent to learn from past interactions by storing successful question-SQL pair
agent_memory = ChromaAgentMemory(
    collection_name="vanna_memory",
    persist_directory=settings.CHROMA_PERSIST_DIR
)

# Tools registry
tools = ToolRegistry()
tools.register_local_tool(db_tool, access_groups=['admin', 'user'])
tools.register_local_tool(SaveQuestionToolArgsTool(), access_groups=['admin'])
tools.register_local_tool(SearchSavedCorrectToolUsesTool(), access_groups=['admin', 'user'])
tools.register_local_tool(SaveTextMemoryTool(), access_groups=['admin', 'user'])

# Configure user Authentication (cookie-based)
class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_email = request_context.get_cookie('vanna_email') or 'guest@example.com'
        group = 'admin' if user_email == 'admin@example.com' else 'user'
        return User(id=user_email, email=user_email, group_memberships=[group])
    
user_resolver = SimpleUserResolver()

# Created agent

agent = Agent(
    llm_service=llm,
    tool_registry=tools,
    user_resolver=user_resolver,
    agent_memory=agent_memory,
    system_prompt_builder=system_prompt_builder,
    lifecycle_hooks=[latency_hook]
)

# Create Flask server
server = VannaFlaskServer(agent)

def run_server():
    """Run Flask server"""
    logger.info("Starting Vanna Flask server...")
    server.run(host=settings.FLASK_HOST, port=settings.FLASK_PORT)

if __name__ == "__main__":
    logger.info("Ingesting CSV before starting server...")
    ingest_csv()
    logger.info("Server starting...")
    run_server()