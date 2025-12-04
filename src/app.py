import os
import logging
import textwrap
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from vanna.integrations.google import GeminiLlmService

from vanna import Agent
from vanna.core.registry import ToolRegistry
from vanna.tools import RunSqlTool

from vanna.integrations.duckdb import DuckDBRunner

from vanna.tools.agent_memory import SaveQuestionToolArgsTool, SearchSavedCorrectToolUsesTool, SaveTextMemoryTool
from vanna.integrations.chromadb import ChromaAgentMemory

from vanna.core.user import UserResolver, User, RequestContext

from fastapi import FastAPI

from vanna.servers.fastapi import VannaFastAPIServer
from vanna.servers.base import ChatHandler
from vanna.servers.fastapi.routes import register_chat_routes
from vanna.core.system_prompt import DefaultSystemPromptBuilder
from vanna.core.enhancer import LlmContextEnhancer
from vanna.core.lifecycle import LifecycleHook
from config import settings

import asyncio
import duckdb
import uvicorn

#from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

SCHEMA_REFERENCE = textwrap.dedent(
    f"""
    DATA SUMMARY:
    TABLE nissan_service_dataset (
        modelname TEXT,
        fueltype TEXT,           -- Petrol or Diesel
        mileage_in_km INTEGER,   -- e.g. 80000
        duration_in_month INTEGER,
        partsprice DECIMAL,
        laborprice DECIMAL,
        totalprice DECIMAL,      -- partsprice + laborprice
        service_replacements_check_reference TEXT -- comma separated codes
    )

    DEDUPED VIEW:
    VIEW nissan_service_replacements_unique (
        modelname TEXT,
        mileage_in_km INTEGER,
        unique_replacements TEXT  -- comma-separated, distinct & sorted
    )

    QUICK HINTS:
    * Filter mileage_in_km for interval-specific answers.
    * Use nissan_service_replacements_unique when a question asks
      "per model" to avoid duplicate fuel-type rows.
    * totalprice already stores parts + labor.
    """
).strip()

class NissanSystemPromptBuilder(DefaultSystemPromptBuilder):
    _cached_prompt = None 

    def __init__(self, dataset_doc: str):
        super().__init__()
        self.dataset_doc = dataset_doc

    async def build_system_prompt(self, user, tools):
        # If already built once reuse it
        if self._cached_prompt is not None:
            return self._cached_prompt

        base_prompt = await super().build_system_prompt(user, tools) or ""

        self._cached_prompt = f"{base_prompt}\n\n{self.dataset_doc}"
        return self._cached_prompt

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


class NoopLlmContextEnhancer(LlmContextEnhancer):
    """Skip expensive memory lookups—system prompt already encodes schema."""

    async def enhance_system_prompt(self, system_prompt, user_message, user):
        return system_prompt


llm_context_enhancer = NoopLlmContextEnhancer()

llm = GeminiLlmService(
    model="gemini-2.0-flash-lite",
    api_key=settings.GEMINI_API_KEY
)

logger.info(f"Gemini LLM Service initialized with model")

def ingest_csv():
    """Ingest CSV file into DuckDB"""
    csv_path = os.path.join(os.path.dirname(__file__), "data", "data.csv")
    table_name = "nissan_service_dataset"
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        return False
    
    try:
        con = duckdb.connect(settings.DUCKDB_PATH)
        
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT DISTINCT
                modelname,
                fueltype,
                mileage_in_km,
                duration_in_month,
                partsprice,
                laborprice,
                totalprice,
                TRIM(service_replacements_check_reference) AS service_replacements_check_reference
            FROM read_csv_auto(
                '{csv_path}',
                header=True,
                normalize_names=True
            )
        """)

        con.execute("""
            CREATE OR REPLACE VIEW nissan_service_replacements_unique AS
            WITH exploded AS (
                SELECT
                    modelname,
                    mileage_in_km,
                    REGEXP_REPLACE(TRIM(token), '\\s+', ' ') AS replacement
                FROM nissan_service_dataset,
                UNNEST(STR_SPLIT(service_replacements_check_reference, ',')) AS tokens(token)
            )
            SELECT
                modelname,
                mileage_in_km,
                STRING_AGG(DISTINCT replacement, ', ' ORDER BY replacement) AS unique_replacements
            FROM exploded
            GROUP BY modelname, mileage_in_km
        """)
        
        result = con.execute(f"SELECT COUNT(*) as cnt FROM {table_name}").fetchall()
        unique_rows = con.execute(
            "SELECT COUNT(*) FROM (SELECT modelname, mileage_in_km FROM nissan_service_replacements_unique)"
        ).fetchall()
        logger.info(f"✓ Table '{table_name}' created with {result[0][0]} rows")
        logger.info("✓ View 'nissan_service_replacements_unique' ready for %s mileage/model pairs", unique_rows[0][0])
        
        con.close()
        return True
        
    except Exception as e:
        logger.error(f"✗ Error ingesting CSV: {e}")
        return False

db_runner = DuckDBRunner(database_path=settings.DUCKDB_PATH)
db_tool = RunSqlTool(sql_runner=db_runner)

# Stop Reloading ONNX Model
#embedding_fn = ONNXMiniLM_L6_V2() 
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Agent memory using ChromaDB (local vector database) - Agent to learn from past interactions by storing successful question-SQL pair
agent_memory = ChromaAgentMemory(
    collection_name="vanna_memory",
    persist_directory=settings.CHROMA_PERSIST_DIR,
    embedding_function=embedding_fn
)

async def warm_up_memory():
    logger.info("Warming up Chroma memory...")
    await agent_memory.save_text_memory(
        "Warmup",
        {"purpose": "warmup"}
    )
    logger.info("Chroma warmed up.")

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
    lifecycle_hooks=[latency_hook],
    llm_context_enhancer=llm_context_enhancer,
)

agent.max_iterations = 1
agent.max_retries = 0
agent.enhancer = None

# Create FastAPI server wrapper
_fastapi_app: FastAPI | None = None
fastapi_server = VannaFastAPIServer(agent)


def _register_startup_events(app: FastAPI) -> None:
    """Ensure DuckDB and Chroma are ready regardless of how the ASGI app is launched."""

    @app.on_event("startup")
    async def _startup_tasks():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, ingest_csv)
        await warm_up_memory()


def create_app() -> FastAPI:
    """Expose FastAPI app instance (for uvicorn/gunicorn)."""
    global _fastapi_app
    if _fastapi_app is None:
        _fastapi_app = fastapi_server.create_app()
        _register_startup_events(_fastapi_app)
    return _fastapi_app


def register_vanna_routes(app: FastAPI, config: dict | None = None) -> None:
    """
    Attach Vanna chat endpoints to an existing FastAPI application.

    Call this from any other service to reuse the agent without running the standalone server.
    """
    chat_handler = ChatHandler(agent)
    register_chat_routes(
        app,
        chat_handler,
        config
        or {
            "dev_mode": False,
            "cdn_url": "https://img.vanna.ai/vanna-components.js",
        },
    )


def run_server():
    """Run FastAPI server with uvicorn."""
    logger.info("Starting Vanna FastAPI server...")
    uvicorn.run(
        create_app(),
        host=settings.FLASK_HOST,
        port=settings.FLASK_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    logger.info("Ingesting CSV before starting server...")
    ingest_csv()
    logger.info("Server starting...")
    run_server()
