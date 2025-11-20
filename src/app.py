import os
import logging

# Enable detailed logging
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
from config import settings

from vanna.core.user import RequestContext
import asyncio
import duckdb

llm = GeminiLlmService(
    model="gemini-2.5-flash",
    api_key=settings.GEMINI_API_KEY
    )

logger.info(f"Gemini LLM Service initialized with model: gemini-1.5-flash")

# Connect to DuckDB and ingest CSV
def ingest_csv():
    """Ingest CSV file into DuckDB"""
    csv_path = os.path.join(os.path.dirname(__file__), "data", "data.csv")
    table_name = "layoffs"
    
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
            SELECT * FROM read_csv_auto('{csv_path}')
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
    agent_memory=agent_memory
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