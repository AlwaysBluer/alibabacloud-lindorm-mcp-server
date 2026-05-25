import argparse
import os
import secrets
from contextlib import asynccontextmanager
from typing import AsyncIterator
from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .utils import *
from .security import (
    validate_api_key,
    validate_lindorm_instance_id,
    validate_model_name,
    validate_required,
)
from .lindorm_vector_search import LindormVectorSearchClient
from .lindorm_wide_table import LindormWideTableClient


class LindormContext:
    def __init__(self, lindorm_search_client: LindormVectorSearchClient, lindorm_sql_client: LindormWideTableClient):
        self.lindorm_search_client = lindorm_search_client
        self.lindorm_sql_client = lindorm_sql_client


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[LindormContext]:
    """Manage application lifecycle for Lindorm"""
    config = server.config

    vector_search_client = LindormVectorSearchClient(
        search_host=config.get("lindorm_search_host"),
        ai_host=config.get("lindorm_ai_host"),
        username=config.get("username"),
        password=config.get("password"),
        text_embedding_model=config.get("text_embedding_model"),
        text_embedding_dimension=config.get("text_embedding_dimension"),
        use_ssl=config.get("use_ssl"),
        verify_ssl=config.get("verify_ssl"),
    )

    sql_client = LindormWideTableClient(
        table_host=config.get("lindorm_table_host"),
        username=config.get("username"),
        password=config.get("password"),
        database=config.get("table_database")
    )

    try:
        yield LindormContext(vector_search_client, sql_client)
    finally:
        pass


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        supplied_key = request.headers.get("x-api-key")
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            supplied_key = auth_header[7:].strip()

        if not supplied_key or not secrets.compare_digest(supplied_key, self.api_key):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        return await call_next(request)


class AuthenticatedFastMCP(FastMCP):
    api_key: str | None = None

    def sse_app(self):
        app = super().sse_app()
        if self.api_key:
            app.add_middleware(APIKeyMiddleware, api_key=self.api_key)
        return app


mcp = AuthenticatedFastMCP("Lindorm", lifespan=server_lifespan, log_level="ERROR", host="127.0.0.1")


@mcp.tool()
def lindorm_retrieve_from_index(index_name: str, query: str,  content_field: str, vector_field: str,
                                top_k: int = 5, ctx: Context = None) -> str:
    """
    Retrieve from an existing indexes(or knowledgebase) using both full-text search and vector search, and return the aggregated results
    :param index_name: the index name, or known as knowledgebase name
    :param query: the query that you want to search in knowledgebase
    :param content_field: the text field that store the content text. You can get it from the index structure by lindorm_get_index_mappings tool
    :param vector_field: the vector field that store the vector index. You can get it from the index structure by lindorm_get_index_mappings tool
    :param top_k: the result number that you want to return
    :return: the most relevant content stored in the knowledgebase.
    """
    lindorm_search_client = ctx.request_context.lifespan_context.lindorm_search_client
    contents = lindorm_search_client.rrf_search(index_name, query, top_k, content_field, vector_field)
    output = f"The retrieving results for query {query} in knowledgebase {index_name} is\n"
    output += "\n".join(f"{i + 1}. {content}" for i, content in enumerate(contents))
    return output


@mcp.tool()
def lindorm_get_index_fields(index_name: str, ctx: Context = None) -> str:
    """
    Get the fields info of the indexes(or knowledgebase), especially get the vector stored field and content stored field.
    :param index_name: the index name, or known as knowledgebase name
    :return: the index fields information
    """
    lindorm_search_client = ctx.request_context.lifespan_context.lindorm_search_client
    mapping = lindorm_search_client.get_index_mappings(index_name)
    fields_info = simplify_mappings(mapping, index_name)
    output = f"The structure(mapping) of index {index_name} is\n"
    output += json.dumps(fields_info, indent=2, ensure_ascii=False)
    return output


@mcp.tool()
def lindorm_list_all_index(ctx: Context = None) -> str:
    """
    List all the indexes(or knowledgebase) you have.
    :return: all the indexes(or knowledgebase) you have
    """
    lindorm_search_client = ctx.request_context.lifespan_context.lindorm_search_client
    all_index = lindorm_search_client.list_indexes()
    output = "All the knowledgebase you have are\n"
    output += "\n".join(f"{i + 1}. {index}" for i, index in enumerate(all_index))
    return output

@mcp.tool()
def lindorm_execute_sql(query: str, ctx: Context = None) -> str:
    """
    Execute SQL query on Lindorm database.
    :param query: The SQL query to execute which start with select
    :return: the results of executing the sql or prompt when meeting certain types of exception
    """
    lindorm_sql_client = ctx.request_context.lifespan_context.lindorm_sql_client
    res = lindorm_sql_client.execute_query(query)
    output = f"The results of executing sql {query} is\n"
    output += res
    return output

@mcp.tool()
def lindorm_show_tables(ctx: Context = None) -> str:
    """
    Get all tables in the Lindorm database
    :return: the tables in the lindorm database
    """
    lindorm_sql_client = ctx.request_context.lifespan_context.lindorm_sql_client
    return lindorm_sql_client.show_tables()

@mcp.tool()
def lindorm_describe_table(table_name: str, ctx: Context = None) -> str:
    """
    Get tables schema in the Lindorm database
    :param table_name: the table name
    :return: the tables schema
    """
    lindorm_sql_client = ctx.request_context.lifespan_context.lindorm_sql_client
    return lindorm_sql_client.describe_table(table_name)


def parse_arguments():
    parser = argparse.ArgumentParser(description="LINDORM MCP Server")
    parser.add_argument("--lindorm_instance_id", type=str, help="Lindorm Search Host")
    parser.add_argument("--using_vpc", type=bool, default=False, help="Whether to use the VPC network")
    parser.add_argument("--username", type=str, help="Lindorm username")
    parser.add_argument("--password", type=str, help="Lindorm password")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-v4", help="Text Embedding Model Name")
    parser.add_argument("--embedding_dimension", type=int, default=1024, help="Text Embedding output dimension")
    parser.add_argument("--database", type=str, default="default", help="The Lindorm Database to execute sql")
    parser.add_argument("--transport", choices=["stdio", "sse"], help="MCP transport protocol")
    parser.add_argument("--host", type=str, help="Host for network transports")
    parser.add_argument("--port", type=int, help="Port for network transports")
    parser.add_argument("--use-ssl", action=argparse.BooleanOptionalAction, default=None,
                        help="Use TLS for Lindorm Search and AI engine connections")
    parser.add_argument("--verify-ssl", action=argparse.BooleanOptionalAction, default=None,
                        help="Verify TLS certificates for Lindorm Search and AI engine connections")
    return parser.parse_args()


def _env_or_arg_bool(env_name: str, arg_value: bool | None, default: bool) -> bool:
    env_value = os.environ.get(env_name)
    if env_value:
        return str_to_bool(env_value)
    if arg_value is not None:
        return arg_value
    return default


def _configure_transport(args):
    transport = os.environ.get("SERVER_TRANSPORT", args.transport or "stdio").replace("_", "-").lower()
    if transport == "streamable-http":
        raise ValueError("streamable-http is not supported by the pinned MCP runtime; use stdio or sse")
    if transport not in {"stdio", "sse"}:
        raise ValueError(f"Unsupported transport: {transport}")

    host = os.environ.get("SERVER_HOST", args.host or "127.0.0.1")
    port = int(os.environ.get("SERVER_PORT", args.port or 8000))
    api_key = os.environ.get("API_KEY")

    if transport != "stdio":
        if not api_key:
            raise ValueError("API_KEY is required when SERVER_TRANSPORT is not stdio")
        api_key = validate_api_key(api_key)
        if host in {"0.0.0.0", "::"} and not str_to_bool(os.environ.get("ALLOW_PUBLIC_BINDING", "false")):
            raise ValueError("Refusing public bind without ALLOW_PUBLIC_BINDING=true")

    mcp.settings.host = host
    mcp.settings.port = port
    mcp.api_key = api_key
    return transport


def main():
    load_dotenv()
    args = parse_arguments()
    instance_id = os.environ.get("LINDORM_INSTANCE_ID", args.lindorm_instance_id)
    using_vpc_env = os.environ.get("USING_VPC_NETWORK")
    if using_vpc_env is not None:
        using_vpc = str_to_bool(using_vpc_env)
    else:
        using_vpc = args.using_vpc
    username = os.environ.get("USERNAME", args.username)
    password = os.environ.get("PASSWORD", args.password)
    embedding_model = os.environ.get("TEXT_EMBEDDING_MODEL", args.embedding_model)
    embedding_dimension = int(os.environ.get("TEXT_EMBEDDING_DIMENSION", args.embedding_dimension))
    table_database = os.environ.get("TABLE_DATABASE", args.database)
    validate_required([
        ("LINDORM_INSTANCE_ID", instance_id),
        ("USERNAME", username),
        ("PASSWORD", password),
        ("TEXT_EMBEDDING_MODEL", embedding_model),
        ("TABLE_DATABASE", table_database),
    ])
    instance_id = validate_lindorm_instance_id(instance_id)
    embedding_model = validate_model_name(embedding_model)

    transport = _configure_transport(args)
    mcp.config = {
        "lindorm_search_host": get_lindorm_search_host(instance_id, using_vpc),
        "lindorm_ai_host": get_lindorm_ai_host(instance_id, using_vpc),
        "lindorm_table_host": get_lindorm_table_host(instance_id, using_vpc),
        "username": username,
        "password": password,
        "text_embedding_model": embedding_model,
        "text_embedding_dimension": embedding_dimension,
        "table_database": table_database,
        "use_ssl": _env_or_arg_bool("LINDORM_USE_SSL", args.use_ssl, True),
        "verify_ssl": _env_or_arg_bool("LINDORM_VERIFY_SSL", args.verify_ssl, True),
    }
    mcp.run(transport=transport)



if __name__ == "__main__":
    main()
