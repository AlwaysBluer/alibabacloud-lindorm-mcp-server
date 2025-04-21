import argparse
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator
from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP

from .utils import *
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
        text_embedding_model=config.get("text_embedding_model")
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


mcp = FastMCP("Lindorm", lifespan=server_lifespan, log_level="ERROR")


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
    parser.add_argument("--username", type=str, default="root", help="Lindorm username")
    parser.add_argument("--password", type=str, help="Lindorm password")
    parser.add_argument("--embedding_model", type=str, help="Text Embedding Model Name")
    parser.add_argument("--database", type=str, default="default", help="The Lindorm Database to execute sql")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_arguments()
    instance_id = os.environ.get("LINDORM_INSTANCE_ID", args.lindorm_instance_id)
    using_vpc_env = os.environ.get("USING_VPC_NETWORK")
    if using_vpc_env is not None:
        using_vpc = str_to_bool(using_vpc_env)
    else:
        using_vpc = args.using_vpc
    mcp.config = {
        "lindorm_search_host": get_lindorm_search_host(instance_id, using_vpc),
        "lindorm_ai_host": get_lindorm_ai_host(instance_id, using_vpc),
        "lindorm_table_host": get_lindorm_table_host(instance_id, using_vpc),
        "username": os.environ.get("USERNAME", args.username),
        "password": os.environ.get("PASSWORD", args.password),
        "text_embedding_model": os.environ.get("TEXT_EMBEDDING_MODEL", args.embedding_model),
        "table_database": os.environ.get("TABLE_DATABASE", args.database)
    }
    mcp.run()



if __name__ == "__main__":
    main()