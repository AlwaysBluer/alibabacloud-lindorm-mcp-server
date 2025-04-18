import argparse
import json
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator
from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP

from utils import get_lindorm_ai_host, get_lindorm_search_host, str_to_bool
from lindorm_vector_search import LindormVectorSearchClient


class LindormContext:
    def __init__(self, lindorm_search_client: LindormVectorSearchClient):
        self.lindorm_search_client = lindorm_search_client


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

    try:
        yield LindormContext(vector_search_client)
    finally:
        pass


mcp = FastMCP("Lindorm", lifespan=server_lifespan, log_level="ERROR")


@mcp.tool()
def lindorm_retrieve_from_index(index_name: str, query: str, top_k: int = 5, content_field: str = "content",
                                vector_field: str = "vector_field", ctx: Context = None) -> str:
    """
    Retrieve from an existing indexes(or knowledgebase) using both full-text search and vector search, and return the aggregated results
    :param index_name: the index name, or known as knowledgebase name
    :param query: the query that you want to search in knowledgebase
    :param top_k: the result that you want to return
    :param content_field: the text field that store the content text. You can get it from the index structure
    :param vector_field: the vector field that store the vector index. You can get it from the index structure
    :return: the most relevant content stored in the knowledgebase.
    """
    lindorm_search_client = ctx.request_context.lifespan_context.lindorm_search_client
    contents = lindorm_search_client.rrf_search(index_name, query, top_k, content_field, vector_field)
    output = f"The retrieving results for query {query} in knowledgebase {index_name} is\n"
    output += "\n".join(f"{i + 1}. {content}" for i, content in enumerate(contents))
    return output


@mcp.tool()
def lindorm_get_index_mappings(index_name: str, ctx: Context = None) -> str:
    """
    Get the structure of the indexes(or knowledgebase), especially get the vector stored field and content stored field.
    :param index_name: the index name, or known as knowledgebase name
    :return: the index structure in json format
    """
    lindorm_search_client = ctx.request_context.lifespan_context.lindorm_search_client
    mapping = lindorm_search_client.get_index_mappings(index_name)
    output = f"The structure(mapping) of index {index_name} is\n"
    output += json.dumps(mapping, indent=2, ensure_ascii=False)
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


def parse_arguments():
    parser = argparse.ArgumentParser(description="LINDORM MCP Server")
    parser.add_argument("--lindorm_instance_id", type=str, help="Lindorm Search Host")
    parser.add_argument("--using_vpc", type=bool, default=False, help="Whether to use the VPC network")
    parser.add_argument("--username", type=str, default="root", help="Lindorm username")
    parser.add_argument("--password", type=str, help="Lindorm password")
    parser.add_argument("--embedding_model", type=str, help="Text Embedding Model Name")
    return parser.parse_args()


if __name__ == "__main__":
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
        "username": os.environ.get("USERNAME", args.username),
        "password": os.environ.get("PASSWORD", args.password),
        "text_embedding_model": os.environ.get("TEXT_EMBEDDING_MODEL", args.embedding_model)
    }
    mcp.run()
